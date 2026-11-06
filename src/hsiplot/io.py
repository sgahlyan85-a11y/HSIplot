"""Reading and writing spectra tables.

Table layout (one row per spectrum, CSV or XLSX)::

    plot_id, group, [extra plot-level columns ...], wl_440.00, wl_444.84, ..., wl_900.00

* ``plot_id``  identifies the experimental unit (plot, pot, tray). All validation holds out whole plots.
* ``group``    the treatment label; one value is the control label (default ``control``), anything else counts as treated.
* extra plot-level columns (biomass, cover, dose ...) must be constant within a plot.
* band columns are named by wavelength in nm (``wl_<nm>``). Legacy names (``Band_01``, ``Reflectance_01`` ...) are
  accepted if the wavelength range is given; the wavelengths are then assumed to be evenly spaced.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("hsiplot")

_WL = re.compile(r"^(?:wl|nm)[_]?(\d+(?:\.\d+)?)$", re.I)
_LEGACY = re.compile(r"^(?:band|reflectance|radiance|b)[_ ]?(\d+)$", re.I)


@dataclass
class SpectraTable:
    """Spectra with plot and group labels.

    ``df`` holds ``plot_id``, ``group``, ``y`` (0 = control, 1 = treated), extra plot-level columns and the band columns.
    """
    df: pd.DataFrame
    bands: List[str]
    wavelengths: np.ndarray
    quantity: str = "unknown"          # 'radiance', 'reflectance' or 'unknown'
    control_label: str = "control"
    meta: Dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.df[self.bands].to_numpy(float)

    @property
    def y(self) -> np.ndarray:
        return self.df["y"].to_numpy()

    @property
    def plots(self) -> np.ndarray:
        return self.df["plot_id"].to_numpy()

    def plot_table(self) -> pd.DataFrame:
        """One row per plot: label and number of spectra."""
        g = self.df.groupby("plot_id")
        out = pd.DataFrame({"y": g["y"].first(), "group": g["group"].first(), "n_spectra": g.size()})
        return out.reset_index()

    def with_X(self, X: np.ndarray, keep: Optional[np.ndarray] = None) -> "SpectraTable":
        """Copy with spectra replaced by ``X`` (rows ``keep`` of the original)."""
        df = self.df if keep is None else self.df.loc[keep]
        df = df.reset_index(drop=True).copy()
        df[self.bands] = X
        return SpectraTable(df, list(self.bands), self.wavelengths, self.quantity, self.control_label, dict(self.meta))

    def select_bands(self, nm_min: Optional[float] = None, nm_max: Optional[float] = None) -> "SpectraTable":
        lo = -np.inf if nm_min is None else nm_min
        hi = np.inf if nm_max is None else nm_max
        m = (self.wavelengths >= lo) & (self.wavelengths <= hi)
        if m.sum() < 5:
            raise ValueError("fewer than 5 bands left after band selection")
        bands = [b for b, k in zip(self.bands, m) if k]
        keep_cols = [c for c in self.df.columns if c not in self.bands] + bands
        return SpectraTable(self.df[keep_cols].copy(), bands, self.wavelengths[m], self.quantity,
                            self.control_label, dict(self.meta))

    def select_plots(self, exclude: Sequence = ()) -> "SpectraTable":
        if not len(exclude):
            return self
        df = self.df[~self.df["plot_id"].isin(list(exclude))].reset_index(drop=True)
        return SpectraTable(df, list(self.bands), self.wavelengths, self.quantity, self.control_label, dict(self.meta))

    def subsample(self, max_per_plot: int, seed: int = 0) -> "SpectraTable":
        """Random subset of at most ``max_per_plot`` spectra from every plot (speed)."""
        if not max_per_plot:
            return self
        rng = np.random.default_rng(seed)
        idx = []
        for p in pd.unique(self.plots):
            ix = np.flatnonzero(self.plots == p)
            idx.append(rng.permutation(ix)[:max_per_plot])
        idx = np.sort(np.concatenate(idx))
        return SpectraTable(self.df.iloc[idx].reset_index(drop=True), list(self.bands), self.wavelengths,
                            self.quantity, self.control_label, dict(self.meta))


def _band_columns(cols: Sequence[str], wavelength_range: Optional[Tuple[float, float]]) -> Tuple[List[str], np.ndarray, bool]:
    wl = [(c, float(m.group(1))) for c in cols if (m := _WL.match(c))]
    if len(wl) >= 5:
        wl.sort(key=lambda t: t[1])
        return [c for c, _ in wl], np.array([w for _, w in wl]), False
    leg = [(c, int(m.group(1))) for c in cols if (m := _LEGACY.match(c))]
    if len(leg) >= 5:
        leg.sort(key=lambda t: t[1])
        names = [c for c, _ in leg]
        if wavelength_range is None:
            raise ValueError("legacy band names (Band_01 ...) carry no wavelengths: pass wavelength_range=(min_nm, max_nm)")
        return names, np.linspace(wavelength_range[0], wavelength_range[1], len(names)), True
    raise ValueError("no band columns found; expected names like 'wl_440.00' or 'Band_01'")


def load_table(
    path,
    plot_col: str = "plot_id",
    group_col: str = "group",
    control_label: str = "control",
    rows_per_plot: Optional[int] = None,
    wavelength_range: Optional[Tuple[float, float]] = None,
    quantity: Optional[str] = None,
) -> SpectraTable:
    """Read a spectra table (CSV or XLSX).

    If the table has no plot column, ``rows_per_plot`` assigns plots in consecutive blocks of that many rows.
    ``quantity`` overrides the value stored in the ``<table>.meta.json`` sidecar written by :func:`build_tables`.
    """
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    plot_col, group_col = plot_col.replace(" ", "_"), group_col.replace(" ", "_")
    bands, wl, assumed = _band_columns(list(df.columns), wavelength_range)
    if assumed:
        log.warning("band names carry no wavelengths; assuming %d evenly spaced bands from %.0f to %.0f nm",
                    len(bands), wl[0], wl[-1])
    if plot_col not in df.columns:
        if not rows_per_plot:
            raise ValueError(f"no '{plot_col}' column; pass rows_per_plot to infer plots from blocks of rows")
        if len(df) % rows_per_plot:
            raise ValueError(f"{len(df)} rows is not a multiple of rows_per_plot={rows_per_plot}")
        df[plot_col] = np.arange(len(df)) // rows_per_plot + 1
    if group_col not in df.columns:
        raise ValueError(f"no '{group_col}' column; columns are {list(df.columns[:8])} ...")
    g = df[group_col]
    if pd.api.types.is_numeric_dtype(g):
        y = (g != 0).astype(int)
    else:
        s = g.astype(str).str.strip().str.lower()
        if control_label.strip().lower() not in set(s):
            raise ValueError(f"control label '{control_label}' not found in {sorted(s.unique())}")
        y = (s != control_label.strip().lower()).astype(int)
    if y.nunique() != 2:
        raise ValueError("need both control and treated spectra")
    out = df.rename(columns={plot_col: "plot_id", group_col: "group"}).copy()
    out["y"] = y.to_numpy()
    # plot-level consistency checks
    nun = out.groupby("plot_id")["y"].nunique()
    if (nun > 1).any():
        raise ValueError(f"plots with mixed labels: {list(nun[nun > 1].index)}")
    ordered = ["plot_id", "group", "y"] + [c for c in out.columns if c not in bands + ["plot_id", "group", "y"]] + bands
    out = out[ordered].copy()
    meta: Dict = {}
    side = path.with_suffix(".meta.json")
    if side.exists():
        meta = json.loads(side.read_text())
    q = quantity or meta.get("quantity", "unknown")
    return SpectraTable(out, bands, wl, q, control_label, meta)


def build_tables(
    metadata_csv,
    spectra_dir,
    out_dir,
    ndvi_min: Optional[float] = None,
    plot_col: str = "plot_id",
    file_col: str = "npz",
    group_col: str = "group",
    subset_col: Optional[str] = "subset",
) -> List[Path]:
    """Combine per-image ``.npz`` files (from :func:`hsiplot.extract.extract_to_npz`) into tables, one per subset.

    ``metadata_csv`` has one row per image: ``plot_id, npz, group`` plus optional ``subset`` (for example a species
    or experiment) and any plot-level columns (biomass, cover ...) which are carried into the table.
    """
    from .preprocess import ndvi

    meta = pd.read_csv(metadata_csv)
    need = {plot_col, file_col, group_col}
    if need - set(meta.columns):
        raise ValueError(f"metadata is missing columns {sorted(need - set(meta.columns))}")
    extra = [c for c in meta.columns if c not in need | {subset_col}]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = meta.groupby(subset_col) if subset_col and subset_col in meta.columns else [("dataset", meta)]
    written = []
    for name, g in groups:
        parts, qty, wls = [], set(), []
        for r in g.sort_values(plot_col).to_dict("records"):
            d = np.load(Path(spectra_dir) / r[file_col], allow_pickle=True)
            qty.add(str(d["quantity"]))
            wl = np.asarray(d["wavelengths"], float)
            wls.append(wl)
            S = np.asarray(d["spectra"], float)
            if ndvi_min is not None:
                keep = ndvi(S, wl) >= ndvi_min
                log.info("plot %s: keeping %d/%d spectra (NDVI >= %s)", r[plot_col], keep.sum(), len(keep), ndvi_min)
                S = S[keep]
            part = pd.DataFrame(S, columns=[f"wl_{w:.2f}" for w in wl])
            for c in reversed(extra):
                part.insert(0, c, r[c])
            part.insert(0, "group", r[group_col])
            part.insert(0, "plot_id", r[plot_col])
            parts.append(part)
        if len(qty) != 1:
            raise ValueError(f"{name}: mixed radiance/reflectance files {sorted(qty)}")
        if not all(np.allclose(wls[0], w, atol=0.05) for w in wls):
            raise ValueError(f"{name}: images have different wavelength grids (different cameras or calibrations)")
        full = pd.concat(parts, ignore_index=True)
        q = qty.pop()
        p = out_dir / f"{name}_{q}.csv"
        full.to_csv(p, index=False)
        p.with_suffix(".meta.json").write_text(json.dumps(
            {"quantity": q, "n_plots": int(full.plot_id.nunique()), "n_spectra": int(len(full))}, indent=1))
        log.info("%s: %d spectra, %d plots, %s -> %s", name, len(full), full.plot_id.nunique(), q, p)
        written.append(p)
    return written
