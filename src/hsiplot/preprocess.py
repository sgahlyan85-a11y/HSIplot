"""Spectrum-level quality control and noise handling.

Order (every step optional, every step logged):

1. invalid values (NaN, zero, negative: radiance and reflectance cannot be <= 0 in a valid measurement)
2. extreme brightness within each plot (deep shadow = low signal; very bright = saturation or glint)
3. roughness within each plot (band-to-band jitter far above that plot's typical level)
4. reflectance ceiling (reflectance data only)
5. vegetation mask by NDVI (soil, shadow)
6. Savitzky-Golay smoothing along wavelength
7. SNV, standard normal variate per spectrum (removes overall brightness; also removes brightness information)

All thresholds are applied **within each plot**, so a plot that is globally darker or noisier than the others is not
removed wholesale. These steps use no group labels, so they cannot leak the treatment into the validation.
Per-band standardisation is *not* done here: it belongs inside the cross-validation loop (see ``validate``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .io import SpectraTable

log = logging.getLogger("hsiplot")


def band_index(wavelengths: np.ndarray, nm: float) -> int:
    return int(np.abs(np.asarray(wavelengths, float) - nm).argmin())


def ndvi(X: np.ndarray, wavelengths: np.ndarray, red_nm: float = 670.0, nir_nm: float = 800.0) -> np.ndarray:
    """NDVI = (NIR - red) / (NIR + red) (Rouse et al., 1974). Approximate when computed from radiance."""
    r, n = band_index(wavelengths, red_nm), band_index(wavelengths, nir_nm)
    return (X[:, n] - X[:, r]) / (X[:, n] + X[:, r] + 1e-12)


@dataclass
class PreprocessConfig:
    invalid: bool = True
    brightness_percentiles: Optional[Tuple[float, float]] = (1.0, 99.9)
    roughness_mad: Optional[float] = 6.0
    reflectance_max: Optional[float] = None
    ndvi_min: Optional[float] = None
    smooth: Optional[Tuple[int, int]] = (9, 2)        # (window, polyorder)
    snv: bool = False
    min_per_plot_for_limits: int = 20

    @classmethod
    def preset(cls, name: str) -> "PreprocessConfig":
        """``none`` (nothing), ``clean`` (default), ``clean_snv`` (clean + SNV)."""
        if name == "none":
            return cls(False, None, None, None, None, None, False)
        if name == "clean":
            return cls()
        if name == "clean_snv":
            return cls(snv=True)
        raise ValueError(f"unknown preset {name!r}")


def _per_plot_limits(values: np.ndarray, groups: np.ndarray, keep: np.ndarray, fn, min_n: int) -> np.ndarray:
    bad = np.zeros(len(values), bool)
    for k in pd.unique(groups):
        ix = (groups == k) & keep
        if ix.sum() < min_n:
            continue
        bad |= fn(values, ix)
    return bad


def preprocess_arrays(X: np.ndarray, groups: np.ndarray, wavelengths: np.ndarray, cfg: PreprocessConfig,
                      quantity: str = "unknown") -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    """Return ``(X_clean, keep_mask, log)``; ``X_clean`` has ``keep_mask.sum()`` rows."""
    X = np.asarray(X, float)
    n = len(X)
    keep = np.ones(n, bool)
    steps: List[Tuple[str, int]] = []
    g = np.asarray(groups)

    if cfg.invalid:
        bad = ~np.isfinite(X).all(1) | (X <= 0).any(1)
        keep &= ~bad
        steps.append(("invalid values (NaN, zero or negative)", int(bad.sum())))

    if cfg.brightness_percentiles:
        lo_p, hi_p = cfg.brightness_percentiles
        b = np.where(keep, np.nanmean(np.where(np.isfinite(X), X, np.nan), 1), np.nan)

        def fn(v, ix):
            lo, hi = np.percentile(v[ix], [lo_p, hi_p])
            return ix & ((v < lo) | (v > hi))
        bad = _per_plot_limits(b, g, keep, fn, cfg.min_per_plot_for_limits)
        keep &= ~bad
        steps.append((f"brightness outside the {lo_p}-{hi_p} percentiles of its plot", int(bad.sum())))

    if cfg.roughness_mad:
        filler = np.nanmedian(X[keep], 0) if keep.any() else np.zeros(X.shape[1])
        sm = savgol_filter(np.where(keep[:, None], X, filler), 9, 2, axis=1)
        rough = np.abs(X - sm).mean(1) / (np.abs(sm).mean(1) + 1e-12)

        def fn2(v, ix):
            r = v[ix]
            med = np.median(r)
            mad = np.median(np.abs(r - med)) + 1e-12
            return ix & (v > med + cfg.roughness_mad * 1.4826 * mad)
        bad = _per_plot_limits(rough, g, keep, fn2, cfg.min_per_plot_for_limits)
        keep &= ~bad
        steps.append((f"roughness above median + {cfg.roughness_mad} MAD within its plot", int(bad.sum())))

    if cfg.reflectance_max is not None:
        if quantity != "reflectance":
            raise ValueError("reflectance_max is only valid for reflectance data (quantity='reflectance')")
        bad = keep & (X > cfg.reflectance_max).any(1)
        keep &= ~bad
        steps.append((f"reflectance above {cfg.reflectance_max} in any band", int(bad.sum())))

    if cfg.ndvi_min is not None:
        bad = keep & (ndvi(X, wavelengths) < cfg.ndvi_min)
        keep &= ~bad
        steps.append((f"NDVI below {cfg.ndvi_min} (soil, shadow)", int(bad.sum())))

    if not keep.any():
        raise ValueError("preprocessing removed every spectrum; relax the limits (see the log of removed spectra per step)")
    Xc = X[keep]
    if cfg.smooth:
        w, o = cfg.smooth
        Xc = savgol_filter(Xc, w, o, axis=1)
    if cfg.snv:
        Xc = (Xc - Xc.mean(1, keepdims=True)) / (Xc.std(1, keepdims=True) + 1e-12)
    for name, c in steps:
        log.info("preprocess: removed %6d  %s", c, name)
    log.info("preprocess: %d of %d spectra kept (%.1f %%); smoothing=%s snv=%s", keep.sum(), n, 100 * keep.mean(), cfg.smooth, cfg.snv)
    return Xc, keep, steps


def preprocess_table(table: SpectraTable, cfg: Optional[PreprocessConfig] = None) -> SpectraTable:
    """Apply :class:`PreprocessConfig` to a table. Raises if any plot would lose all its spectra."""
    cfg = cfg or PreprocessConfig()
    Xc, keep, steps = preprocess_arrays(table.X, table.plots, table.wavelengths, cfg, table.quantity)
    out = table.with_X(Xc, keep)
    lost = set(pd.unique(table.plots)) - set(pd.unique(out.plots))
    if lost:
        raise ValueError(f"preprocessing removed every spectrum of plot(s) {sorted(lost)}; inspect plot_qc() and relax the limits "
                         "or exclude those plots explicitly")
    out.meta["preprocess_log"] = steps
    return out


def plot_qc(table: SpectraTable, dark_ratio: float = 0.2, bright_ratio: float = 5.0, max_invalid_pct: float = 5.0) -> pd.DataFrame:
    """Per-plot quality table. A plot far darker or brighter than the median plot, or with many invalid spectra, is flagged CHECK:
    its exposure differs, so comparisons with other plots need care (calibrate to reflectance; or exclude and say so)."""
    X, g = table.X, table.plots
    rows = []
    for k in pd.unique(g):
        x = X[g == k]
        rows.append({"plot_id": k, "n_spectra": len(x), "median_brightness": float(np.median(x.mean(1))),
                     "pct_invalid": float(100 * (~np.isfinite(x).all(1) | (x <= 0).any(1)).mean())})
    q = pd.DataFrame(rows)
    q["brightness_vs_median_plot"] = (q.median_brightness / q.median_brightness.median()).round(2)
    q["flag"] = np.where((q.brightness_vs_median_plot > bright_ratio) | (q.brightness_vs_median_plot < dark_ratio)
                         | (q.pct_invalid > max_invalid_pct), "CHECK", "")
    return q


def band_noise_report(table: SpectraTable, window: int = 9, order: int = 2, max_spectra: int = 200000) -> pd.DataFrame:
    """Per-band signal-to-noise: residual of each band around a Savitzky-Golay fit across wavelength.
    Use it to decide whether bands should be dropped (``SpectraTable.select_bands``)."""
    X = table.X[:max_spectra]
    X = X[np.isfinite(X).all(1) & (X > 0).all(1)]
    sm = savgol_filter(X, window, order, axis=1)
    res = X - sm
    return pd.DataFrame({"wavelength_nm": table.wavelengths.round(1), "mean_signal": sm.mean(0),
                         "snr": sm.mean(0) / res.std(0), "relative_noise": np.abs(res).mean(0) / np.abs(sm).mean(0)})
