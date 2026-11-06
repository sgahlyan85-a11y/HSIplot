"""Command-line interface: ``hsiplot <command> --help``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__
from .io import SpectraTable, build_tables, load_table
from .preprocess import PreprocessConfig, band_noise_report, plot_qc, preprocess_table

log = logging.getLogger("hsiplot")


def _table_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("table")
    g.add_argument("table", help="spectra table (.csv or .xlsx)")
    g.add_argument("--plot-col", default="plot_id")
    g.add_argument("--group-col", default="group")
    g.add_argument("--control-label", default="control", help="value of the group column that marks control plots")
    g.add_argument("--rows-per-plot", type=int, help="infer plot ids from consecutive blocks of this many rows (if no plot column)")
    g.add_argument("--wavelength-range", type=float, nargs=2, metavar=("MIN_NM", "MAX_NM"),
                   help="for legacy band names (Band_01 ...): assumed evenly spaced range")
    g.add_argument("--quantity", choices=["radiance", "reflectance"], help="override the quantity stored with the table")
    g.add_argument("--exclude-plots", nargs="*", default=[], help="plot ids to drop (report the decision)")
    g.add_argument("--band-range", type=float, nargs=2, metavar=("MIN_NM", "MAX_NM"), help="use only bands in this range")
    g.add_argument("--max-per-plot", type=int, default=300, help="spectra per plot used by the model (speed); 0 = all")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-jobs", type=int, default=1, help="parallel workers (-1 = all cores)")
    g = p.add_argument_group("preprocessing")
    g.add_argument("--preset", choices=["none", "clean", "clean_snv"], default="clean")
    g.add_argument("--ndvi-min", type=float, help="drop spectra below this NDVI (soil, shadow)")
    g.add_argument("--refl-max", type=float, help="reflectance data only: drop spectra with any band above this")


def _load(a) -> SpectraTable:
    t = load_table(a.table, a.plot_col, a.group_col, a.control_label, a.rows_per_plot,
                   tuple(a.wavelength_range) if a.wavelength_range else None, a.quantity)
    ex = [int(x) for x in a.exclude_plots] if t.plots.dtype.kind in "iu" else list(a.exclude_plots)
    t = t.select_plots(ex)
    if a.band_range:
        t = t.select_bands(*a.band_range)
    return t


def _prep(a, t: SpectraTable) -> SpectraTable:
    cfg = PreprocessConfig.preset(a.preset)
    cfg.ndvi_min, cfg.reflectance_max = a.ndvi_min, a.refl_max
    return preprocess_table(t, cfg)


def cmd_analyze(a) -> int:
    from .permanova import group_difference_test
    from .validate import band_importance, random_split_demo, run_classification
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.table).stem
    t = _load(a)
    qc = plot_qc(t)
    qc.to_csv(out / f"{name}_qc.csv", index=False)
    band_noise_report(t).to_csv(out / f"{name}_band_noise.csv", index=False)
    if (qc.flag == "CHECK").any():
        flagged = qc[qc.flag == "CHECK"].to_string(index=False)
        log.warning("plots flagged CHECK (exposure or invalid spectra) - see %s_qc.csv:\n%s", name, flagged)
    t = _prep(a, t).subsample(a.max_per_plot, a.seed)
    log.info("%d spectra, %d plots (%d control)", len(t.df), t.df.plot_id.nunique(), t.plot_table().eval("y == 0").sum())
    res = run_classification(t, a.scheme, a.model, a.trees, a.n_perm, a.max_perm, a.seed, a.n_jobs)
    summ = {"classification": res.summary(), "group_difference_permanova": group_difference_test(t, seed=a.seed).summary()}
    summ.update(name=name, preset=a.preset, quantity=t.quantity,
                random_split_balanced_accuracy_INVALID=round(random_split_demo(t, a.model, a.trees, a.seed), 3))
    if t.quantity == "unknown":
        log.warning("quantity (radiance or reflectance) is unknown for this table; pass --quantity so results are labelled correctly")
    res.plot_scores.to_csv(out / f"{name}_plot_scores.csv", index=False)
    band_importance(t, seed=a.seed).to_csv(out / f"{name}_band_importance.csv", index=False)
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    _figure(t, res, out / f"{name}_figure.png")
    print(json.dumps(summ, indent=1, default=str))
    print(res.plot_scores.to_string(index=False))
    return 0


def _figure(t: SpectraTable, res, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mean = t.df.groupby("plot_id")[t.bands].mean()
    y = t.df.groupby("plot_id")["y"].first()
    for pid, row in mean.iterrows():
        ax[0].plot(t.wavelengths, row.values, color="tab:red" if y[pid] == 0 else "tab:gray", lw=1.5 if y[pid] == 0 else 0.8)
    ax[0].set(xlabel="wavelength (nm)", ylabel=t.quantity, title="plot-mean spectra (red = control)")
    s = res.plot_scores.sort_values("mean_p_treated")
    ax[1].barh(s.plot_id.astype(str), s.mean_p_treated, color=["tab:red" if v == 0 else "tab:gray" for v in s.y])
    ax[1].axvline(0.5, color="k", lw=0.5)
    ax[1].set(xlabel="mean P(treated) when the plot was held out", ylabel="plot",
              title=f"classifier plot AUC {res.statistic:.2f}, p = {res.p_value if res.p_value is None else round(res.p_value, 3)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_trait(a) -> int:
    from .trait import run_trait_model
    t = _prep(a, _load(a))
    res = run_trait_model(t, a.target, a.trees, a.n_perm, a.seed, a.n_jobs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.table).stem
    res.predictions.to_csv(out / f"{name}_{a.target}_predictions.csv", index=False)
    (out / f"{name}_{a.target}_summary.json").write_text(json.dumps(res.summary(), indent=1))
    print(json.dumps(res.summary(), indent=1))
    return 0


def cmd_power(a) -> int:
    from .power import detectability
    t = _prep(a, _load(a)).subsample(a.max_per_plot, a.seed)
    df = detectability(t, a.effects, a.nm_from, a.nm_to, test=a.test, scheme=a.scheme, model=a.model, n_estimators=a.trees,
                       n_perm=a.n_perm, max_perm=a.max_perm, seed=a.seed, n_jobs=a.n_jobs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.name or Path(a.table).stem}_detectability.csv", index=False)
    print(df.to_string(index=False))
    return 0


def cmd_demo(a) -> int:
    from .synthetic import make_synthetic_table
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for label, eff in (("with_effect", a.effect), ("no_effect", 0.0)):
        t = make_synthetic_table(n_plots=8, n_control=3, n_spectra=60, n_bands=48, effect=eff, plot_sd=0.05, seed=a.seed)
        p = out / f"synthetic_{label}.csv"
        t.df.drop(columns="y").to_csv(p, index=False)
        print(f"\n=== synthetic data, {label} (NIR change in treated plots: {eff:.0%}) ===")
        ns = argparse.Namespace(**{**vars(a), "table": str(p), "plot_col": "plot_id", "group_col": "group", "control_label": "control",
                                   "rows_per_plot": None, "wavelength_range": None, "quantity": "radiance", "exclude_plots": [],
                                   "band_range": None, "max_per_plot": 60, "preset": "clean", "ndvi_min": None, "refl_max": None,
                                   "name": f"demo_{label}", "scheme": "paired", "model": "rf", "trees": 30, "n_perm": None,
                                   "max_perm": 500, "n_jobs": a.n_jobs})
        cmd_analyze(ns)
    return 0


def cmd_check(a) -> int:
    from .extract import check_file
    rep = check_file(a.lo_file, a.calibration_folder, a.frame, a.field_calibration)
    for name, status in rep["steps"]:
        print(f"[{'ok' if status == 'ok' else 'FAIL'}] {name}" + ("" if status == "ok" else f"  {status}"))
    for k, v in rep.items():
        if k not in ("steps", "hint"):
            print(f"  {k}: {v}")
    if "hint" in rep:
        print(rep["hint"])
    return 0 if all(s == "ok" for _, s in rep["steps"]) else 1


def cmd_extract(a) -> int:
    from .extract import extract_to_npz
    extract_to_npz(a.lo_file, a.out, a.calibration_folder, a.frame, a.field_calibration, a.white_reference, a.dark_reference)
    return 0


def cmd_reference(a) -> int:
    from .calibrate import white_reference
    z = np.load(a.panel_npz, allow_pickle=True)
    S = z["spectra"].astype(float)
    if a.dark:
        S = S - np.load(a.dark)
    w, rep = white_reference(S, z["coords"] if "coords" in z.files else None, a.roi, a.auto_bright, a.panel_reflectance)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, w.astype(np.float32))
    out.with_name(out.stem + "_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["warnings"]:
        print("WARNING: do not use this reference until the warnings are resolved", file=sys.stderr)
        return 2
    return 0


def cmd_build(a) -> int:
    build_tables(a.metadata, a.spectra_dir, a.out_dir, a.ndvi_min)
    return 0


def cmd_convert(a) -> int:
    import csv

    import openpyxl
    wb = openpyxl.load_workbook(a.xlsx, read_only=True, data_only=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        for row in wb.active.iter_rows(values_only=True):
            w.writerow(row)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="hsiplot", description="Plot-level validation of hyperspectral spectra.")
    ap.add_argument("--version", action="version", version=f"hsiplot {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p, perm=True):
        g = p.add_argument_group("model")
        g.add_argument("--scheme", choices=["paired", "lopo"], default="paired", help="whole-plot cross-validation scheme")
        g.add_argument("--model", choices=["rf", "logreg"], default="rf")
        g.add_argument("--trees", type=int, default=100)
        g.add_argument("--n-perm", type=int, default=None, help="permutations; default all labellings up to --max-perm; 0 = none")
        g.add_argument("--max-perm", type=int, default=500)
        p.add_argument("--out", default="results")
        p.add_argument("--name", default=None)

    p = sub.add_parser("analyze", help="plot-level classification: group separation with a permutation test")
    _table_args(p)
    model_args(p)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("trait", help="plot-level regression of a plot-level trait (biomass, cover ...)")
    _table_args(p)
    p.add_argument("--target", required=True)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--n-perm", type=int, default=199)
    p.add_argument("--out", default="results")
    p.add_argument("--name", default=None)
    p.set_defaults(fn=cmd_trait)

    p = sub.add_parser("power", help="smallest spectral effect the design can detect")
    _table_args(p)
    model_args(p)
    p.add_argument("--effects", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.4])
    p.add_argument("--nm-from", type=float, default=750.0)
    p.add_argument("--nm-to", type=float, default=None)
    p.add_argument("--test", choices=["permanova", "classifier"], default="permanova",
                   help="permanova: fast plot-level test; classifier: full cross-validated classification (slow)")
    p.set_defaults(fn=cmd_power)

    p = sub.add_parser("demo", help="run the analysis on synthetic data (no SDK, no real data)")
    p.add_argument("--effect", type=float, default=0.5, help="relative NIR increase in treated plots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/demo")
    p.add_argument("--n-jobs", type=int, default=1)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("check", help="step-by-step check of the Living Optics SDK and one .lo file")
    p.add_argument("lo_file")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--calibration-folder", help="factory calibration folder (needed for the decode step)")
    p.add_argument("--field-calibration", help="optional field calibration file")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("extract", help="decode one .lo frame to an .npz (radiance, or reflectance with --white-reference)")
    p.add_argument("lo_file")
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--calibration-folder", required=True, help="factory calibration folder supplied by Living Optics for your camera")
    p.add_argument("--field-calibration", help="optional field calibration file")
    p.add_argument("--white-reference")
    p.add_argument("--dark-reference")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("reference", help="build a white-panel reference from an extracted panel image")
    p.add_argument("panel_npz")
    p.add_argument("--out", required=True)
    p.add_argument("--roi", type=float, nargs=4, metavar=("ROW0", "ROW1", "COL0", "COL1"))
    p.add_argument("--auto-bright", type=float, help="use spectra above this brightness percentile")
    p.add_argument("--panel-reflectance", type=float, default=0.99)
    p.add_argument("--dark")
    p.set_defaults(fn=cmd_reference)

    p = sub.add_parser("build", help="combine per-image .npz files into spectra tables")
    p.add_argument("--metadata", required=True)
    p.add_argument("--spectra-dir", required=True)
    p.add_argument("--out-dir", default="data/tables")
    p.add_argument("--ndvi-min", type=float)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("convert", help="stream a large .xlsx to .csv with low memory")
    p.add_argument("xlsx")
    p.add_argument("csv")
    p.set_defaults(fn=cmd_convert)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return a.fn(a)
    except (ValueError, ImportError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
