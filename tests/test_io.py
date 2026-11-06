import numpy as np
import pandas as pd
import pytest

from hsiplot.io import build_tables, load_table
from hsiplot.synthetic import make_synthetic_table


def test_roundtrip_and_wavelengths(tmp_path):
    t = make_synthetic_table(n_plots=6, n_spectra=10, n_bands=20)
    p = tmp_path / "t.csv"
    t.df.drop(columns="y").to_csv(p, index=False)
    r = load_table(p)
    assert len(r.bands) == 20 and np.allclose(r.wavelengths, t.wavelengths, atol=0.01)
    assert set(r.df.y.unique()) == {0, 1} and "biomass" in r.df.columns


def test_legacy_band_names_need_range(tmp_path):
    df = pd.DataFrame(np.random.rand(40, 10) + 1, columns=[f"Reflectance_{i:02d}" for i in range(1, 11)])
    df["Herbicide"] = ["Control"] * 10 + ["Treatment"] * 30
    df["Plot_ID"] = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
    p = tmp_path / "legacy.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="wavelength_range"):
        load_table(p, "Plot_ID", "Herbicide", "Control")
    t = load_table(p, "Plot_ID", "Herbicide", "Control", wavelength_range=(440, 900))
    assert t.wavelengths[0] == 440 and t.wavelengths[-1] == 900 and t.df.y.tolist().count(0) == 10


def test_rows_per_plot_inference_and_errors(tmp_path):
    df = pd.DataFrame(np.random.rand(40, 6) + 1, columns=[f"Band_{i}" for i in range(1, 7)])
    df["group"] = ["control"] * 10 + ["treated"] * 30
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="rows_per_plot"):
        load_table(p, wavelength_range=(400, 900))
    assert load_table(p, rows_per_plot=10, wavelength_range=(400, 900)).df.plot_id.nunique() == 4
    with pytest.raises(ValueError, match="multiple"):
        load_table(p, rows_per_plot=7, wavelength_range=(400, 900))
    with pytest.raises(ValueError, match="control label"):
        load_table(p, rows_per_plot=10, control_label="ctrl", wavelength_range=(400, 900))


def test_mixed_labels_in_one_plot_rejected(tmp_path):
    t = make_synthetic_table(n_plots=4, n_control=2, n_spectra=10, n_bands=10)
    d = t.df.drop(columns="y").copy()
    d.loc[0, "group"] = "treated" if d.loc[0, "group"] == "control" else "control"
    p = tmp_path / "m.csv"
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="mixed labels"):
        load_table(p)


def _npz(path, n=50, wl=None, quantity="radiance", seed=0, nir=1.0):
    wl = np.linspace(440, 900, 20) if wl is None else wl
    rng = np.random.default_rng(seed)
    S = (1e-4 * (1 + (wl > 700) * nir) * rng.lognormal(0, 0.05, (n, 1)) + rng.normal(0, 1e-6, (n, len(wl)))).astype("float32")
    np.savez(path, spectra=S, coords=rng.integers(0, 100, (n, 2)), wavelengths=wl, quantity=quantity)


def test_build_tables(tmp_path):
    for i in range(1, 5):
        _npz(tmp_path / f"p{i}.npz", seed=i, nir=1.0 if i > 1 else 0.1)
    meta = pd.DataFrame({"plot_id": [1, 2, 3, 4], "npz": [f"p{i}.npz" for i in range(1, 5)],
                         "group": ["control", "treated", "treated", "control"], "subset": ["a", "a", "b", "b"],
                         "biomass": [1., 2., 3., 4.]})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    out = build_tables(tmp_path / "meta.csv", tmp_path, tmp_path / "out", ndvi_min=0.0)
    assert {p.name for p in out} == {"a_radiance.csv", "b_radiance.csv"}
    t = load_table(out[0])
    assert t.quantity == "radiance" and "biomass" in t.df.columns and t.df.plot_id.nunique() == 2


def test_build_rejects_mixed_quantity_and_grids(tmp_path):
    _npz(tmp_path / "a.npz", quantity="radiance")
    _npz(tmp_path / "b.npz", quantity="reflectance", seed=1)
    pd.DataFrame({"plot_id": [1, 2], "npz": ["a.npz", "b.npz"], "group": ["control", "treated"]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(ValueError, match="mixed"):
        build_tables(tmp_path / "m.csv", tmp_path, tmp_path / "o")
    _npz(tmp_path / "c.npz", wl=np.linspace(450, 910, 20))
    pd.DataFrame({"plot_id": [1, 2], "npz": ["a.npz", "c.npz"], "group": ["control", "treated"]}).to_csv(tmp_path / "m2.csv", index=False)
    with pytest.raises(ValueError, match="wavelength grids"):
        build_tables(tmp_path / "m2.csv", tmp_path, tmp_path / "o2")
