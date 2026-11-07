import sys
import types

import numpy as np
import pandas as pd
import pytest

from hsiplot import cli
from hsiplot.extract import check_file, extract_to_npz
from hsiplot.power import detectability
from hsiplot.synthetic import make_synthetic_table
from hsiplot.trait import run_trait_model


def test_trait_model_finds_brightness_driven_trait_and_rejects_bad_input():
    t = make_synthetic_table(n_plots=12, n_control=3, n_spectra=30, n_bands=20, effect=0.0, plot_sd=0.35, seed=5)
    r = run_trait_model(t, "biomass", n_estimators=60, n_perm=19)
    assert r.r2 > 0.3 and r.p_value <= 0.1 and len(r.predictions) == 12
    with pytest.raises(ValueError, match="not found"):
        run_trait_model(t, "nope")
    bad = t.df.copy()
    bad.loc[0, "biomass"] += 5
    with pytest.raises(ValueError, match="constant within a plot"):
        run_trait_model(t.with_X(t.X).__class__(bad, t.bands, t.wavelengths), "biomass")
    with pytest.raises(ValueError, match="fewer than 6"):
        run_trait_model(make_synthetic_table(n_plots=4, n_spectra=20, n_bands=20), "biomass")


def test_detectability_increases_with_effect():
    t = make_synthetic_table(n_plots=8, n_control=2, n_spectra=40, n_bands=20, effect=0.0, plot_sd=0.01, seed=9)
    d = detectability(t, effects=[0.0, 1.0], seed=0)
    assert not d.loc[0, "detected"] and d.loc[1, "detected"] and d.loc[1, "statistic"] > d.loc[0, "statistic"]
    c = detectability(t, effects=[1.0], test="classifier", n_estimators=20, n_perm=0)
    assert c.loc[0, "statistic"] == 1.0
    with pytest.raises(ValueError):
        detectability(t, test="x")


@pytest.fixture
def fake_sdk(monkeypatch):
    """Test double of the Living Optics SDK interface, used to test the extraction logic without the proprietary SDK."""
    wl = np.linspace(440, 900, 96)

    class Meta:
        wavelengths = wl
        sampling_coordinates = np.random.default_rng(0).integers(0, 500, (100, 2))

    class F:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __len__(self):
            return 1

        def seek(self, i):
            pass

        def read(self):
            return b"raw"

    class Dec:
        @classmethod
        def from_calibration(cls, *a):
            d = cls()
            d.args = a
            return d

        def __call__(self, raw, conv):
            return Meta(), np.zeros((8, 8, 3)), np.full((100, 96), 2e-4)

    mods = {"lo": types.ModuleType("lo"), "lo.sdk": types.ModuleType("lo.sdk"), "lo.sdk.api": types.ModuleType("lo.sdk.api"),
            "lo.sdk.api.acquisition": types.ModuleType("a"), "lo.sdk.api.acquisition.io": types.ModuleType("io"),
            "lo.sdk.api.acquisition.io.open": types.ModuleType("open"), "lo.sdk.api.acquisition.data": types.ModuleType("data"),
            "lo.sdk.api.acquisition.data.decode": types.ModuleType("decode"),
            "lo.sdk.api.acquisition.data.formats": types.ModuleType("formats")}
    mods["lo.sdk.api.acquisition.io.open"].open = lambda path: F()
    mods["lo.sdk.api.acquisition.data.decode"].SpectralDecoder = Dec
    mods["lo.sdk.api.acquisition.data.formats"].LORAWtoRGB8 = lambda x: x
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)


def test_extraction_radiance_and_reflectance(fake_sdk, tmp_path):
    p = extract_to_npz("x.lo", tmp_path / "a.npz", "calib")
    z = np.load(p)
    assert z["spectra"].shape == (100, 96) and str(z["quantity"]) == "radiance"
    np.save(tmp_path / "w.npy", np.full(96, 4e-4, dtype="float32"))
    z2 = np.load(extract_to_npz("x.lo", tmp_path / "b.npz", "calib", white_reference=str(tmp_path / "w.npy")))
    assert str(z2["quantity"]) == "reflectance" and np.allclose(z2["spectra"], 0.5)
    with pytest.raises(ValueError, match="out of range"):
        extract_to_npz("x.lo", tmp_path / "c.npz", "calib", frame=3)
    with pytest.raises(ValueError, match="dark_reference"):
        extract_to_npz("x.lo", tmp_path / "d.npz", "calib", dark_reference=str(tmp_path / "w.npy"))
    rep = check_file("x.lo", "calib")
    assert all(s == "ok" for _, s in rep["steps"]) and rep["n_bands"] == 96


def test_missing_sdk_gives_actionable_message():
    rep = check_file("x.lo", "calib")
    assert rep["steps"][0][1].startswith("FAILED") and "SpectralDecoder" in rep["hint"]


def test_cli_analyze_writes_outputs_and_reports_errors(tmp_path, capsys):
    t = make_synthetic_table(n_plots=8, n_control=2, n_spectra=40, n_bands=20, effect=1.0, plot_sd=0.05, seed=2)
    csv = tmp_path / "s.csv"
    t.df.drop(columns="y").to_csv(csv, index=False)
    out = tmp_path / "res"
    rc = cli.main(["analyze", str(csv), "--out", str(out), "--trees", "20", "--n-perm", "4", "--max-per-plot", "30"])
    assert rc == 0
    for f in ("s_summary.json", "s_plot_scores.csv", "s_qc.csv", "s_band_noise.csv", "s_figure.png", "s_band_importance.csv"):
        assert (out / f).exists(), f
    assert "plot_auc" in capsys.readouterr().out
    assert cli.main(["analyze", str(csv), "--group-col", "nope", "--out", str(out)]) == 1
    rc = cli.main(["trait", str(csv), "--target", "biomass", "--out", str(out), "--n-perm", "0", "--trees", "20"])
    assert rc == 0 and (out / "s_biomass_summary.json").exists()
    pd.DataFrame().to_csv(tmp_path / "empty.csv")
    assert cli.main(["analyze", str(tmp_path / "empty.csv")]) == 1


def test_extraction_requires_calibration_folder(fake_sdk, tmp_path):
    with pytest.raises(ValueError, match="calibration_folder"):
        extract_to_npz("x.lo", tmp_path / "a.npz", "")


def test_extract_cli(fake_sdk, tmp_path):
    from hsiplot.cli import main
    out = tmp_path / "p.npz"
    assert main(["extract", "x.lo", "--calibration-folder", "calib", "--out", str(out)]) in (0, None)
    assert np.load(out)["spectra"].shape == (100, 96)
