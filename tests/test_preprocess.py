import numpy as np
import pytest

from hsiplot.preprocess import PreprocessConfig, band_noise_report, ndvi, plot_qc, preprocess_arrays, preprocess_table
from hsiplot.synthetic import make_synthetic_table


def test_invalid_values_removed_and_logged():
    t = make_synthetic_table(n_plots=4, n_spectra=100, n_bands=20)
    X = t.X.copy()
    X[3, 5] = -1.0
    X[10, 0] = np.nan
    _, keep, steps = preprocess_arrays(X, t.plots, t.wavelengths, PreprocessConfig(brightness_percentiles=None, roughness_mad=None))
    assert not keep[3] and not keep[10] and steps[0][1] == 2


def test_limits_are_per_plot_so_a_dark_plot_survives():
    t = make_synthetic_table(n_plots=4, n_spectra=200, n_bands=20)
    X = t.X.copy()
    X[t.plots == 2] *= 0.05                                   # plot 2 is 20x darker (different exposure)
    out = preprocess_table(t.with_X(X))
    assert set(out.plots) == {1, 2, 3, 4}
    assert (out.plots == 2).sum() > 150


def test_plot_that_loses_every_spectrum_raises():
    t = make_synthetic_table(n_plots=4, n_spectra=100, n_bands=20)
    X = t.X.copy()
    X[t.plots == 1] = -1.0
    with pytest.raises(ValueError, match="plot"):
        preprocess_table(t.with_X(X))


def test_snv_and_smoothing():
    t = make_synthetic_table(n_plots=4, n_spectra=100, n_bands=20)
    s = preprocess_table(t, PreprocessConfig.preset("clean_snv")).X
    assert np.allclose(s.mean(1), 0, atol=1e-9) and np.allclose(s.std(1), 1, atol=1e-6)
    raw, sm = t.X[:200], preprocess_table(t, PreprocessConfig.preset("clean")).X[:200]
    assert np.abs(np.diff(sm, 2, axis=1)).mean() < np.abs(np.diff(raw, 2, axis=1)).mean()


def test_none_preset_changes_nothing():
    t = make_synthetic_table(n_plots=4, n_spectra=50, n_bands=20)
    assert np.allclose(preprocess_table(t, PreprocessConfig.preset("none")).X, t.X)


def test_reflectance_ceiling_only_for_reflectance():
    t = make_synthetic_table(n_plots=4, n_spectra=50, n_bands=20)
    with pytest.raises(ValueError, match="reflectance"):
        preprocess_table(t, PreprocessConfig(reflectance_max=1.5))
    r = make_synthetic_table(n_plots=4, n_spectra=50, n_bands=20, quantity="reflectance")
    X = r.X * 1e3                                              # reflectance-like values below 1
    X[0, 3] = 3.0
    out = preprocess_table(r.with_X(X), PreprocessConfig(reflectance_max=1.5))
    assert len(out.df) < len(r.df)


def test_ndvi_and_qc_and_noise_report():
    t = make_synthetic_table(n_plots=4, n_spectra=50, n_bands=40)
    n = ndvi(t.X, t.wavelengths)
    assert n.shape == (len(t.df),) and (n > 0).all()
    X = t.X.copy()
    X[t.plots == 3] *= 0.01
    q = plot_qc(t.with_X(X))
    assert q.set_index("plot_id").loc[3, "flag"] == "CHECK" and (q.flag == "CHECK").sum() == 1
    r = band_noise_report(t)
    assert len(r) == 40 and (r.snr > 0).all()


def test_all_spectra_removed_gives_clear_error():
    t = make_synthetic_table(n_plots=4, n_spectra=30, n_bands=20)
    with pytest.raises(ValueError, match="removed every spectrum"):
        preprocess_table(t.with_X(-np.abs(t.X)))
