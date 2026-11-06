import pytest

from hsiplot.synthetic import make_synthetic_table


@pytest.fixture(scope="session")
def small_effect():
    """8 plots (2 control), strong treatment effect, small plot-to-plot variation."""
    return make_synthetic_table(n_plots=8, n_control=2, n_spectra=60, n_bands=30, effect=1.0, plot_sd=0.05, seed=3)


@pytest.fixture(scope="session")
def small_null():
    return make_synthetic_table(n_plots=8, n_control=2, n_spectra=60, n_bands=30, effect=0.0, plot_sd=0.15, seed=4)
