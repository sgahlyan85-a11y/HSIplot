import numpy as np
import pytest

from hsiplot.permanova import group_difference_test
from hsiplot.synthetic import make_synthetic_table
from hsiplot.validate import _folds, random_split_demo, run_classification

FAST = dict(n_estimators=30, seed=0)


def test_paired_folds_hold_out_one_plot_of_each_class():
    y = np.array([0, 0, 1, 1, 1, 1])
    f = _folds(y, "paired")
    assert len(f) == 2 * 4 and all(y[a] == 0 and y[b] == 1 for a, b in f)
    assert _folds(y, "lopo") == [(i,) for i in range(6)]
    with pytest.raises(ValueError, match="at least 2"):
        _folds(np.array([0, 1, 1, 1]), "paired")


def test_known_effect_is_recognised_in_unseen_plots(small_effect):
    r = run_classification(small_effect, **FAST)
    assert r.statistic == 1.0 and r.p_exact and r.n_assignments == 28
    # the refit permutation test is conservative with 2 control plots (other labellings can also score 1.0): p is not tiny
    assert r.p_value < 0.5


def test_permanova_detects_known_effect_and_not_null(small_effect, small_null):
    a = group_difference_test(small_effect)
    assert a.p_exact and a.n_assignments == 28 and a.p_value == pytest.approx(1 / 28)
    assert group_difference_test(small_null).p_value > 0.05


def test_permanova_false_positive_rate_on_null():
    hits = 0
    for s in range(20):
        t = make_synthetic_table(n_plots=8, n_control=2, n_spectra=20, n_bands=20, effect=0.0, plot_sd=0.15, seed=300 + s)
        hits += group_difference_test(t).p_value <= 0.05
    assert hits <= 3                           # nominal 5 % of 20 = 1; allow sampling noise


def test_no_effect_is_not_detected_across_seeds():
    p_small, aucs = 0, []
    for s in range(6):
        t = make_synthetic_table(n_plots=8, n_control=2, n_spectra=40, n_bands=20, effect=0.0, plot_sd=0.15, seed=100 + s)
        r = run_classification(t, n_estimators=20, seed=s)
        aucs.append(r.statistic)
        p_small += r.p_value <= 0.05
    assert p_small <= 1                       # at most chance-level false positives
    assert 0.2 <= np.mean(aucs) <= 0.8        # paired AUC on null data is centred on 0.5, not far below it


def test_plot_memorisation_inflates_random_split_but_not_plot_validation():
    """Plots have their own signatures and there is NO treatment effect: a random split scores high, whole-plot validation does not."""
    t = make_synthetic_table(n_plots=8, n_control=2, n_spectra=60, n_bands=30, effect=0.0, plot_sd=0.3, seed=7)
    inflated = random_split_demo(t, n_estimators=30)
    r = run_classification(t, n_estimators=30)
    assert inflated > 0.8
    assert r.p_value > 0.05


def test_lopo_and_logreg_run(small_effect):
    r = run_classification(small_effect, scheme="lopo", model="logreg", n_perm=0)
    assert r.p_value is None and r.statistic >= 0.5 and len(r.plot_scores) == 8
    assert r.summary()["smallest_attainable_p"] == round(1 / 28, 4)


def test_sampled_permutation_uses_plus_one_correction(small_effect):
    r = run_classification(small_effect, n_estimators=20, n_perm=5, max_perm=5)
    assert not r.p_exact and r.n_perm_used == 5 and r.p_value >= 1 / 6 - 1e-9


def test_rejects_nan_and_unbalanced_designs(small_effect):
    X = small_effect.X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_classification(small_effect.with_X(X))
    one_ctrl = make_synthetic_table(n_plots=6, n_control=1, n_spectra=20, n_bands=20)
    with pytest.raises(ValueError, match="at least 2"):
        run_classification(one_ctrl)


def test_results_are_reproducible(small_effect):
    a = run_classification(small_effect, n_perm=0, **FAST).plot_scores
    b = run_classification(small_effect, n_perm=0, **FAST).plot_scores
    assert a.equals(b)
