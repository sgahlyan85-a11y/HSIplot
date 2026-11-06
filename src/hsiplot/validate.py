"""Plot-level validation of a spectra classifier.

Why this module exists
----------------------
Thousands of spectra come from each plot and are near-copies of each other. If spectra are split at random into
training and test sets, the same plot is on both sides and the classifier can score highly by recognising the plot
instead of the treatment (pseudoreplication; Hurlbert 1984). Cross-validation for grouped data must hold out whole
groups (Roberts et al. 2017). Everything here holds out whole plots.

Schemes
-------
``paired`` (default)
    Leave-pair-out: each fold holds out one control plot and one treated plot and trains on all the others. Class
    proportions in the training set are therefore identical in every fold, and the statistic is the fraction of
    (control, treated) pairs in which the treated plot receives the higher P(treated) (an unbiased-by-design AUC
    estimator for small samples; Airola et al. 2011). With no signal it is 0.5.
``lopo``
    Leave-one-plot-out. Simple, but with very few controls the training prior shifts whenever a control plot is held
    out, which pushes null-data AUCs *below* 0.5. Use it for comparison, not as the primary statistic.

Significance
------------
The permutation test reassigns the treatment labels to plots in every possible way (or a random sample of them), repeats
the *whole* cross-validation under each assignment, and reports how often the statistic is at least as large as observed
(Ojala & Garriga 2010). The smallest attainable p is 1 / (number of assignments), e.g. 1/66 for 2 controls among 12 plots.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import SpectraTable

log = logging.getLogger("hsiplot")
EPS = 1e-12


def make_estimator(model: str = "rf", n_estimators: int = 100, seed: int = 0):
    """``rf``: Random Forest (scale-invariant, so no standardisation is needed; Breiman 2001).
    ``logreg``: L2 logistic regression with per-band standardisation fitted inside each training fold (a scale-sensitive baseline)."""
    if model == "rf":
        return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5, class_weight="balanced_subsample",
                                      n_jobs=1, random_state=seed)
    if model == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    raise ValueError(f"unknown model {model!r}; use 'rf' or 'logreg'")


def _folds(plot_y: np.ndarray, scheme: str) -> List[Tuple[int, ...]]:
    ctrl, trt = np.flatnonzero(plot_y == 0), np.flatnonzero(plot_y == 1)
    if len(ctrl) < 2 or len(trt) < 2:
        raise ValueError(f"need at least 2 control and 2 treated plots (have {len(ctrl)} and {len(trt)})")
    if scheme == "lopo":
        return [(int(p),) for p in range(len(plot_y))]
    if scheme == "paired":
        return [(int(c), int(t)) for c in ctrl for t in trt]
    raise ValueError(f"unknown scheme {scheme!r}; use 'paired' or 'lopo'")


def _fold_scores(X, y_row, plot_idx, test, model, n_estimators, seed) -> np.ndarray:
    """Mean P(treated) of each held-out plot, trained on all other plots."""
    te = np.isin(plot_idx, test)
    est = make_estimator(model, n_estimators, seed).fit(X[~te], y_row[~te])
    p = est.predict_proba(X[te])[:, list(est.classes_).index(1)]
    pi = plot_idx[te]
    return np.array([p[pi == t].mean() for t in test])


def _aggregate(folds, outs, plot_y: np.ndarray, scheme: str) -> Tuple[float, np.ndarray]:
    """Combine fold outputs into (statistic, per-plot mean score)."""
    sums, counts, hits = np.zeros(len(plot_y)), np.zeros(len(plot_y)), []
    for test, s in zip(folds, outs):
        for t, v in zip(test, s):
            sums[t] += v
            counts[t] += 1
        if scheme == "paired":
            c_score, t_score = (s[0], s[1]) if plot_y[test[0]] == 0 else (s[1], s[0])
            hits.append(1.0 if t_score > c_score + EPS else 0.5 if abs(t_score - c_score) <= EPS else 0.0)
    scores = sums / counts
    stat = float(np.mean(hits)) if scheme == "paired" else float(roc_auc_score(plot_y, scores))
    return stat, scores


def _cv_statistic(X, plot_idx, plot_y, scheme, model, n_estimators, seed) -> Tuple[float, np.ndarray]:
    """Serial cross-validation for one labelling of the plots."""
    y_row = plot_y[plot_idx]
    folds = _folds(plot_y, scheme)
    outs = [_fold_scores(X, y_row, plot_idx, t, model, n_estimators, seed) for t in folds]
    return _aggregate(folds, outs, plot_y, scheme)


@dataclass
class ClassificationResult:
    scheme: str
    model: str
    plot_scores: pd.DataFrame            # plot_id, group, y, mean_p_treated
    statistic: float                     # plot-level AUC (see module docstring)
    p_value: Optional[float]
    p_exact: bool
    n_assignments: int                   # total possible label assignments
    n_perm_used: int
    min_p: float
    null: Optional[np.ndarray]
    n_spectra: int
    n_fits: int

    def summary(self) -> Dict:
        return {"scheme": self.scheme, "model": self.model, "n_plots": int(len(self.plot_scores)),
                "n_control": int((self.plot_scores.y == 0).sum()), "n_treated": int((self.plot_scores.y == 1).sum()),
                "n_spectra": self.n_spectra, "plot_auc": round(self.statistic, 4),
                "permutation_p": None if self.p_value is None else round(self.p_value, 4),
                "p_is_exact": self.p_exact, "n_permutations": self.n_perm_used,
                "smallest_attainable_p": round(self.min_p, 4)}


def run_classification(
    table: SpectraTable,
    scheme: str = "paired",
    model: str = "rf",
    n_estimators: int = 100,
    n_perm: Optional[int] = None,
    max_perm: int = 500,
    seed: int = 0,
    n_jobs: int = 1,
) -> ClassificationResult:
    """Plot-level cross-validated classification with a refit permutation test.

    n_perm : None -> test every possible labelling if there are at most ``max_perm`` of them, else a random sample of
             ``max_perm``; 0 -> skip the test; k -> at most k labellings.
    """
    X = table.X
    plot_ids, plot_idx = np.unique(table.plots, return_inverse=True)
    plot_y = np.array([int(table.y[plot_idx == i][0]) for i in range(len(plot_ids))])
    groups = np.array([table.df["group"].to_numpy()[plot_idx == i][0] for i in range(len(plot_ids))])
    if not (np.isfinite(X).all()):
        raise ValueError("spectra contain NaN/inf; run preprocess_table first")
    folds = _folds(plot_y, scheme)

    # observed statistic: folds in parallel
    y_row = plot_y[plot_idx]
    outs = Parallel(n_jobs=n_jobs)(delayed(_fold_scores)(X, y_row, plot_idx, t, model, n_estimators, seed) for t in folds)
    stat, scores = _aggregate(folds, outs, plot_y, scheme)
    n_fits = len(folds)

    n_ctrl = int((plot_y == 0).sum())
    total = comb(len(plot_ids), n_ctrl)
    p, exact, null, used = None, False, None, 0
    if n_perm != 0:
        cap = max_perm if n_perm is None else n_perm
        rng = np.random.default_rng(seed)
        if total <= cap:
            assignments = list(itertools.combinations(range(len(plot_ids)), n_ctrl))
            exact = True
        else:
            assignments = [tuple(rng.choice(len(plot_ids), n_ctrl, replace=False)) for _ in range(cap)]

        def one(ctrl):
            py = np.ones(len(plot_ids), int)
            py[list(ctrl)] = 0
            return _cv_statistic(X, plot_idx, py, scheme, model, n_estimators, seed)[0]

        log.info("permutation test: %d labelling(s), each with a full cross-validation", len(assignments))
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(c) for c in assignments))
        used = len(assignments)
        n_fits += used * len(folds)
        ge = int((null >= stat - 1e-9).sum())
        p = ge / used if exact else (1 + ge) / (1 + used)
    min_p = 1.0 / total if (p is None or exact) else 1.0 / (1 + used)
    ps = pd.DataFrame({"plot_id": plot_ids, "group": groups, "y": plot_y, "mean_p_treated": scores.round(4)})
    return ClassificationResult(scheme, model, ps, stat, p, exact, total, used, min_p, null,
                                int(len(X)), n_fits)


def random_split_demo(table: SpectraTable, model: str = "rf", n_estimators: int = 100, seed: int = 0) -> float:
    """Balanced accuracy of a **random split by spectrum** (70/30, stratified).

    This number is *not* evidence: spectra of the same plot are on both sides of the split, so it measures how well the
    classifier recognises plots it has already seen. Compare it with the plot-level result to see the inflation."""
    X, y = table.X, table.y
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    est = make_estimator(model, n_estimators, seed).fit(X[tr], y[tr])
    return float(balanced_accuracy_score(y[te], est.predict(X[te])))


def band_importance(table: SpectraTable, n_estimators: int = 200, seed: int = 0) -> pd.DataFrame:
    """Impurity-based Random Forest importance per band, fitted on all spectra. Descriptive only: it can be biased towards
    correlated or high-variance features (Strobl et al. 2007), and the bands are strongly correlated."""
    est = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=seed).fit(table.X, table.y)
    return pd.DataFrame({"wavelength_nm": table.wavelengths.round(1), "importance": est.feature_importances_})
