"""Plot-level test of group differences in the whole spectrum (PERMANOVA; Anderson 2001).

Each plot is reduced to its mean spectrum (n = number of plots). The pseudo-F statistic compares the distance between
group centroids with the spread of plots inside groups (Euclidean distance, bands standardised across plots by default).
Significance comes from reassigning the group labels to plots in every possible way (or a random sample), so it needs no
distributional assumptions and its resolution is limited only by the number of plots: with 2 control plots among 12 the
smallest attainable p is 1/66.

This test asks "do the groups differ in spectrum more than plots differ within a group?". The classification analysis
(``run_classification``) asks the harder question "can a model recognise the group of a plot it has never seen?". Report both.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import Dict

import numpy as np

from .io import SpectraTable


def _pseudo_f(M: np.ndarray, is_ctrl: np.ndarray) -> float:
    n = len(M)
    grand = M.mean(0)
    ss_between = 0.0
    ss_within = 0.0
    for mask in (is_ctrl, ~is_ctrl):
        g = M[mask]
        c = g.mean(0)
        ss_between += len(g) * float(((c - grand) ** 2).sum())
        ss_within += float(((g - c) ** 2).sum())
    return (ss_between / 1) / (ss_within / (n - 2) + 1e-300)


@dataclass
class PermanovaResult:
    pseudo_f: float
    p_value: float
    p_exact: bool
    n_assignments: int
    n_perm_used: int
    n_plots: int

    def summary(self) -> Dict:
        return {"pseudo_F": round(self.pseudo_f, 3), "permutation_p": round(self.p_value, 4), "p_is_exact": self.p_exact,
                "n_permutations": self.n_perm_used, "smallest_attainable_p": round(1 / self.n_assignments, 4), "n_plots": self.n_plots}


def group_difference_test(table: SpectraTable, standardize: bool = True, max_perm: int = 9999, seed: int = 0) -> PermanovaResult:
    """Pseudo-F test on plot-mean spectra with exact (or sampled) label permutation."""
    g = table.df.groupby("plot_id")
    M = g[table.bands].mean().to_numpy()
    y = g["y"].first().to_numpy()
    if (y == 0).sum() < 2 or (y == 1).sum() < 2:
        raise ValueError("need at least 2 control and 2 treated plots")
    if standardize:
        M = (M - M.mean(0)) / (M.std(0) + 1e-12)
    n, k = len(M), int((y == 0).sum())
    obs = _pseudo_f(M, y == 0)
    total = comb(n, k)
    rng = np.random.default_rng(seed)
    if total <= max_perm:
        assignments = itertools.combinations(range(n), k)
        exact = True
    else:
        assignments = (tuple(rng.choice(n, k, replace=False)) for _ in range(max_perm))
        exact = False
    stats = []
    for c in assignments:
        m = np.zeros(n, bool)
        m[list(c)] = True
        stats.append(_pseudo_f(M, m))
    stats = np.array(stats)
    ge = int((stats >= obs * (1 - 1e-9)).sum())
    p = ge / len(stats) if exact else (1 + ge) / (1 + len(stats))
    return PermanovaResult(obs, p, exact, total, len(stats), n)
