"""Plot-level model of a continuous trait (biomass, cover, yield ...) from spectra.

A trait measured once per plot has n = number of plots independent observations, not thousands. Each plot is therefore
reduced to its mean spectrum and a Random Forest regressor is validated leave-one-plot-out. The statistic is R^2 of the
held-out predictions (negative = worse than predicting the training mean); significance comes from refitting under
shuffled trait values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from .io import SpectraTable


def _loo(M: np.ndarray, t: np.ndarray, n_estimators: int, seed: int) -> np.ndarray:
    pred = np.zeros(len(t))
    for i in range(len(t)):
        tr = np.arange(len(t)) != i
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1).fit(M[tr], t[tr])
        pred[i] = rf.predict(M[i:i + 1])[0]
    return pred


def _r2(t: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def _null_r2(M, t_perm, n_estimators, seed) -> float:
    return _r2(t_perm, _loo(M, t_perm, n_estimators, seed))


@dataclass
class TraitResult:
    target: str
    predictions: pd.DataFrame
    r2: float
    rmse: float
    p_value: float
    n_perm: int

    def summary(self) -> Dict:
        return {"target": self.target, "n_plots": int(len(self.predictions)), "r2": round(self.r2, 3),
                "rmse": round(self.rmse, 3), "permutation_p": round(self.p_value, 4), "n_permutations": self.n_perm}


def run_trait_model(table: SpectraTable, target: str, n_estimators: int = 300, n_perm: int = 199, seed: int = 0,
                    n_jobs: int = 1) -> TraitResult:
    if target not in table.df.columns:
        raise ValueError(f"column {target!r} not found; plot-level columns: {[c for c in table.df.columns if c not in table.bands]}")
    g = table.df.groupby("plot_id")
    if (g[target].nunique(dropna=False) > 1).any():
        raise ValueError(f"{target!r} must be constant within a plot (one value per plot)")
    M = g[table.bands].mean().to_numpy()
    t = g[target].first().to_numpy(float)
    ids = g[target].first().index.to_numpy()
    if len(t) < 6:
        raise ValueError("fewer than 6 plots: a plot-level model is not meaningful")
    pred = _loo(M, t, n_estimators, seed)
    r2 = _r2(t, pred)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(t) for _ in range(n_perm)]
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(_null_r2)(M, tp, n_estimators, seed) for tp in perms)) \
        if n_perm else np.array([])
    p = (1 + int((null >= r2 - 1e-9).sum())) / (1 + len(null)) if n_perm else float("nan")
    rmse = float(np.sqrt(((t - pred) ** 2).mean()))
    return TraitResult(target, pd.DataFrame({"plot_id": ids, target: t, "predicted": pred.round(4)}), r2, rmse, p, int(n_perm))
