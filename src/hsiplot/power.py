"""Detectability: what size of spectral effect could this design find?

A known multiplicative change is applied to a wavelength range of the treated plots (real spectra and the real
between-plot variation are kept), and the full plot-level analysis, including the refit permutation test, is rerun.
The smallest effect with p <= alpha is the practical sensitivity of the design. Report it next to any null result.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .io import SpectraTable
from .permanova import group_difference_test
from .validate import run_classification


def detectability(table: SpectraTable, effects: Sequence[float] = (0, 0.05, 0.1, 0.2, 0.4), nm_from: float = 750.0,
                  nm_to: Optional[float] = None, alpha: float = 0.05, test: str = "permanova", **kwargs) -> pd.DataFrame:
    """Rerun a plot-level test with spectra of treated plots scaled by ``1 + effect`` in ``[nm_from, nm_to]``.

    test : ``permanova`` (default; fast, plot-mean spectra) or ``classifier`` (full cross-validated classification with refit
           permutation test; slow). Extra keyword arguments go to ``run_classification`` when ``test='classifier'``.
    """
    if test not in ("permanova", "classifier"):
        raise ValueError("test must be 'permanova' or 'classifier'")
    mask = (table.wavelengths >= nm_from) & (table.wavelengths <= (table.wavelengths.max() if nm_to is None else nm_to))
    if not mask.any():
        raise ValueError("no bands in the requested wavelength range")
    treated = table.y == 1
    rows = []
    for e in effects:
        X = table.X.copy()
        X[treated[:, None] & mask[None, :]] *= (1 + e)
        tt = table.with_X(X)
        if test == "permanova":
            r = group_difference_test(tt, seed=kwargs.get("seed", 0))
            rows.append({"effect": e, "statistic": round(r.pseudo_f, 3), "p": r.p_value, "detected": bool(r.p_value <= alpha)})
        else:
            res = run_classification(tt, **kwargs)
            rows.append({"effect": e, "statistic": round(res.statistic, 3), "p": res.p_value,
                         "detected": bool(res.p_value is not None and res.p_value <= alpha)})
    return pd.DataFrame(rows)
