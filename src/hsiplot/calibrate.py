"""Radiance -> reflectance calibration with a white reference panel.

The sensor records radiance L = illumination x reflectance x (geometry and exposure terms). A panel of known
reflectance photographed under the same light and exposure gives L_panel = illumination x rho_panel x (same terms), so

    reflectance = (L_target - L_dark) / (L_panel - L_dark) x rho_panel

``L_dark`` is optional (leave it out if the decoder already returns dark-corrected radiance).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


def white_reference(
    spectra: Array,
    coords: Optional[Array] = None,
    roi: Optional[Sequence[float]] = None,
    auto_bright: Optional[float] = None,
    panel_reflectance: Union[float, Array] = 1.0,
    max_cv: float = 0.05,
) -> Tuple[Array, Dict]:
    """Return ``(reference, report)``.

    ``reference`` is the mean panel spectrum divided by the panel reflectance, so that
    ``to_reflectance(x, reference)`` yields reflectance directly.

    Parameters
    ----------
    spectra : (N, bands) radiance spectra of the panel image.
    coords : (N, 2) scene coordinates (row, col) of each spectrum, needed with ``roi``.
    roi : (row0, row1, col0, col1) region of the scene that contains only panel.
    auto_bright : percentile; use spectra whose mean brightness is above it (the panel is the brightest object).
    panel_reflectance : certified reflectance (scalar or per band).
    max_cv : warn if the mean coefficient of variation across selected spectra exceeds this.
    """
    S = np.asarray(spectra, float)
    if roi is not None:
        if coords is None:
            raise ValueError("roi requires sampling coordinates")
        r0, r1, c0, c1 = roi
        c = np.asarray(coords)
        sel = (c[:, 0] >= r0) & (c[:, 0] < r1) & (c[:, 1] >= c0) & (c[:, 1] < c1)
    elif auto_bright is not None:
        b = S.mean(1)
        sel = b >= np.percentile(b, auto_bright)
    else:
        raise ValueError("give roi or auto_bright")
    P = S[sel]
    if len(P) < 20:
        raise ValueError(f"only {len(P)} panel spectra selected; check the region")
    mean = P.mean(0)
    cv = float((P.std(0) / mean).mean()) if (mean > 0).all() else float("inf")
    report: Dict = {"n_panel_spectra": int(len(P)), "mean_cv": round(cv, 4), "warnings": []}
    if not np.isfinite(cv) or cv > max_cv:
        report["warnings"].append(
            f"panel spectra vary by {100 * cv:.1f} % (limit {100 * max_cv:.0f} %): region contains non-panel pixels or lighting is uneven")
    if (mean <= 0).any():
        report["warnings"].append("non-positive panel mean in some band")
    return mean / np.asarray(panel_reflectance, float), report


def to_reflectance(spectra: Array, white: Array, dark: Optional[Array] = None) -> Array:
    """``(spectra - dark) / white``; ``white`` comes from :func:`white_reference` (and used the same ``dark``)."""
    S = np.asarray(spectra, float)
    W = np.asarray(white, float)
    if W.shape != (S.shape[1],):
        raise ValueError(f"white reference shape {W.shape} != ({S.shape[1]},)")
    if (W <= 0).any():
        raise ValueError("white reference has non-positive bands")
    if dark is not None:
        S = S - np.asarray(dark, float)
    return S / W
