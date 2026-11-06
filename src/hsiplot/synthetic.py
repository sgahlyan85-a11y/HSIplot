"""Synthetic plots with a known effect, for tests, demos and sanity checks. The numbers have no scientific meaning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import SpectraTable


def make_synthetic_table(n_plots: int = 12, n_control: int = 2, n_spectra: int = 200, n_bands: int = 96,
                         effect: float = 0.25, plot_sd: float = 0.15, seed: int = 0, quantity: str = "radiance") -> SpectraTable:
    """Plots with leaf-like spectra. ``effect`` is the relative increase of NIR (> 700 nm) in treated plots
    (0 = no effect: a negative control). ``plot_sd`` is the between-plot variation of overall brightness
    and NIR level, which a valid analysis must not confuse with treatment."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(440, 900, n_bands)
    base = 1e-4 * (1 + 0.5 * np.exp(-((wl - 550) / 40) ** 2) + 2 * (wl > 720))
    ctrl = set(rng.choice(n_plots, n_control, replace=False).tolist())
    parts = []
    for p in range(n_plots):
        gain = rng.lognormal(0, plot_sd)
        nir = 1 + rng.normal(0, plot_sd / 2) * (wl > 700)
        trt = 0 if p in ctrl else effect
        mean = base * gain * nir * (1 + trt * (wl > 700))
        S = mean * rng.lognormal(0, 0.1, (n_spectra, 1)) + rng.normal(0, 3e-6, (n_spectra, n_bands))
        d = pd.DataFrame(S, columns=[f"wl_{w:.2f}" for w in wl])
        d.insert(0, "group", "control" if p in ctrl else "treated")
        d.insert(0, "plot_id", p + 1)
        d["biomass"] = 10 + 40 * gain + rng.normal(0, 2)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["y"] = (df.group != "control").astype(int)
    bands = [c for c in df.columns if c.startswith("wl_")]
    cols = ["plot_id", "group", "y", "biomass"] + bands
    return SpectraTable(df[cols].copy(), bands, wl, quantity, "control")
