import numpy as np
import pytest

from hsiplot.calibrate import to_reflectance, white_reference

rng = np.random.default_rng(1)
WL = np.linspace(440, 900, 96)
LIGHT = 2e-4 * (1 + 0.8 * np.exp(-(((WL - 600) / 150) ** 2)))
TRUE = 0.05 + 0.4 * (WL > 700) + 0.05 * np.exp(-(((WL - 550) / 30) ** 2))
DARK = np.full(96, 2e-6)
RHO = 0.99


def shoot(refl, n, light=LIGHT):
    return light * refl * rng.lognormal(0, 0.03, (n, 1)) + DARK + rng.normal(0, 2e-6, (n, 96))


PANEL = shoot(RHO, 500)
COORDS = rng.integers(0, 2000, (500, 2))
LEAF = shoot(TRUE, 2000)


def test_recovers_known_reflectance():
    w, rep = white_reference(PANEL - DARK, COORDS, roi=(0, 2000, 0, 2000), panel_reflectance=RHO)
    assert not rep["warnings"]
    assert np.abs(to_reflectance(LEAF - DARK, w).mean(0) - TRUE).max() < 0.01
    assert np.abs(to_reflectance(LEAF, w, DARK).mean(0) - TRUE).max() < 0.01


def test_reflectance_is_independent_of_light_level():
    dim = 0.5
    leaf_dim = (LEAF - DARK) * dim + DARK
    w_dim, _ = white_reference((PANEL - DARK) * dim, COORDS, roi=(0, 2000, 0, 2000), panel_reflectance=RHO)
    assert np.abs(to_reflectance(leaf_dim - DARK, w_dim).mean(0) - TRUE).max() < 0.01
    assert abs(LEAF.mean() / leaf_dim.mean() - 1) > 0.3        # radiance itself changed


def test_non_uniform_panel_is_flagged():
    mixed = np.vstack([PANEL - DARK, LEAF[:300] - DARK])
    _, rep = white_reference(mixed, auto_bright=0, panel_reflectance=RHO)
    assert rep["warnings"]


def test_input_validation():
    with pytest.raises(ValueError):
        white_reference(PANEL)
    with pytest.raises(ValueError):
        white_reference(PANEL, COORDS, roi=(0, 1, 0, 1))        # too few spectra
    with pytest.raises(ValueError):
        to_reflectance(LEAF, np.ones(5))
    with pytest.raises(ValueError):
        to_reflectance(LEAF, np.zeros(96))
