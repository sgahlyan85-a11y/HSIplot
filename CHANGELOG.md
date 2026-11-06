# Changelog

## 1.0.0
- Installable package `hsiplot` with a command-line interface (`hsiplot analyze | trait | power | extract | reference | build | check | demo | convert`).
- Whole-plot validation: leave-pair-out (default) and leave-one-plot-out cross-validation; permutation test that refits the model under every label assignment.
- Plot-level PERMANOVA on plot-mean spectra as a complementary, fast test.
- Plot-level model for continuous traits (biomass, cover) with permutation test.
- Detectability check: smallest injected spectral change the design detects.
- Radiance-to-reflectance calibration with white-panel quality checks.
- Per-plot quality control and noise handling; per-band noise report; band selection by wavelength.
- Tables carry wavelengths in the band column names (`wl_<nm>`); legacy names are accepted with an explicit wavelength range.
- Test suite with positive and negative controls, calibration against a known answer, and a leakage demonstration; CI on Python 3.9-3.12.
