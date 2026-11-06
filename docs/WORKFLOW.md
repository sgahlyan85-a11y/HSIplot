# Workflow

Eight stages from the field to a reportable result. Every command has `--help`.

| # | Stage | Command | Output | Check before moving on |
|---|---|---|---|---|
| 0 | Design and acquisition | field work | `.lo` files, panel image, notes | see below |
| 1 | SDK and file check | `hsiplot check file.lo` | step-by-step report | every step ok, shapes as expected |
| 2 | Extraction | `hsiplot extract file.lo --out x.npz` | radiance `.npz` | N x 96, wavelengths ~440-900 nm |
| 3 | Calibration | `hsiplot reference`, `extract --white-reference` | reflectance `.npz` | panel CV < 5 %, no saturation |
| 4 | Tables | `hsiplot build` | one table per subset | plots, groups and units correct |
| 5 | QC and cleaning | inside `hsiplot analyze` | `*_qc.csv`, `*_band_noise.csv` | no plot flagged CHECK without a decision |
| 6 | Group separation | `hsiplot analyze` | classifier and PERMANOVA results | presets `none`, `clean`, `clean_snv` agree |
| 7 | Sensitivity | `hsiplot power` | smallest detectable effect | report next to any null result |
| 8 | Traits | `hsiplot trait` | R^2 and permutation p | n = number of plots |

## Stage 0: design and acquisition

The analysis cannot be stronger than the design.
- Replicate controls: with 2 control plots among 12 the smallest attainable p is 1/66.
- Randomise treatment allocation to plots and record it.
- Image a white panel in every session, under the same light and exposure as the plants, filling the sampled area, not saturated. Re-image when the light changes.
- Fix exposure within a session, or record it.
- Record date, time, weather, camera serial, calibration folder and field calibration for each file.

## Stage 1-2: check and extract

```bash
hsiplot check path/to/file.lo --calibration-folder path/to/factory_calibration                     # reports which step fails
hsiplot extract path/to/file.lo --calibration-folder path/to/factory_calibration --out data/spectra/plot01.npz
```
`--calibration-folder` is the factory calibration folder Living Optics supplies for your camera (add `--field-calibration` if you recorded a field calibration). Values are radiance unless a white reference is given.
Extraction needs a Living Optics SDK with the spectral decoder (see README). Everything after this stage runs without the SDK.

## Stage 3: reflectance

```bash
hsiplot extract panel.lo --out data/spectra/panel.npz
hsiplot reference data/spectra/panel.npz --roi ROW0 ROW1 COL0 COL1 --panel-reflectance 0.99 --out data/white_ref.npy
hsiplot extract plot01.lo --white-reference data/white_ref.npy --out data/spectra/plot01.npz
```
Use the panel's certified reflectance. A warning from `reference` (non-uniform panel) means: do not use that reference. Sanity-check the result: low in red (~670 nm), steep rise across the red edge (~700-750 nm), NIR plateau; values above about 1.5 suggest glint or a wrong reference (`--refl-max`).

## Stage 4: tables

Describe images in a metadata file (see `examples/plots_metadata_template.csv`): `plot_id, npz, group`, optional `subset` (species, experiment) and plot-level columns such as `biomass`.
```bash
hsiplot build --metadata metadata.csv --spectra-dir data/spectra --out-dir data/tables
```
Tables name bands by wavelength (`wl_<nm>`). A table from elsewhere with columns `Band_01 ...` works with `--wavelength-range MIN MAX`; if it has no plot column, `--rows-per-plot N` assigns plots in blocks of N rows. Use `--group-col` and `--control-label` for other label names. `hsiplot convert big.xlsx big.csv` streams large Excel files with low memory.

## Stage 5-6: quality control and group separation

```bash
hsiplot analyze data/tables/speciesA_reflectance.csv --name speciesA
hsiplot analyze ... --preset none          # sensitivity: no cleaning
hsiplot analyze ... --preset clean_snv     # sensitivity: with SNV
hsiplot analyze ... --exclude-plots 9      # after deciding about a flagged plot; report it
```
Outputs in `results/`: `*_summary.json` (classifier and PERMANOVA), `*_plot_scores.csv`, `*_qc.csv`, `*_band_noise.csv`, `*_band_importance.csv`, `*_figure.png`.

Interpretation:
- A high plot AUC with a small permutation p means a model recognised the group of plots it had not seen.
- A small PERMANOVA p means the groups differ in mean spectrum more than plots differ within a group.
- AUC near 0.5 means no detectable signal; `lopo` on null data gives values below 0.5 (see METHODS section 5).
- `random_split_balanced_accuracy_INVALID` is printed to show leakage. It is not evidence.
- The classifier permutation test is conservative with few plots. Not significant means "not shown".

## Stage 7: sensitivity

```bash
hsiplot power data/tables/speciesA_reflectance.csv --effects 0 0.05 0.1 0.2 0.4
```
Reports the smallest relative change in the chosen bands (default >= 750 nm) that the design detects. Write results as "no separation was found; the design detected a X % change but not a Y % change".

## Stage 8: traits

```bash
hsiplot trait data/tables/speciesA_reflectance.csv --target biomass
```
The trait column must be constant within a plot. With about 12 plots, treat the result as exploratory.

## Reporting checklist

- [ ] Plots per group and how treatments were allocated
- [ ] Radiance or reflectance; how the reference was made; panel reflectance used
- [ ] Camera serial, SDK version, calibration files
- [ ] Cleaning steps and thresholds, spectra removed per step, plots excluded and why
- [ ] Validation scheme, plot-level statistic, permutation p, PERMANOVA p
- [ ] Results under `none`, `clean`, `clean_snv`
- [ ] Smallest detectable effect
- [ ] Limits: few controls, one date and camera, no external validation
