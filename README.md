# hsiplot

Plot-level validation of hyperspectral spectra: calibration, quality control, leakage-free cross-validation and permutation tests for plot experiments (treatment vs control).

Hyperspectral cameras return thousands of spectra per plot. Those spectra are repeated measurements of one experimental unit, not independent samples. Splitting them at random between training and test sets lets a classifier recognise the plot instead of the treatment, and reported accuracies come out far too high. `hsiplot` holds out whole plots, repeats the entire analysis under every possible treatment assignment to obtain an honest p-value, and reports what size of effect the design could have detected.

Written for the Living Optics snapshot camera (`.lo` files), but everything after extraction works on any table of spectra with plot and group labels.

## What it does

| Function | Command | Python |
|---|---|---|
| Decode `.lo` frames to spectra | `hsiplot extract`, `hsiplot check` | `hsiplot.extract` |
| Radiance to reflectance with a white panel, with quality checks | `hsiplot reference` | `white_reference`, `to_reflectance` |
| Combine images into tables | `hsiplot build` | `build_tables` |
| Per-plot quality control, noise handling, per-band noise report | inside `analyze` | `PreprocessConfig`, `preprocess_table`, `plot_qc`, `band_noise_report` |
| Whole-plot cross-validated classification with refit permutation test | `hsiplot analyze` | `run_classification` |
| Plot-level group difference (PERMANOVA) | `hsiplot analyze` | `group_difference_test` |
| Smallest detectable effect | `hsiplot power` | `detectability` |
| Plot-level model of biomass, cover and similar traits | `hsiplot trait` | `run_trait_model` |

Details and references: [docs/METHODS.md](docs/METHODS.md). Step-by-step procedure: [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Install

```bash
# in the downloaded repository folder
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
Python 3.9 or newer (tested on 3.9-3.12). The Living Optics SDK is only needed for extraction (see below).

To check an installation, open `examples/check_my_setup.ipynb`: it tests the analysis on made-up data, reports which Living Optics SDK components are installed, and optionally runs your own table or `.lo` file.

## Try it in two minutes (no SDK, no data)

```bash
hsiplot demo
```
Generates synthetic plots (8 plots, 3 of them control) with and without a treatment effect, and analyses both; it takes about two minutes. The no-effect run is not declared significant, the effect run is recognised on unseen plots (plot AUC 1.0) and by PERMANOVA, and the random-split score is printed next to the valid result to show the inflation. With 8 plots the permutation p-values cannot go below 1/56, which is why both tests are reported.

## Use it on your data

A table with one row per spectrum:

```
plot_id,group,biomass,wl_440.00,wl_444.84,...,wl_900.00
1,control,17.4,0.00022,0.00022,...
```
`plot_id` is the experimental unit, `group` the treatment (the control label is `control` unless `--control-label` says otherwise), further plot-level columns are optional, and band columns are named by wavelength in nm.

```bash
hsiplot analyze spectra.csv --name experiment1            # group separation
hsiplot analyze spectra.csv --preset none                 # no cleaning (sensitivity)
hsiplot analyze spectra.csv --preset clean_snv            # with SNV (sensitivity)
hsiplot power   spectra.csv                               # smallest detectable effect
hsiplot trait   spectra.csv --target biomass              # plot-level trait model
```
Results are written to `results/`: summary JSON, per-plot scores, quality-control tables, band importance and a figure. Tables with other column names (`Band_01 ...`, `Herbicide`) work with `--wavelength-range`, `--group-col`, `--control-label` and `--rows-per-plot`; see `hsiplot analyze --help`.

From Python:

```python
from hsiplot import (load_table, PreprocessConfig, preprocess_table, plot_qc,
                     run_classification, group_difference_test, detectability)

table = load_table("spectra.csv", group_col="group", control_label="control")
print(plot_qc(table))                                    # exposure problems per plot
table = preprocess_table(table, PreprocessConfig.preset("clean")).subsample(300)
result = run_classification(table, scheme="paired", model="rf", n_jobs=-1)
print(result.summary())
print(group_difference_test(table).summary())
print(detectability(table, effects=[0, 0.1, 0.2, 0.4]))
```

## From `.lo` files to results

```bash
hsiplot check   file.lo --calibration-folder CALIB                                 # SDK and file check
hsiplot extract file.lo --calibration-folder CALIB --out data/spectra/plot01.npz     # radiance
hsiplot reference panel.npz --roi R0 R1 C0 C1 --out data/white_ref.npy
hsiplot extract file.lo --calibration-folder CALIB --white-reference data/white_ref.npy --out data/spectra/plot01.npz   # reflectance
hsiplot build   --metadata metadata.csv --spectra-dir data/spectra --out-dir data/tables
```
See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the field protocol, the checks at each stage and a reporting checklist.

### SDK requirements

Extraction needs the proprietary Living Optics SDK, in an edition that includes the spectral decoder (account at https://cloud.livingoptics.com), and the factory calibration folder supplied for your camera. The SDK is not on PyPI and is not distributed here. Install `hsiplot` into the same Python environment as the SDK. The code follows the official Living Optics examples (github.com/livingoptics/python-examples): `open`, `SpectralDecoder.from_calibration(factory_calibration_folder, field_calibration_file)` and `decoder(frame, LORAWtoRGB8)`. `hsiplot check` reports the result of each step. Everything after extraction runs without the SDK.

## Reading the results

- **Plot AUC and permutation p** (classification): can a model recognise the group of a plot it never saw? AUC near 0.5 means no detectable signal.
- **PERMANOVA p**: do groups differ in mean spectrum more than plots differ within a group?
- **Smallest detectable effect** (`power`): report it next to any null result.
- `random_split_balanced_accuracy_INVALID` is printed only to show leakage; it is not evidence.
- With few control plots the smallest attainable p is 1 / (number of label assignments), for example 1/66 for 2 controls among 12 plots, and the classifier test is conservative. A non-significant result means "not shown", not "absent".

## Limits

Few plots give low power. Radiance is not comparable between sessions (calibrate to reflectance). Validation inside one experiment, date and camera does not show transfer to other conditions. See METHODS section 9.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```
The tests include positive and negative controls, calibration against a known answer, a leakage demonstration and the command-line interface. Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). History: [CHANGELOG.md](CHANGELOG.md).

## Author, citation, licence

Sumit Kumar, University of Lincoln. MIT licence; the Living Optics SDK is proprietary and not included.

To cite, use the "Cite this repository" button on GitHub (from `CITATION.cff`), or:

```bibtex
@software{kumar_hsiplot_2026,
  author  = {Kumar, Sumit},
  title   = {hsiplot: plot-level validation of hyperspectral spectra},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/sgahlyan85-a11y/HSIplot}
}
```
