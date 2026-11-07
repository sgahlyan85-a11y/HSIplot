# Methods

What each step does, why, and the literature behind it. Section numbers match `docs/WORKFLOW.md`.

## 1. The data

A Living Optics frame holds one RGB scene and about 4,384 sparse spectra of 96 bands (roughly 440-900 nm). Spectra are **radiance**, the light reaching the sensor. **Reflectance** is the fraction of incident light a surface returns; it needs a panel of known reflectance imaged in the same session. Radiance depends on illumination, exposure and geometry, so it is comparable within one session only.

## 2. Calibration to reflectance (`hsiplot reference`, `hsiplot extract --white-reference`)

    reflectance = (L_target - L_dark) / (L_panel - L_dark) x rho_panel

`L_dark` is optional. The reference spectrum is the mean of the spectra inside a region that contains only panel (`--roi`) or of the brightest spectra (`--auto-bright`). The tool reports the coefficient of variation across those spectra and warns above 5 %, which indicates non-panel pixels or uneven light. Unit tests check that the arithmetic recovers a known reflectance and that reflectance does not change when the light level changes.

## 3. Quality control and noise handling (`hsiplot analyze`, `PreprocessConfig`)

| Step | Purpose |
|---|---|
| remove NaN, zero, negative values | invalid measurements |
| brightness percentiles within each plot | deep shadow (low signal) and saturation or glint |
| roughness above median + k x MAD within each plot | band-to-band jitter well above that plot's typical level |
| reflectance ceiling (reflectance only) | glint, or a wrong white reference |
| NDVI mask (Rouse et al. 1974) | soil and shadow; approximate when computed from radiance |
| Savitzky-Golay smoothing, 9 bands, order 2 (Savitzky & Golay 1964) | removes band-to-band jitter and keeps broad features |
| SNV (Barnes et al. 1989) | removes overall brightness per spectrum; also removes brightness information |

Thresholds act **within each plot**, so a plot that is globally darker or noisier is not removed wholesale, and the run stops if a plot would lose every spectrum. None of these steps uses group labels, so they cannot leak the treatment into the validation. Per-band standardisation is done inside the cross-validation loop (fitted on training plots only) for models that need it. Random Forests are invariant to monotone rescaling of each band, so they do not need standardisation; the optional logistic-regression baseline does.

`plot_qc` flags plots whose brightness is far from the median plot or that have many invalid spectra: such a plot was measured under a different exposure. Decide explicitly whether to keep or exclude it, and report the decision. `band_noise_report` gives per-band signal-to-noise so that bands can be dropped on evidence.

Always compare presets `none`, `clean` and `clean_snv`. A conclusion that depends on the preprocessing is not robust.

## 4. Why validation must hold out whole plots

Thousands of spectra from one plot are near-copies. Treating them as independent observations is pseudoreplication (Hurlbert 1984). If spectra are split at random into training and test sets, spectra of the same plot appear on both sides, and a classifier can score highly by recognising the plot instead of the treatment. Cross-validation for grouped data must hold out whole groups (Roberts et al. 2017). The same applies to variables that are constant within a plot (plot id, biomass, cover): they identify the plot and must not be used to predict the plot's treatment.

`random_split_demo` (printed as `random_split_balanced_accuracy_INVALID`) shows the inflation: it is the score of a random split by spectrum, which is not evidence of anything.

## 5. Classification (`hsiplot analyze`, `run_classification`)

**Cross-validation schemes**
- `paired` (default): leave-pair-out. Each fold holds out one control plot and one treated plot and trains on all others, so the class proportions of the training set are the same in every fold. The statistic is the fraction of (control, treated) pairs in which the treated plot receives the higher mean P(treated). It is an AUC estimator designed for small samples (Airola et al. 2011) and equals 0.5 when there is no signal.
- `lopo`: leave-one-plot-out. With very few controls, holding out a control plot leaves fewer controls in training, which raises that plot's predicted P(treated); null data then give an AUC *below* 0.5. Use it as a comparison, not as the primary statistic.

**Models.** Random Forest (Breiman 2001; scikit-learn, Pedregosa et al. 2011) with balanced class weights and `min_samples_leaf = 5`; `logreg` (L2, standardised inside each fold) as a linear baseline.

**Permutation test.** The group labels are reassigned to plots in every possible way (or a random sample when there are more than `--max-perm`), the **entire cross-validation is repeated** under each assignment, and p is the proportion of assignments whose statistic is at least as large as the observed one (Ojala & Garriga 2010). The smallest attainable p is 1 / (number of assignments), for example 1/66 for 2 controls among 12 plots; sampled tests use (1 + k)/(1 + N).

**The test is valid but can be conservative.** When the control plots form their own cluster in spectral space, other labellings that are consistent with that cluster can also score high, so p can stay well above its minimum even for a clear effect, particularly with 8-12 plots. A non-significant result therefore means "not shown", not "absent". Use the detectability analysis (section 7) to say what the design could find.

**Band importance** (impurity-based) is descriptive. It can be biased towards correlated or high-variance variables (Strobl et al. 2007), and spectral bands are strongly correlated.

## 6. Plot-level group difference (PERMANOVA; Anderson 2001)

Each plot is reduced to its mean spectrum (bands standardised across plots). The pseudo-F compares the distance between group centroids with the spread of plots within a group; significance is by exact label permutation. It answers "do the groups differ in spectrum more than plots differ within a group?", needs no model fitting, and is more powerful than the classifier test when the groups differ in mean spectrum. The classifier answers the harder question "can a model recognise the group of an unseen plot?". Report both.

## 7. Detectability (`hsiplot power`)

A null result is informative only if the design could have found an effect. The tool multiplies chosen bands (default >= 750 nm) of the treated plots by (1 + effect), keeps the real spectra and the real between-plot variation, and reruns the test. The smallest effect with p <= 0.05 is the practical sensitivity. Report it next to any null result. It describes sensitivity to a multiplicative change in the chosen bands only; other kinds of effects (a shifted red edge, a narrow absorption feature) need their own check.

## 8. Continuous traits (`hsiplot trait`)

A trait measured once per plot has n = number of plots independent observations. Each plot is reduced to its mean spectrum and a Random Forest regressor is validated leave-one-plot-out. The statistic is R^2 of held-out predictions (negative = worse than predicting the training mean) and p comes from refitting under shuffled trait values. With about 12 plots the analysis is exploratory.

## 9. Limits

- Few plots per group give low power; the smallest attainable p is bounded by the number of label assignments.
- Radiance is not comparable between sessions; use reflectance for any cross-date comparison.
- A model validated within one experiment, date and camera does not demonstrate transfer to other conditions; that needs spatially or temporally independent validation (Ploton et al. 2020).
- Soil, shadow and non-target spectra stay in the data unless masked; check the effect of `--ndvi-min`.
- Sampling a subset of spectra per plot (`--max-per-plot`) is for speed; the plot, not the spectrum, remains the unit.

## 10. Verification in this repository

`pytest` runs: calibration against a known answer; preprocessing behaviour (per-plot limits, clear errors); a positive control (known effect is recognised in unseen plots and by PERMANOVA); negative controls (no effect is not declared significant beyond chance); a leakage demonstration (plots with unique signatures and no treatment effect give a high random-split score but a non-significant plot-level result); reproducibility with a fixed seed; the extraction logic (frame selection, reference calibration, error handling) through a test double of the SDK interface, since the SDK is proprietary; and the command-line interface.

## References

- Airola, A., Pahikkala, T., Waegeman, W., De Baets, B., & Salakoski, T. (2011). An experimental comparison of cross-validation techniques for estimating the area under the ROC curve. *Computational Statistics & Data Analysis*, 55(4), 1828-1844.
- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance. *Austral Ecology*, 26, 32-46.
- Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.
- Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187-211.
- Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research*, 11, 1833-1863.
- Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540.
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.
- Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *NASA SP-351*, 309-317.
- Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.
