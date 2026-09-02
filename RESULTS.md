# SIH26162 Results

Last updated: 2026-09-02
Run: Kaggle notebook 1, full seven-country feature build and six-country model evaluation

## Outcome

The source-level FIRMS pipeline ran successfully for all seven countries. On the complete six-country training population, LightGBM substantially outperformed the rule and Isolation Forest baselines, but its geographic generalization is weaker than the earlier Libya-and-Algeria-only experiment.

The current evidence does **not** justify building the Stage 06 GRU/TCN branch. Intensity features matter more than temporal features, removing seasonality slightly improves the model, and performance declines substantially when Nigeria is held out. The next high-value task is to repair cross-country evaluation and exposure-dependent features before running the untouched India holdout.

## Experimental contract

- Unit of analysis: a spatial source produced by 1 km metric grid clustering, not an individual FIRMS detection.
- Training/model-selection countries: Iraq, Algeria, Nigeria, Libya, Angola, and Indonesia.
- Untouched holdout: India. No Indian row may be used for fitting, feature selection, threshold selection, or hyperparameter selection.
- Supervision: active EOG/World Bank gas-flare sites.
- Unmatched sources are **unlabelled**, not verified negatives. EOG does not label kilns, cement plants, steelworks, or every gas flare.
- Validation grouping: 10 km `block_id`, so fragments of one physical flare cannot cross ordinary CV folds.
- Excluded model inputs: latitude, longitude, country, NASA `type`, `eog_dist_m`, and NOAA-20-derived features.

Because this is a positive-unlabelled problem, reported precision is a lower bound and conventional F1 is an operational proxy rather than a complete measure of industrial-source classification.

## Source construction

| Country | FIRMS detections | Sources | EOG sites | EOG sites recovered | Site recall | Labelled source rows |
|---|---:|---:|---:|---:|---:|---:|
| Iraq | 1,289,983 | 47,978 | 266 | 231 | 86.8% | 822 |
| Algeria | 744,252 | 19,122 | 438 | 397 | 90.6% | 1,256 |
| Nigeria | 3,914,564 | 554,299 | 436 | 333 | 76.4% | 980 |
| Libya | 427,517 | 4,824 | 202 | 175 | 86.6% | 622 |
| Angola | 4,869,716 | 970,253 | 73 | 11 | 15.1% | 35 |
| Indonesia | 501,771 | 183,558 | 370 | 188 | 50.8% | 380 |
| India | 7,594,269 | 1,258,787 | 193 | 133 | 68.9% | 343 |

`EOG sites` counts physical flare sites. `Labelled source rows` can be larger because one physical flare may intersect several 1 km source cells.

The feature builder retains only sources detected by the two uniformly available instruments, MODIS and VIIRS S-NPP. This produces 1,687,353 foreign training sources with 3,766 EOG-matched source rows and 992,536 Indian feature rows with 276 EOG-matched source rows.

The source-construction recall of 68.9% is the current upstream ceiling for evaluating recovery of the 193 known Indian EOG sites unless clustering or EOG matching is changed.

## Baselines

Results use the full six-country training population.

| Method | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Best simple rule: `n_days >= 10` and `month_entropy >= 0.7` | 0.561 | 0.434 | 0.489 | 0.245 | 0.717 |
| Isolation Forest, all features | 0.120 | 0.279 | 0.168 | 0.060 | 0.892 |
| LightGBM, 49 features | 0.845 | 0.379 | **0.523** | **0.575** | **0.974** |

The earlier Libya-and-Algeria-only LightGBM F1 of approximately 0.691 should not be presented as the final model result. Adding the full geographic training population reduces F1 to 0.523 and exposes substantial regional variation.

The global LightGBM operating threshold was 0.747, selected from spatially grouped out-of-fold predictions. At that threshold the model produced 1,688 positive predictions: 1,427 EOG-matched positives, 261 nominal false positives, and 2,339 missed EOG-matched source rows.

## Feature ablation

### Cumulative groups

| Feature groups | Features | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|
| Temporal only | 9 | 0.445 | 0.419 | 0.906 |
| Temporal + intensity | 34 | 0.513 | 0.535 | 0.965 |
| + seasonality | 36 | 0.515 | 0.545 | 0.965 |
| + local-time features | 39 | **0.524** | 0.565 | 0.973 |
| All seven feature groups | 49 | 0.518 | 0.572 | 0.974 |

Small differences between the `LGBM full` row and the cumulative all-groups row arise from the feature ordering used during stochastic LightGBM training.

### Leave-one-group-out

| Removed group | F1 | PR-AUC | Interpretation |
|---|---:|---:|---|
| Temporal | 0.514 | 0.568 | Only a small reduction; weak support for a sequence model |
| Intensity | 0.497 | 0.543 | Largest harmful removal; intensity is essential |
| Seasonality | **0.530** | 0.577 | Improves the result; current seasonality features are not helping |
| Timing | 0.524 | 0.565 | Similar F1 but lower ROC-AUC |
| Cross-instrument | 0.521 | 0.580 | Similar F1 and slightly better PR-AUC |
| Confidence | 0.522 | 0.580 | Similar F1 and slightly better PR-AUC |
| Spatial | 0.521 | 0.565 | Little incremental value |

`month_entropy` dominates gain importance at 301,754, more than four times `night_frac`, the second-ranked feature at 68,366. `n_months` is fourth. This concentration, combined with the improvement obtained by removing seasonality, warrants a robustness check before India is scored.

## Leave-one-country-out evaluation

| Held-out country | Positive rows | PR-AUC | ROC-AUC | Reported F1 |
|---|---:|---:|---:|---:|
| Algeria | 1,128 | 0.647 | 0.898 | 0.636 |
| Angola | 35 | 0.150 | 0.919 | 0.024 |
| Indonesia | 380 | 0.107 | 0.905 | 0.217 |
| Iraq | 781 | 0.653 | 0.910 | 0.679 |
| Libya | 577 | 0.748 | 0.906 | 0.728 |
| Nigeria | 865 | 0.281 | 0.876 | 0.351 |

### LOCO limitation

The current Stage 05 implementation calls `best_f1_threshold(yt, p)` after predicting each held-out country. It therefore uses the held-out labels to select that country's threshold. The reported LOCO F1, precision, and recall are optimistic and must not be treated as deployment estimates.

PR-AUC and ROC-AUC remain threshold-independent and are usable for diagnosis. They show strong ranking performance for Libya, Iraq, and Algeria but a large decline for Nigeria. Angola contains only 35 matched source rows and is particularly unsuitable for conventional binary precision/F1 interpretation because its unmatched population includes unlabelled flares and other industrial sources.

A valid LOCO procedure must select the threshold using only grouped out-of-fold predictions from the remaining countries, then apply that frozen threshold to the held-out country.

## Frozen global-threshold diagnostic

Applying the global grouped-CV threshold of 0.747 to the saved OOF predictions gives the following within-country diagnostic. This is not LOCO because the corresponding training folds still contain other spatial blocks from the same country.

| Country | Precision | Recall | F1 |
|---|---:|---:|---:|
| Algeria | 0.869 | 0.407 | 0.554 |
| Angola | 0.500 | 0.029 | 0.054 |
| Indonesia | 0.460 | 0.105 | 0.171 |
| Iraq | 0.844 | 0.588 | 0.693 |
| Libya | 0.890 | 0.503 | 0.642 |
| Nigeria | 0.886 | 0.206 | 0.334 |

This confirms that one global threshold produces high nominal precision but low recall, especially in Nigeria and the vegetation-fire background countries.

## Coverage-horizon risk

The observation windows differ by country:

- India: 2019-2024, six years.
- Iraq, Algeria, Nigeria, and Libya: 2021-2024, four years.
- Angola and Indonesia: 2022-2024, three years.

Several features depend directly on exposure length: `n_det`, `n_days`, `n_months`, `n_years`, `span_days`, and total FRP. `n_months` counts distinct year-month pairs rather than calendar months of the year. These features can therefore encode the country-specific observation window even though explicit country and coordinates were removed. India is also outside the training range for maximum exposure.

Before final India inference, the pipeline should either:

1. rebuild every country on a common window such as 2022-2024; or
2. replace raw exposure-dependent counts with annualized or opportunity-normalized quantities and validate that the values are comparable across windows.

The common-window experiment is the cleaner primary analysis. A full-history, exposure-normalized model can be retained as a sensitivity analysis.

## Calibration

The saved OOF predictions contain 1,687,353 rows with no missing values or duplicate `source_id` values. The ten equal-frequency calibration bins are nearly all zero because EOG-matched rows are extremely rare; the highest decile has mean predicted probability 0.0182 and observed positive prevalence 0.0206.

The small Brier score is dominated by the overwhelming unlabelled majority and should not be interpreted as proof of good positive-class calibration. Future calibration reporting should use top-tail or log-spaced probability bins and a precision-recall curve.

## Decisions

### Stage 06: temporal GRU/TCN

**Do not build yet.** Temporal-only LightGBM reaches F1 0.445, adding intensity raises it to 0.513, and removing the temporal group from the full model only lowers F1 from 0.523 to 0.514. Geographic generalization, label incompleteness, coverage normalization, and threshold validity are more important than model capacity.

### Stage 07: imagery branch

**Keep dropped.** Sentinel-2, Landsat, and WorldCover imagery are not included. External acquisition would add the most expensive branch before the tabular evaluation is trustworthy.

### Stage 08: stacking

**Keep deferred.** There is only one validated predictive branch. Stacking is not justified without an independently useful second branch.

### Stage 09: India holdout

**Data ready; evaluation not yet authorized.** `features_India.parquet` exists and contains 992,536 modeled sources. India must remain untouched until the model specification, feature window, and operating threshold are frozen using foreign countries only.

### Stage 10: error analysis

After India inference, evaluate:

- known EOG flare-site recall, including onshore/offshore and flare-size strata;
- recall against thermally plausible GEM and fuel-burning WRI facilities;
- OSM flare and kiln matches;
- kiln versus crop-burning confusions in kiln-heavy states;
- a spatially stratified manual sample of unmatched high-score predictions for precision estimation.

Facility absence must not be counted as proof of a false positive because the facility catalogues are incomplete.

## Required next run

Stage 05b now implements the first four items below in
`kaggle/kg_05b_robust_tabular.py`. The code has passed local common-window,
binary CV, PU bagging, and nested LOCO smoke tests on Libya and Algeria. Full
six-country results remain pending until the Kaggle run completes. Exact cells
and required artifacts are listed in `KAGGLE_05B.md`.

1. Correct LOCO so thresholds are selected without held-out-country labels.
2. Rebuild or retrain on a common 2022-2024 observation window.
3. Test a reduced LightGBM specification without seasonality and coverage-dependent raw counts.
4. Compare models using foreign-country PR-AUC, frozen-threshold recall/precision, and LOCO stability.
5. Freeze the feature set, hyperparameters, and threshold in a manifest.
6. Run India once and save source-level scores without retuning.
7. Perform EOG/facility/manual-review validation and Stage 10 error analysis.

## Evidence files

Raw result artifacts are stored in `../output nb1/`:

- `02_source_summary.csv`
- `04_baselines.csv`
- `05_lgbm_ablation.csv`
- `05_lgbm_leave_country_out.csv`
- `05_feature_importance.csv`
- `05_oof_predictions.parquet`
- `features_<country>.parquet`
- `sih-ag-p-log.log`

The Kaggle run completed without a model error. The only warnings in the log are debugger and notebook-conversion warnings unrelated to the experiment.
