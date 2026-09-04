# SIH26162 results

Last updated: 2026-09-04

## Final decision

The frozen foreign-country branch is `guarded_cv_tabular`, a compact tabular LightGBM
ensemble augmented with a guarded visual residual from Sentinel-2 and WorldCover context.
The final confirmation used three fresh seeds on one fixed panel of 294 QA-approved
foreign sources, including 87 EOG-matched positives.

| Branch | Macro F1 | Macro PR-AUC | Macro ROC-AUC | Worst-country PR-AUC |
|---|---:|---:|---:|---:|
| Compact tabular baseline | 0.7669 | 0.9054 | 0.9199 | 0.8250 |
| Guarded CV plus tabular | **0.8166** | **0.9293** | **0.9481** | 0.8400 |
| All 77 tabular features, diagnostic only | 0.7347 | 0.8935 | 0.9225 | **0.8489** |

Observation: guarded fusion gained 0.0239 macro PR-AUC and 0.0497 macro F1 over the
protected compact baseline. PR-AUC improved in five of six countries. Indonesia declined
by 0.0051, within the preregistered tolerance of 0.02.

Observation: every fresh seed produced a positive macro PR-AUC gain: 0.0234, 0.0228,
and 0.0218. All 12 fixed acceptance conditions passed.

Observation: the positive-stratified, 10 km block bootstrap estimated a mean PR-AUC gain
of 0.0238 with a 95% interval from -0.0006 to 0.0515.

Interpretation: the visual residual is stable enough to retain under the specified
non-inferiority policy. The interval crosses zero, so the evidence does not establish a
conventionally significant positive gain. The result supports a guarded hybrid, not a
claim that imagery independently solves the task.

## Evaluation contract

- Prediction unit: one 1 km source cell, not an individual detection.
- Development countries: Algeria, Angola, Indonesia, Iraq, Libya, and Nigeria.
- Spatial grouping: 10 km `block_id` to reduce leakage between fragments of one site.
- Supervision: active EOG gas-flare sites.
- Unmatched sources: unlabelled, not verified negatives.
- Excluded inputs: latitude, longitude, country, NASA `type`, `eog_dist_m`, and
  NOAA-20-derived features.
- Thresholds and fusion weights: learned only from training-country folds.
- NB12b role: seed-stability confirmation on the same fixed foreign panel, not a new
  country test and not a population-precision study.

## Data and source construction

The audit covered 19,342,072 FIRMS detections across seven countries. There were no
null values, malformed acquisition dates, or exact duplicate rows. A total of 754 rows,
about 0.004%, had non-positive FRP.

| Country | Detections | Sources | EOG sites | Sites recovered | Source-construction recall |
|---|---:|---:|---:|---:|---:|
| Iraq | 1,289,983 | 47,978 | 266 | 231 | 86.8% |
| Algeria | 744,252 | 19,122 | 438 | 397 | 90.6% |
| Nigeria | 3,914,564 | 554,299 | 436 | 333 | 76.4% |
| Libya | 427,517 | 4,824 | 202 | 175 | 86.6% |
| Angola | 4,869,716 | 970,253 | 73 | 11 | 15.1% |
| Indonesia | 501,771 | 183,558 | 370 | 188 | 50.8% |
| India | 7,594,269 | 1,258,787 | 193 | 133 | 68.9% |

Libya illustrates why source aggregation matters: 427,517 detections become 4,824 source
cells, about 89 detections per source. A 1 km grid retained roughly 88% to 92% of known
EOG sites in the Libya and Algeria resolution sweep while limiting fragmentation.

The common-window feature population contains 1,634,155 foreign sources with 3,506
EOG-matched source rows. MODIS and VIIRS S-NPP are used uniformly. NOAA-20 is absent
for Angola and Indonesia, so using its presence or ratios would create a country shortcut.

## Experiment progression

Results below use different cohorts and protocols. They show the research progression and
must not be read as one directly comparable leaderboard.

### Initial tabular work

The early Libya plus Algeria experiment reported LightGBM F1 0.691, PR-AUC 0.740,
and ROC-AUC 0.930. It established that source-level aggregation and engineered thermal
features were useful, but it did not test geographic transfer.

On the larger six-country common-window population, the full 49-feature LightGBM reached
F1 0.523, PR-AUC 0.575, and ROC-AUC 0.974. Intensity was the most important ablation
family. Removing seasonality slightly improved F1, which exposed coverage and regional
shortcut risk.

### Corrected country transfer

Stage 05b selected each held-out-country threshold only from the remaining countries.

| Held-out country | F1 | PR-AUC | Recall |
|---|---:|---:|---:|
| Algeria | 0.596 | 0.660 | 0.475 |
| Angola | 0.165 | 0.231 | 0.281 |
| Indonesia | 0.144 | 0.098 | 0.117 |
| Iraq | 0.641 | 0.568 | 0.638 |
| Libya | 0.667 | 0.757 | 0.550 |
| Nigeria | 0.028 | 0.194 | 0.014 |

Macro F1 was 0.374 and macro PR-AUC was 0.418. Nigeria's collapse showed that the main
problem was geographic transfer rather than insufficient sequence-model capacity.

Country and EOG-site weighting in Stage 05c improved some operating points but remained
unstable. The unweighted development variant had macro-country F1 0.529 and macro-country
PR-AUC 0.499 on its grouped development predictions. Equal-country weighting performed
worse, so simple reweighting was not adopted as the solution.

Nested selection in Stage 05d raised country-held-out macro F1 to 0.435 but macro PR-AUC
remained 0.420. The NB8 domain revamp improved these to 0.441 and 0.460. These values
still reflected large country variation.

### Imagery and fusion

The acquisition pipeline requested a bounded foreign-country panel from Sentinel-2 L2A
and ESA WorldCover. The final NB6 v2 manifest contains 295 successful chips, four failed
chips, and 301 pending candidates. After QA and label alignment, 294 sources entered the
fixed evaluation panel.

NB9 compared four branches on that EOG-enriched panel:

| Branch | Macro F1 | Macro PR-AUC | Macro ROC-AUC |
|---|---:|---:|---:|
| Early fusion | **0.7916** | **0.9059** | **0.9202** |
| Late fusion | 0.7772 | 0.8710 | 0.9076 |
| Thermal only | 0.6620 | 0.8138 | 0.8491 |
| Image only | 0.6744 | 0.7423 | 0.8604 |

Image-only performance was insufficient. Early fusion was retained as the stronger design.

NB11 tested temporal descriptors and a TCN:

| Branch | Macro F1 | Macro PR-AUC | Macro ROC-AUC |
|---|---:|---:|---:|
| Nested champion | **0.8140** | **0.9089** | **0.9250** |
| NB9 baseline | 0.7916 | 0.9059 | 0.9202 |
| Temporal descriptors | 0.7933 | 0.9023 | 0.9189 |
| TCN only | 0.7083 | 0.8034 | 0.8570 |

The TCN did not justify takeover. Temporal evidence remains useful as engineered summary
features, but the neural sequence branch added complexity without stronger ranking.

NB12 introduced CNN embeddings, multispectral summaries, morphology, and protected
tabular baselines. Its discovery result favored guarded CV plus tabular fusion at PR-AUC
0.9283 versus 0.9082 for compact tabular. NB12b then repeated the fixed comparison with
fresh seeds and produced the final decision shown at the top of this document.

At a 20% review budget, the compact baseline found 55 of 87 known positives across the
six country panels, while guarded fusion found 54. The only loss was one known positive
in Libya. This met the preregistered operational tolerance but shows why the visual branch
must remain guarded.

## Historical India evaluation

India is no longer a pristine project-level holdout. A completed NB10 run evaluated the
superseded NB8 `regularized_raw` model before the later NB12 and NB12b work.

| Model and cohort | Sources | Positive rows | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical NB8 model on India | 706,686 | 197 | 0.093 | 0.391 | 0.151 | 0.142 | 0.915 |

Of 187 active EOG sites in that run, 97 were recoverable after candidate construction and
49 were detected. Recall was 50.5% of recoverable sites and 26.2% of all active sites.

The final NB12b model has not been evaluated on India. India remained absent from NB12
and NB12b training, feature selection, thresholding, fusion-weight selection, and seed
confirmation. Any future NB12b India result must be described as a final-model transfer
evaluation, not as a first untouched test.

## Limitations

1. EOG labels identify gas flares, not every type of industrial thermal source. F1 and
   nominal precision measure agreement with incomplete proxy labels.
2. The NB12b panel is EOG-enriched. Its precision and prevalence do not represent the
   full source population.
3. NB12b reuses the same 294-source cohort. Three seeds measure algorithmic variation,
   not new-country sampling variation.
4. The bootstrap interval crosses zero. The visual residual passed the predefined guard,
   but its positive benefit remains uncertain.
5. Candidate construction limits site recall, especially in Angola and Indonesia.
6. Facility catalogues are incomplete. Absence from a catalogue is not proof that a
   high-scoring source is a false positive.

## Frozen artifacts and next work

The exact NB12b ensemble is stored under `artifacts/nb12b/`: nine compact LightGBM models,
three visual PU models, one fusion calibration file, and the hashed embedding and morphology
caches. `results/nb12b_confirmatory/12b_manifest.json` records the protocol, versions,
input hashes, output hashes, and all 12 acceptance checks.

The highest-value next steps are:

1. Run the frozen NB12b model on India without retuning, while reporting that the strict
   project-level holdout was already spent by NB10.
2. Build a population-scale review queue and estimate precision with spatially stratified
   manual review.
3. Separate flare, kiln, cement, steel, crop-burning, and wildfire errors using independent
   validation sources.
4. Improve candidate coverage before claiming facility-level recall.

The evidence registry in `results/README.md` maps each stage to its retained files.
