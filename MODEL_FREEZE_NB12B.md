# Model freeze: SIH-NB12B-2026-09-04

## Status

The foreign-country ranking model is frozen at source commit `288de63`.
The promoted branch is `guarded_cv_tabular` under protocol
`12b-guarded-confirmatory-ensemble-v1`.

This freeze covers model structure, selected features, seeds, learned artifacts,
score construction, and missing-image fallback. It does not freeze a deployment
classification threshold. The pilot threshold was optimized inside the enriched
foreign evaluation panel and is not calibrated for the India source population.

## Frozen deployment contract

- Training and validation countries: Algeria, Angola, Indonesia, Iraq, Libya,
  and Nigeria.
- Untouched final holdout: India. India was not loaded during NB12b.
- Confirmatory seeds: 272, 273, and 274. Discovery seed 271 is excluded.
- Compact models: nine LightGBM models, three per seed.
- Visual models: three PU visual probes, one per seed.
- Visual residual weight: 0.5 for every seed.
- Final score: average the three seed-specific guarded fusion scores.
- Missing imagery: average the three seed-specific calibrated compact scores.
- Spatial validation unit: 10 km block, with outer leave-one-country-out folds.
- Label framing: EOG matches are positives; unmatched sources remain unlabeled.

Within each seed, the three compact probabilities are averaged, compact and
visual scores are calibrated from foreign out-of-fold references, and the
guarded residual is applied as:

`tabular + alpha * 4 * tabular * (1 - tabular) * (visual - tabular)`

No additional tuning on the six-country panel is allowed under this freeze.
Any material change creates a new protocol and freeze identifier.

## Confirmatory result

| Metric | Compact | Guarded CV + tabular | Change |
|---|---:|---:|---:|
| Macro country PR-AUC | 0.905404 | 0.929336 | +0.023932 |
| Macro country F1 | 0.766876 | 0.816620 | +0.049744 |
| Macro country ROC-AUC | 0.919853 | 0.948094 | +0.028241 |
| Worst-country PR-AUC | 0.825015 | 0.840037 | +0.015022 |

The guarded branch improved PR-AUC in five of six countries. Indonesia changed
by -0.005054 PR-AUC, inside the preregistered -0.020 country guard. All three
fresh seeds produced positive macro PR-AUC gains. The paired 10 km block
bootstrap mean gain was +0.023764 with a 95% interval from -0.000632 to
+0.051502. The interval crosses zero, so the run supports the frozen
non-inferiority guard but does not establish conventional positive statistical
significance.

All 12 preregistered acceptance conditions passed.

## Integrity anchors

- Stable panel: 294 sources, 87 EOG positives, 6 countries.
- Stable panel cohort SHA256:
  `7c44d70f3505d65d92bba4abbd2c2e9bebf3eef6a1eb9a432880612ffef3a2f0`
- Embedding cache SHA256:
  `44b78d9137b0cebc3f07d6aba2322a0139d402f04c1329fbac84ecc43a74f00b`
- Morphology cache SHA256:
  `b873bf53eaf9161f9d4f775bc84af9670a7d7d52124988a921347894e0127719`
- Frozen ResNet-18 checkpoint SHA256:
  `e3a335e38d1d189ad3b0eba4be4004a9c52c5e846317b6737ac9f0fac57e1ac8`
- Acceptance record SHA256:
  `972f3ac8afeb8d1d056e07ca7db39d28432ce108192dff5e644184f9f3693ba1`
- Ensemble predictions SHA256:
  `6386efa6739e0710182e311cbb36314891afe89eeaad79723741a35132e982b8`
- Frozen model report:
  `output/pdf/SIH_Frozen_Model_Report_NB12B.pdf`
- Frozen model report SHA256:
  `6cc027e3d4c4dce9807e9298280dd07ca77b80212b377a4354ac613d28c074ec`

The complete artifact inventory and all model hashes are recorded in
`outputs/12b_manifest.json` inside the audited NB12b result package.

## Known limits and next evaluation

- The 294-source panel is enriched and is not a population precision estimate.
- Three seeds measure algorithmic variance, not new-country sampling variance.
- EOG labels cover gas flares. Kilns, cement plants, and steel facilities can be
  industrial positives inside the unlabeled pool.
- The spatial bootstrap interval crosses zero.
- At a 20 percent review budget, guarded recovered 54 positives versus 55 for
  compact. The single lost positive was in Libya.
- India imagery, India labels, and an India score are still absent.

The next step is deployment-equivalent threshold or review-budget calibration
using foreign out-of-fold predictions only, followed by one sealed India run.
The recommended India path is a compact FIRMS prefilter plus bounded imagery
reranking so the full source population remains feasible on free Kaggle compute.
