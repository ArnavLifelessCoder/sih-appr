# NB13: frozen NB12b India transfer evaluation

NB13 is the completed India evaluation of the frozen `guarded_cv_tabular` ensemble.
It does not fit models, features, thresholds, or fusion weights on India. It verifies the committed NB12b
model hashes, preserves the saved feature order and calibration references, and
compares the final guarded score with the compact early-fusion baseline.

India was already evaluated by the superseded NB8 model. NB13 is therefore a
final-model transfer audit, not a first untouched project holdout.

Completed across three acquisition batches: 300 fixed sources, 299 successful images,
one terminal failure, and 72 scored EOG-positive sites. Guarded PR-AUC was 0.8022
versus compact 0.7788, with a paired gain interval of [-0.0136, 0.0611]. The
prewritten guard retained guarded fusion. This is an India-informed branch decision,
not an independent evaluation of the policy selected using that decision.
See [retained evidence](../../results/nb13_india_transfer/README.md).

## Why the evaluation uses a panel

The frozen compact model itself consumes Sentinel-2 and WorldCover summary
features, and the guarded residual also needs CNN embeddings and morphology.
Downloading imagery for more than 700,000 India sources is not practical on a
free Kaggle account. Filling every image feature with missing values would be an
untested distribution shift and would not evaluate the selected model faithfully.

NB13 instead freezes 300 India sources before image acquisition:

- up to 96 distinct EOG-positive sites;
- the remaining rows are unlabelled controls;
- every source occupies a distinct 10 km block;
- the order is deterministic and label-interleaved;
- thermal percentile ranks are computed from the full India common-window table,
  not from the enriched panel.

The panel is deliberately EOG-enriched. Its PR-AUC compares model rankings on
this panel but does not estimate population precision.

## Kaggle setup

Import `notebooks/kaggle/13_india_guarded_transfer.ipynb`.

Set:

- Accelerator: GPU T4. CPU also works, but final CNN embedding is slower.
- Internet: On. Public Sentinel-2, WorldCover, the pinned encoder checkpoint,
  and the public Git repository are downloaded during the run.

The setup cell checks the frozen NB12b runtime versions and installs the recorded
scikit-learn, LightGBM, and joblib versions only when Kaggle differs.

For the first version, attach exactly one saved NB10 India holdout output. Its
saved working cache should expose `cache/features_India_2022_2024.parquet`.
NB2 is foreign-only and must not be used as the India feature input.

The notebook caps acquisition at 100 new attempts or 55 minutes per version.
If the last cell reports pending chips:

1. Choose Save Version and Save and Run All.
2. Create the next version of NB13.
3. Keep NB10 attached.
4. Add the previous NB13 saved output as an input.
5. Run all cells again.

The second and third versions reuse verified chips before making new requests.
Attach the full saved Kaggle output, not only its compact results ZIP, which omits chips.
Do not attach two earlier NB13 versions at once. A fourth version may be needed
only if a time limit stops a batch before 100 attempts.

If failures remain after pending reaches zero, inspect `download_manifest.csv`.
The scorer accepts terminal failures only when at least 80 percent of the fixed
panel passes imagery QA and at least 20 positive and 40 unlabelled rows remain.
Set `retry_failed=True` for one explicit retry version if failures are material.

## Frozen evaluation outputs

The final version produces:

- `13_india_predictions.parquet`;
- `13_india_ranking_metrics.csv`;
- `13_india_review_budgets.csv`;
- `13_india_block_bootstrap.json`;
- `13_india_decision.json`;
- `13_india_review_top100.csv`;
- `13_manifest.json`;
- exact India embedding and morphology caches;
- `nb13_india_transfer_results.zip`.

Report the compact and guarded EOG-proxy PR-AUC and ROC-AUC, guarded minus
compact block-bootstrap interval, and positive-site recall at fixed 10, 20, and
30 percent review budgets. Do not report population precision from this panel.

NB13 intentionally leaves F1 and threshold blank. The committed NB12b schema
states that its pilot threshold is not deployment-calibrated, so applying or
retuning that threshold on India would create a misleading headline number.

## Decision rule after the run

Keep the guarded branch for India ranking only if its PR-AUC is no more than 0.01
below compact and it reduces positive-site recovery at the 20 percent review
budget by no more than one site. If it violates either guard, use the compact
score for India and retain guarded fusion only for the foreign-country evidence.

This rule was written before NB13 results were available and passed on the completed
panel. Do not change it after seeing India scores. Independent future evaluation is
needed to assess the selected policy without reusing this branch-selection evidence.
