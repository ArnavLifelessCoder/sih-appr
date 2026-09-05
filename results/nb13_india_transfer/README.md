# NB13 India evidence

Status: complete. Protocol: `13-india-guarded-transfer-v1`.
Imported from the local `nb13 300/nb13_india_transfer/` saved-run export. Native
filenames and bytes are preserved. The inference code logged commit `62ca483`.

## Result

| Metric | Compact | Guarded CV plus tabular |
|---|---:|---:|
| PR-AUC | 0.7788305 | 0.8022484 |
| ROC-AUC | 0.8650269 | 0.8813632 |
| Known sites found at 20% review | 45 / 72 | 47 / 72 |

The fixed panel has 300 sources in distinct 10 km blocks. Acquisition succeeded for
299, including all 72 positive sites; one unlabelled source failed coverage checks.
The final India rule retains guarded ranking. The 95% interval for the PR-AUC gain
is -0.01360 to 0.06106, so a significant positive improvement is not established.
See [../../RESULTS.md](../../RESULTS.md) for interpretation and all review budgets.

## File groups

| Files | Role |
|---|---|
| `13_manifest.json` | frozen model identity, environment, output hashes, limitations |
| `13_india_predictions.parquet` | 299 rows of paired scores and review metadata |
| `13_india_ranking_metrics.csv` | PR-AUC and ROC-AUC; intentionally empty F1/threshold |
| `13_india_review_budgets.csv` | top 10%, 20%, 30% known-site recovery |
| `13_india_review_top100.csv` | guarded review list; includes unlabelled candidates |
| `13_india_block_bootstrap.json` | paired block interval, repeats and seed |
| `13_india_decision.json` | recorded two-condition India guard and branch choice |
| `13_cv_embeddings.*`, `13_morphology.*` | encoder/morphology caches and provenance |
| `india_panel.csv`, `india_panel.parquet`, `run_config.json` | frozen panel identity and thermal inputs |
| `download_manifest.csv`, `run_state.json` | terminal acquisition status and failure explanation |
| `image_features.parquet`, `image_quality.csv`, `coverage_by_label.csv` | pixel summaries, QA and label coverage |

The ten outputs listed in `13_manifest.json` are independently verifiable with
`python scripts/verify_nb13_evidence.py` from the repository root. The verifier also
recomputes ranking/review metrics and checks panel and model provenance.

These are EOG-enriched panel metrics, not population precision. No India F1 or
deployment-calibrated threshold exists. The India branch decision uses this panel's
metrics; future independent validation remains necessary.

## External inputs

The full NB10 feature cache and original image `.npz` chips are not committed.
The downloaded compact ZIP did not contain the image chips. Save the full final
Kaggle output for any later image inspection or repeat extraction. The checkpoint
URL and hash are in the embedding manifest. Saved features allow score auditing
without downloading imagery again.
