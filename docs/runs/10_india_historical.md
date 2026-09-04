# Historical India evaluation with the superseded NB8 model

Status: completed. Protocol: `09-india-frozen-holdout-v1`.

This run predates NB12 and NB12b. It evaluated the frozen NB8 `regularized_raw` model
on India and must not be described as an evaluation of the final guarded model.

## Inputs used

1. The saved source and detection outputs for India.
2. The NB8 foreign-country output containing `05e_manifest.json` and the three
   `05e_final_model_*.txt` files.
3. The original EOG workbook for India-side evaluation only.

The implementation is `kaggle/kg_09_final_india.py`. It rebuilds India on the same
2022-2024 window, applies the three frozen NB8 models and the threshold selected from
foreign-country OOF predictions, and performs no fitting or tuning on India.

## Completed result

| Sources | Positive rows | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 706,686 | 197 | 0.3314 | 0.0934 | 0.3909 | 0.1508 | 0.1418 | 0.9150 |

There were 187 active EOG sites in the run. Candidate construction made 97 recoverable;
49 were detected. Recall was 50.5% of recoverable sites and 26.2% of all active sites.

The exact evidence is retained in `results/nb10_india_historical/`:

- `09_india_metrics.csv`
- `09_india_site_metrics.csv`
- `09_manifest.json`

The 22 MB row-level prediction table and notebook log remain outside Git. The final
NB12b guarded model has not been evaluated on India, but this completed historical run
means India is no longer a pristine project-level holdout.
