# Historical India evaluation with the superseded NB8 model

Status: completed. Protocol: `09-india-frozen-holdout-v1`.

This run predates NB12 and NB12b. It evaluated the frozen NB8 `regularized_raw` model
on India and must not be described as an evaluation of the final guarded model.

## Inputs used

1. The saved source and detection outputs for India.
2. The NB8 foreign-country output containing `05e_manifest.json` and the three
   `05e_final_model_*.txt` files.
3. The original EOG workbook for India-side evaluation only.

The recorded run rebuilt India on the 2022-2024 window and applied three frozen
NB8 models with the foreign OOF threshold. The retained manifest is the authority
for this historical run. `kaggle/kg_09_final_india.py` is an older NB5-based
implementation, not an exact reproduction of this NB8 three-model execution.

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

The 22 MB row-level prediction table and notebook log remain outside Git. This run
means India is no longer a pristine project-level holdout. The final NB12b model
was subsequently evaluated on a different, enriched image panel in
[NB13](13_india_guarded_transfer.md). Its metrics are not directly comparable here.
