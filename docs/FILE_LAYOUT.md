# File ownership and retention

The Git repository is `sih26162/` inside the local SiH workspace. The neighboring
folders are original run downloads, not additional repositories. Preserve those
downloads as provenance and recovery copies. Compact authoritative evidence is
organized within this repository, using native run filenames.

## Repository ownership

| Location | Owner and policy |
|---|---|
| `README.md`, `RESULTS.md`, `ARCHITECTURE.md` | current overview, evidence interpretation and technical design |
| `kaggle/` | reusable experiment and inference implementations |
| `notebooks/kaggle/` | importable notebook entry points; historical numbers stay stable |
| `docs/runs/` | exact input, runtime and resumption instructions |
| `docs/data/` | acquisition requirements and external validation needs |
| `results/` | compact native experiment evidence, predictions and manifests |
| `artifacts/nb12b/` | frozen models, calibration, foreign embeddings and morphology |
| `scripts/verify_nb13_evidence.py` | repeatable read-only evidence verification |
| `tests/` | regression tests for numerical and inference behavior |
| `docs/archive/`, `output/pdf/` | explicitly dated historical reports |
| `data/`, `cache/`, `work/` | ignored local data and derived working files |

NB13's 19 native audit files occupy approximately 2.62 MB. Its embeddings and
morphology stay under `results/nb13_india_transfer/` because the native manifest
references them beside the prediction outputs. No duplicate India model files are
needed: the India run loaded the existing NB12b model artifacts.

## External download mapping

Paths below are relative to the parent SiH workspace, outside the Git repository.

| Download folder | Evidence retained in the repository |
|---|---|
| `output nb1/` | initial source, baseline and Stage 05 result folders |
| `stage05b_results/` | `results/stage05b_robust_tabular/` |
| `stage05c_results/` | `results/stage05c_balanced_tabular/` |
| `nb5_nested_results/` | `results/stage05d_nested_tabular/` |
| `nb4_sih_cv_pilot/` | `results/nb04_cv_pilot/` |
| `nb6_context_results/`, `nb6_context_results-v2/` | final evidence in `results/nb06_image_context/` |
| `nb8_domain_revamp/` | `results/nb08_domain_revamp/` |
| `nb9_final_fusion_results/` | `results/nb09_final_fusion/` |
| `nb10_india_holdout_results/` | `results/nb10_india_historical/` |
| `nb11_multimodal_results/` | `results/nb11_multimodal/` |
| `nb12_cv_tabular_results/` | `results/nb12_cv_tabular/` |
| `nb12b_confirmatory_results/` | `results/nb12b_confirmatory/`, `artifacts/nb12b/` |
| `nb13_india_transfer_results-200/` | intermediate acquisition evidence; final evidence supersedes it |
| `nb13 300/` | authoritative final import in `results/nb13_india_transfer/` |
| `nb13_india_transfer_results/` | download folder reused across exports; inspect its manifest rather than infer batch from its name |

The original downloads were not deleted or renamed during this organization. Local
paths can appear in immutable run metadata and do not imply that those paths exist
in a fresh clone. The architecture and runbooks define portable input discovery.

## What Git excludes

Raw FIRMS archives, full-country features, raw image chips, population-scale prediction
tables, logs, ZIPs, temporary scripts, caches and editor files remain excluded. Compact
parquets are allowed only under `results/` and `artifacts/`. Frozen native evidence
must retain its bytes; `.gitattributes` protects hash-sensitive files from newline
conversion. Original result JSON and CSV files should never be reformatted.

The compact India ZIP excludes chips. The full saved Kaggle output must be retained
for resuming acquisition, visually reviewing imagery or reproducing encoder features.
An extracted ZIP plus its metadata cannot substitute for missing `.npz` images.

For new experiments, use a new numbered run and result folder. Preserve old metrics
and clearly record changed sampling, splits, labels, features and calibration. Avoid
renaming source modules referenced by saved joblib classes or embedded notebooks.
