# Kaggle Run: Stage 05b Robust Tabular Evaluation

This run uses foreign countries only. It rebuilds source features on the shared
2022-2024 observation window, compares reduced binary LightGBM with bagged PU
LightGBM, and runs corrected nested leave-one-country-out evaluation.

India must not be added to `build_common_features()` or `run()`.

## Prerequisites

- The repository is available at `/kaggle/working/sih`.
- The original SIH dataset is attached as a Kaggle input.
- Stage 02 detection caches are available in `/kaggle/working/cache`.
- Internet access is not required for the experiment itself.

If the Stage 02 caches are absent, Cell 2 recreates them for the six foreign
countries. It does not process India.

## Cell 1: environment check

```python
%cd /kaggle/working/sih/kaggle

import sys
import lightgbm
import pandas
import sklearn

print("Python:", sys.version)
print("LightGBM:", lightgbm.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)

from kg_common import DATA, CACHE, OUT, TRAIN_COUNTRIES, HOLDOUT
print("DATA:", DATA)
print("CACHE:", CACHE)
print("OUT:", OUT)
print("Training countries:", TRAIN_COUNTRIES)
print("Holdout:", HOLDOUT)
assert HOLDOUT == "India"
assert HOLDOUT not in TRAIN_COUNTRIES
```

## Cell 2: ensure foreign Stage 02 caches exist

```python
from kg_common import CACHE, TRAIN_COUNTRIES

required = [
    CACHE / f"detections_{country}.parquet"
    for country in TRAIN_COUNTRIES
]
missing = [path for path in required if not path.exists()]

if missing:
    print("Missing Stage 02 caches:")
    for path in missing:
        print(" ", path.name)

    from kg_02_source_clustering import main as build_sources
    build_sources(TRAIN_COUNTRIES)
else:
    print("All foreign Stage 02 caches are present.")
```

## Cell 3: build common-window features

```python
from kg_05b_robust_tabular import build_common_features

common_window_summary = build_common_features()
display(common_window_summary)
```

Expected output:

- `cache/features_<country>_2022_2024.parquet` for all six foreign countries
- `outputs/05b_common_window_summary.csv`

## Cell 4: run robust models and corrected LOCO

```python
from kg_05b_robust_tabular import run

model_results, corrected_loco = run(
    n_splits=5,
    inner_splits=3,
    pu_bags=3,
    unlabeled_per_positive=10,
    rounds=500,
)

display(model_results)
display(corrected_loco)
```

Do not change model settings after viewing India. India is not loaded by this
cell, and the generated manifest records that fact.

## Cell 5: inspect and package outputs

```python
import json
import pandas as pd
from kg_common import OUT

display(pd.read_csv(OUT / "05b_robust_models.csv"))
display(pd.read_csv(OUT / "05b_corrected_loco.csv"))
display(pd.read_csv(OUT / "05b_feature_importance.csv").head(20))

with open(OUT / "05b_feature_manifest.json", encoding="utf-8") as stream:
    manifest = json.load(stream)

print(json.dumps(manifest, indent=2))
assert manifest["holdout_country"] == "India"
assert manifest["holdout_loaded"] is False
```

Kaggle automatically preserves files written under `/kaggle/working` when a
notebook version is saved with output.

## Required files to download

Download these files after the run:

```text
outputs/05b_common_window_summary.csv
outputs/05b_robust_models.csv
outputs/05b_corrected_loco.csv
outputs/05b_feature_importance.csv
outputs/05b_feature_manifest.json
cache/05b_oof_predictions.parquet
cache/05b_corrected_loco_predictions.parquet
```

Also retain the complete Kaggle notebook log. Do not run India inference in this
notebook version.
