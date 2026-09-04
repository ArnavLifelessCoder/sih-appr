# Kaggle Run: Stage 05c Balanced Tabular Evaluation

Create Notebook 3 and attach the saved output of Notebook 2. Also attach the
original SIH dataset so the EOG workbooks remain available. Use a T4 GPU if it is
free, but this LightGBM run is CPU-bound and also works with Accelerator None.

India is not loaded or scored in this notebook.

## Cell 1: clone the current repository

```python
!rm -rf /kaggle/working/sih
!git clone https://github.com/ArnavLifelessCoder/sih-appr.git /kaggle/working/sih
%cd /kaggle/working/sih/kaggle
!git log -1 --oneline
```

## Cell 2: locate and copy Notebook 2 common-window features

```python
from pathlib import Path
import shutil

from kg_common import CACHE, TRAIN_COUNTRIES, HOLDOUT

CACHE.mkdir(parents=True, exist_ok=True)
for country in TRAIN_COUNTRIES:
    name = f"features_{country}_2022_2024.parquet"
    matches = [
        path for path in Path("/kaggle/input").rglob(name)
        if "india" not in str(path).lower()
    ]
    if not matches:
        raise FileNotFoundError(f"Attach Notebook 2 output: missing {name}")
    source = matches[0]
    target = CACHE / name
    shutil.copy2(source, target)
    print(country, source, "->", target)

assert HOLDOUT == "India"
assert HOLDOUT not in TRAIN_COUNTRIES
assert not list(CACHE.glob("features_India_2022_2024.parquet"))
print("All six foreign common-window feature files are ready.")
```

## Cell 3: run Stage 05c

```python
from kg_05c_balanced_tabular import run

variant_results, corrected_loco = run(
    n_splits=5,
    inner_splits=3,
    rounds=500,
)

display(variant_results)
display(corrected_loco)
```

Expected runtime is mainly determined by CPU allocation. Keep the notebook running
until all four weighting variants, six corrected LOCO fits, and the final foreign
model finish.

## Cell 4: inspect the country and site results

```python
import json
import pandas as pd
from kg_common import OUT, CACHE

display(pd.read_csv(OUT / "05c_weighting_variants.csv"))
display(pd.read_csv(OUT / "05c_country_oof.csv"))
display(pd.read_csv(OUT / "05c_site_oof.csv"))
display(pd.read_csv(OUT / "05c_corrected_loco.csv"))

with open(OUT / "05c_model_manifest.json", encoding="utf-8") as stream:
    manifest = json.load(stream)

print(json.dumps(manifest, indent=2))
assert manifest["holdout_country"] == "India"
assert manifest["holdout_loaded"] is False
assert (CACHE / "05c_final_foreign_model.txt").exists()
```

## Cell 5: zip the necessary files

```python
from pathlib import Path
import zipfile

bundle = Path("/kaggle/working/stage05c_results.zip")
required = [
    "outputs/05c_weighting_variants.csv",
    "outputs/05c_country_oof.csv",
    "outputs/05c_site_oof.csv",
    "outputs/05c_corrected_loco.csv",
    "outputs/05c_loco_country_metrics.csv",
    "outputs/05c_loco_site_metrics.csv",
    "outputs/05c_feature_importance.csv",
    "outputs/05c_model_manifest.json",
    "cache/05c_oof_predictions.parquet",
    "cache/05c_corrected_loco_predictions.parquet",
    "cache/05c_final_foreign_model.txt",
]

root = Path("/kaggle/working")
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in required:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        archive.write(path, arcname=relative)
        print("added", relative, path.stat().st_size)

print("Created", bundle, bundle.stat().st_size)
```

Download `stage05c_results.zip` and retain the full notebook log. Do not add an
India inference cell to this notebook version.
