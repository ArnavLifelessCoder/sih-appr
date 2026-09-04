# NB5: nested country-held-out tabular evaluation

Name: `sih nb5 nested evaluation`. Accelerator: None (CPU). Internet: On.
Attach NB2's saved outputs and the original SIH dataset containing the EOG workbook.
No NOAA-20 downloads, imagery training, or India inference happen in this run.

The debugging workflow identified fold-external weights, unmatched seeds and
global model selection before LOCO. This notebook fixes those issues, then merges
spatial blocks connected by known physical flare sites. Unknown site overlap and
the limitations of EOG proxy labels remain. This is one matched-seed experiment,
not a repeated-seed uncertainty estimate. Historical foreign data informed the
research design, so these are development results, not a newly untouched test.

The implementation is tracked in `kaggle/kg_05d_nested_tabular.py`. This runbook records
the historical Kaggle cells and should not be treated as the final model specification.

## Cell 1: dependencies

```python
%pip install -q lightgbm scikit-learn pandas numpy scipy pyarrow pyproj openpyxl pytest
```

## Cell 2: clone or update code without deleting files

```python
from pathlib import Path
import subprocess
import sys

repo = Path("/kaggle/working/sih")
if not repo.exists():
    subprocess.run([
        "git", "clone", "https://github.com/ArnavLifelessCoder/sih-appr.git", str(repo)
    ], check=True)
else:
    subprocess.run(["git", "pull", "--ff-only", "origin", "main"], cwd=repo, check=True)

assert (repo / "kaggle/kg_05d_nested_tabular.py").exists(), "Push the NB5 changes first."
subprocess.run(["git", "log", "-1", "--oneline"], cwd=repo, check=True)
sys.path.insert(0, str(repo / "kaggle"))
```

## Cell 3: locate inputs and copy only six foreign feature files

```python
import os
import shutil

input_root = Path("/kaggle/input")
workbook_name = "Flare-Volume-Estimates-by-individual-Flare-Location-2012-2025.xlsx"
workbooks = list(input_root.rglob(workbook_name))
assert len(workbooks) == 1, f"Attach exactly one original SIH dataset. EOG matches: {workbooks}"
assert workbooks[0].parent.name == "flare_inventory"
assert workbooks[0].parents[1].name == "eog"
os.environ["SIH_DATA"] = str(workbooks[0].parents[2])

from kg_common import CACHE, OUT, TRAIN_COUNTRIES, HOLDOUT

for country in TRAIN_COUNTRIES:
    name = f"features_{country}_2022_2024.parquet"
    matches = list(input_root.rglob(name))
    assert len(matches) == 1, f"Need one NB2 feature file for {country}; found {matches}"
    shutil.copy2(matches[0], CACHE / name)
    print(country, "<-", matches[0])

assert HOLDOUT == "India" and HOLDOUT not in TRAIN_COUNTRIES
assert not list(CACHE.glob("features_India*.parquet")), "Use a clean NB5 session."
from kg_05d_nested_tabular import PROTOCOL_VERSION
assert PROTOCOL_VERSION == "05d-nested-v1"
print("Inputs ready. India excluded. Protocol:", PROTOCOL_VERSION)
```

## Cell 4: fast regression tests

```python
subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_robust_evaluation.py", "tests/test_nested_evaluation.py",
    "-q", "-p", "no:cacheprovider",
], cwd=repo, check=True)
```

## Cell 5: full run

```python
from kg_05d_nested_tabular import run

development_results, country_holdout_results = run(
    n_splits=5,
    inner_splits=3,
    rounds=500,
    seed=31,
)

display(development_results)
display(country_holdout_results)
```

This uses 99 LightGBM fits: six outer countries each with four variants times
three inner folds plus one outer fit, followed by 20 all-country development fits
and one final foreign model. Budget more time than NB3; runtime depends on Kaggle
CPU allocation. Partial country results are saved after each outer country, but
automatic resume is not implemented. A manifest blocks accidental overwrite;
restart with a fresh session if a full rerun is needed. Do not interpret partial
country results as final or change settings based on them.

## Cell 6: review results and completion state

```python
import json
import pandas as pd

manifest = json.loads((OUT / "05d_manifest.json").read_text())
assert manifest["status"] == "complete"
assert manifest["holdout_loaded"] is False
loco = pd.read_csv(OUT / "05d_nested_loco.csv")
assert len(loco) == 6 and set(loco.country) == set(TRAIN_COUNTRIES)
display(loco[["country", "model_variant", "threshold_exact", "precision", "recall", "f1", "pr_auc"]])
display(pd.read_csv(OUT / "05d_loco_site_metrics.csv"))
display(pd.read_csv(OUT / "05d_inner_selection.csv"))
print("Macro-country F1:", loco.f1.mean())
print("Macro-country PR-AUC:", loco.pr_auc.mean())
print("Spatial blocks before/after site linking:",
      manifest["spatial_groups_before"], manifest["spatial_groups_after"])
```

## Cell 7: ZIP necessary outputs, not input datasets

```python
import zipfile
from IPython.display import FileLink

assert json.loads((OUT / "05d_manifest.json").read_text())["status"] == "complete"
required_outputs = [
    "05d_manifest.json", "05d_spatial_group_map.csv", "05d_nested_loco.csv",
    "05d_inner_selection.csv", "05d_loco_site_metrics.csv",
    "05d_development_variants.csv", "05d_development_countries.csv",
    "05d_development_sites.csv", "05d_feature_importance.csv",
]
required_cache = [
    "05d_loco_predictions.parquet", "05d_oof_predictions.parquet",
    "05d_final_foreign_model.txt",
]
files = [(OUT / name, f"outputs/{name}") for name in required_outputs]
files += [(CACHE / name, f"cache/{name}") for name in required_cache]
files += [(path, f"code/{path.name}") for path in sorted((repo / "kaggle").glob("*.py"))]
for path, _ in files:
    assert path.is_file(), f"Missing required output: {path}"

bundle = Path("/kaggle/working/nb5_nested_results.zip")
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path, name in files:
        archive.write(path, arcname=name)
print(bundle, f"{bundle.stat().st_size / 1024**2:.1f} MiB")
display(FileLink(str(bundle)))
```

Download `nb5_nested_results.zip` and keep the saved notebook log. Review the six
country table and site recovery before choosing any new experiment. The final
foreign model is an artifact for later review, not permission to score India.
