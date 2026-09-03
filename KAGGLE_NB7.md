# NB7: frozen final India run

Name: `sih nb7 final india`. Accelerator: None (CPU). Internet: On.

Run this beside NB6. Attach three inputs:

1. The complete `sih nb2` saved output containing `detections_India.parquet` and `sources_India.parquet`.
2. The complete NB5 saved output containing `05d_manifest.json` and `05d_final_foreign_model.txt`.
3. The original SIH dataset containing the EOG workbook.

NB7 rebuilds India on the same 2022 to 2024 window and applies the frozen NB5
model and threshold. It does not retrain, select features, or tune on India.

## Cell 1: dependencies

```python
%pip install -q lightgbm==4.6.0 scikit-learn pandas numpy scipy pyarrow pyproj openpyxl
```

## Cell 2: load the committed implementation

```python
from pathlib import Path
import os
import subprocess
import sys

repo = Path("/kaggle/working/sih")
if not repo.exists():
    subprocess.run([
        "git", "clone", "https://github.com/ArnavLifelessCoder/sih-appr.git", str(repo)
    ], check=True)
else:
    subprocess.run(["git", "pull", "--ff-only", "origin", "main"], cwd=repo, check=True)

sys.path.insert(0, str(repo / "kaggle"))
subprocess.run(["git", "log", "-1", "--oneline"], cwd=repo, check=True)
```

## Cell 3: locate the original data root

```python
input_root = Path("/kaggle/input")
workbook_name = "Flare-Volume-Estimates-by-individual-Flare-Location-2012-2025.xlsx"
workbooks = list(input_root.rglob(workbook_name))
assert len(workbooks) == 1, f"Attach exactly one original SIH dataset; found {workbooks}"
assert workbooks[0].parent.name == "flare_inventory"
assert workbooks[0].parents[1].name == "eog"
os.environ["SIH_DATA"] = str(workbooks[0].parents[2])

from kg_09_final_india import find_unique

print("India detections:", find_unique(input_root, "detections_India.parquet"))
print("India sources:", find_unique(input_root, "sources_India.parquet"))
print("NB5 manifest:", find_unique(input_root, "05d_manifest.json"))
print("NB5 model:", find_unique(input_root, "05d_final_foreign_model.txt"))
```

## Cell 4: run frozen India inference

```python
from kg_09_final_india import run

india_metrics, india_site_recall, india_predictions = run()
display(india_metrics)
display(india_site_recall)
display(
    india_predictions.nlargest(25, "eog_like_score")[
        ["source_id", "lat", "lon", "eog_like_score", "predicted_eog_like", "is_eog_flare"]
    ]
)
```

Do not alter the feature list or threshold after seeing this output. The reported
precision and F1 use incomplete EOG proxy labels. They are not verified industrial
source precision or F1.

## Cell 5: verify and package the complete output

```python
import json
import zipfile
from IPython.display import FileLink

root = Path("/kaggle/working/nb7_final_india")
manifest = json.loads((root / "09_manifest.json").read_text())
assert manifest["status"] == "complete"
assert manifest["model_action"] == "loaded frozen NB5 model; no fitting or tuning"
assert len(india_predictions) == manifest["n_sources"]
assert india_predictions.source_id.is_unique

required = [
    "09_manifest.json",
    "09_india_predictions.parquet",
    "09_review_top1000.csv",
    "09_india_eog_proxy_metrics.csv",
    "09_india_site_recall.csv",
]
bundle = Path("/kaggle/working/nb7_final_india.zip")
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in required:
        path = root / name
        assert path.is_file(), f"Missing output: {path}"
        archive.write(path, arcname=name)
print(bundle, f"{bundle.stat().st_size / 1024**2:.1f} MiB")
display(FileLink(str(bundle)))
```

Save a version with all outputs after the run completes. NB6 can continue in a
separate notebook. Its imagery features are a future comparison branch and do not
change this frozen thermal result.
