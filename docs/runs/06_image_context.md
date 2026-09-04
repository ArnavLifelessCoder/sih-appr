# NB6: bounded imagery acquisition and context features

Run independently of NB5. Name: `sih nb6 image context`. CPU, Internet On.
Attach NB2 full saved outputs and NB4 full saved outputs, not just their ZIPs.
No original EOG workbook, GPU, credentials or Indian data are needed.

The self-contained `notebooks/kaggle/06_image_context.ipynb` is ready to import
into Kaggle and includes its runtime modules. It does not need a GitHub push or
clone. The copy-paste alternative below uses the repository and requires these
NB6 files to be published first.

The first target is 600 foreign sources: 100 per country, up to 32 known-positive
sites plus random unlabelled controls from distinct spatial blocks. Countries
with fewer independent positive sites receive additional unlabelled controls.
This is an enriched development sample, not population ground truth.

Each acquisition run attempts at most 100 new sources and checks a 60-minute
budget between sources. A request already in progress may exceed the budget.
It stops with less than 1 GiB free disk or six consecutive acquisition failures.
This bounds work for a free-plan workflow without assuming a particular account
quota. Reused downloads do not count toward the 100 new-attempt limit.

## Cell 1: dependencies

```python
%pip install -q rasterio==1.4.4 numpy pandas pyarrow requests pyproj matplotlib pytest
```

## Cell 2: clone the published implementation

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
assert (repo / "kaggle/kg_07_context.py").exists(), "NB6 code must be pushed first."
subprocess.run(["git", "log", "-1", "--oneline"], cwd=repo, check=True)
sys.path.insert(0, str(repo / "kaggle"))

from kg_07_context import (
    PROTOCOL, prepare, run_batch, export_features, make_preview, bundle,
)
assert PROTOCOL == "nb6-context-v1"
```

## Cell 3: offline regression tests

```python
subprocess.run([
    sys.executable, "-m", "pytest", "tests/test_imagery_context.py",
    "-q", "-p", "no:cacheprovider",
], cwd=repo, check=True)
```

## Cell 4: freeze the source sample

```python
import pandas as pd

ROOT = Path("/kaggle/working/nb6_context_v1")
INPUT = Path("/kaggle/input")

sample = prepare(
    INPUT, ROOT,
    n_per_country=100,
    positive_per_country=32,
    seed=42,
)
display(sample.groupby("country").agg(
    sources=("source_id", "size"), eog_positive=("is_eog_flare", "sum")
))
assert len(sample) == 600
assert "India" not in set(sample.country)
```

## Cell 5: reuse existing NB4 or previous NB6 chips

```python
manifest = run_batch(ROOT, INPUT, max_new=0, offline=True)
display(pd.crosstab(manifest.country, manifest.status))
print("Reused successful chips:", int(manifest.status.eq("ok").sum()))
```

## Cell 6: bounded acquisition batch

```python
manifest = run_batch(
    ROOT,
    INPUT,
    max_new=100,
    max_minutes=60,
    retry_failed=False,
)
display(pd.crosstab(manifest.country, manifest.status))
display(manifest.loc[manifest.status.eq("failed"), ["country", "source_id", "error"]]
        if "error" in manifest else pd.DataFrame())
```

Do not enable repeated automatic execution. One call bounds this session's work.
If resources allow, manually running this cell again continues with pending
sources, skipping both successful and failed attempts. Set `retry_failed=True`
only after investigating the failure cause. Do not lower quality thresholds to
inflate the success count.

## Cell 7: extract features and inspect quality

```python
import json
from IPython.display import Image, display

features, quality = export_features(ROOT)
display(features.head())
display(pd.read_csv(ROOT / "coverage_by_country_label.csv"))
display(quality.loc[quality.review_reflectance_tail.fillna(False)])
print("Pixel-derived model features:", len(features.columns) - 1)
print(json.dumps(json.loads((ROOT / "run_state.json").read_text()), indent=2))
display(Image(filename=str(make_preview(ROOT))))
```

There are 77 model features plus `source_id` as the join key. Features summarize
six bands, NDVI/NDBI/MNDWI, and WorldCover fractions over the full 2 km square and
central 500 m square, plus distance to built-up pixels inside the chip. Band and
index summaries are p10, median and p90. Negative or near-zero-denominator index
pixels are excluded, not clipped into valid measurements.

Labels, country, coordinates, scene dates, quality fractions and download status
are not model features. Pending/failed sample rows remain in the feature table
with missing image features. Zero WorldCover coverage gives missing fractions,
not a fake background class. No built-up pixels gives a missing local distance,
not a claimed distance to the nearest city globally.

## Cell 8: package the completed work

```python
from IPython.display import FileLink

zip_path = bundle(ROOT)
print(zip_path, f"{zip_path.stat().st_size / 1024**2:.1f} MiB")
display(FileLink(str(zip_path)))
```

If all acquisitions fail, Cell 7 deliberately stops. Cell 8 can still be run to
package failure records. Successful chips, sidecars, partial manifests, image
features and acquisition source code are bundled. Input thermal datasets are not.

## Resume after saving a notebook version

Save the notebook outputs before the session ends. To continue in another
notebook, attach NB2 plus the latest NB6 full output. NB4 is optional if its
relevant chips were already reused. Keep the same sample settings and seed; run
the cells again. Cell 5 imports successful prior chips and compatible NB6 failure
records; Cell 6 attempts the next pending sources. ZIP creation alone does not
persist an unsaved Kaggle session.

## What happens after this notebook

Review coverage by country and label, acquisition failures and the country-wide
preview. Then compare thermal-only, image-context-only and combined models on
identical site-safe splits. A matched-chip comparison is not a population metric:
later reporting must also show full eligible-population coverage. Selection and
thresholds must stay inside training partitions, following NB5's corrected
protocol. Do not infer that a balanced sample has natural class prevalence.

NB6 fits no model and reports no accuracy gain. It prepares a reusable image
branch without committing to a CNN, stacking, or final India implementation.
WorldCover is from 2021; the single Sentinel-2 scenes are from 2022-2024, not
seasonal composites. SWIR data retain their native 20 m information despite the
common 10 m storage grid. Data collection remains subject to clear-sky selection
and incomplete facility labels.
