# Kaggle notebooks

These notebooks are self-contained entry points for the main completed runs. Reusable
implementations live in `../kaggle/`, while the exact input and runtime instructions live
in `../docs/runs/`.

| Notebook | Purpose | Status |
|---|---|---|
| `kaggle/04_cv_pilot.ipynb` | Three-source smoke test and bounded imagery pilot | completed |
| `kaggle/06_image_context.ipynb` | Resumable Sentinel-2 and WorldCover acquisition and feature extraction | completed as NB6 v2 |
| `kaggle/08_domain_revamp.ipynb` | Regularized country-transfer tabular variants | completed, superseded |
| `kaggle/09_final_fusion.ipynb` | Thermal, image, early-fusion, and late-fusion comparison | completed, superseded |
| `kaggle/11_cv_ts_cascade.ipynb` | Tabular, temporal-descriptor, and TCN cascade | completed, TCN rejected |
| `kaggle/12_cv_tabular.ipynb` | CV plus tabular discovery experiment | completed |
| `kaggle/12b_guarded_confirmation.ipynb` | Fresh-seed confirmation and final artifact packaging | completed, frozen final branch |

Stages 05b, 05c, and 05d are represented by their source modules and exact runbooks rather
than separate clean notebook exports. The historical India run is implemented by
`../kaggle/kg_09_final_india.py`; its retained metrics are under
`../results/nb10_india_historical/`.

Do not attach multiple generations of the same notebook output to one Kaggle run. Duplicate
source or cache filenames intentionally fail preflight because they make provenance unclear.
