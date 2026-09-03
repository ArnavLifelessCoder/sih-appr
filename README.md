# SIH26162: Persistent industrial thermal source detection

Separating industrial thermal sources (gas flares, kilns, furnaces) from natural and
agricultural fires, using NASA FIRMS active-fire detections plus contextual geospatial data.

## Data

Not committed (1.5 GB extracted). Upload the source ZIP as a Kaggle Dataset; `kg_common.py`
auto-discovers the root under `/kaggle/input`. Locally, set `SIH_DATA=/path/to/data`.

| Path | Contents | Role |
|---|---|---|
| `data/firms/` | 78 CSVs: MODIS (28), VIIRS S-NPP (28), VIIRS NOAA-20 (22) | model input |
| `data/eog/flare_inventory/` | World Bank / Payne Institute flare inventory, 2 XLSX | labels |
| `data/facilities/` | GEM (7 XLSX), OSM India (24 GeoJSON), WRI GPPD (1 CSV) | validation only |

### Verified audit numbers

19,342,072 FIRMS detections across 7 countries, 2019–2024. Zero nulls, zero malformed
dates, zero exact duplicate rows. 754 rows (0.004%) have `frp <= 0`.

| Country | MODIS | VIIRS_N20 | VIIRS_SNPP | Total | Years |
|---|---|---|---|---|---|
| India | 496,769 | 3,574,545 | 3,522,955 | 7,594,269 | 2019–24 |
| Angola | 1,039,703 | N/A | 3,830,013 | 4,869,716 | 2022–24 |
| Nigeria | 278,411 | 1,818,841 | 1,817,312 | 3,914,564 | 2021–24 |
| Iraq | 142,218 | 562,857 | 584,908 | 1,289,983 | 2021–24 |
| Algeria | 23,476 | 359,698 | 361,078 | 744,252 | 2021–24 |
| Indonesia | 86,977 | N/A | 414,794 | 501,771 | 2022–24 |
| Libya | 20,952 | 201,137 | 205,428 | 427,517 | 2021–24 |

EOG active flare sites, 2019–2024: Algeria 438, Nigeria 436, Iraq 266, Libya 202
(**1,342 training positives total**), India 193, Indonesia 370, Angola 73.

## Design decisions

**Unit of analysis is a source, not a detection row.** Libya collapses from 427,517
detections to 4,824 sources (89 detections each). Row-level modelling would train on
near-duplicates of the same physical object.

**Clustering is metric grid snapping at 1000 m**, chosen by a sweep over 500–5000 m:

| grid | sources (Libya/Algeria) | EOG recall | fragmentation |
|---|---|---|---|
| 500 m | 7,561 / 34,642 | 88% / 90% | 6.6x / 5.8x |
| **1000 m** | **4,824 / 19,122** | **88% / 92%** | **4.2x / 3.9x** |
| 2000 m | 3,263 / 10,861 | 83% / 90% | 2.9x / 2.9x |
| 5000 m | 1,700 / 4,768 | 74% / 77% | 2.1x / 2.1x |

Grid snapping is preferred over DBSCAN/agglomeration because 8-neighbour chaining merges
a flare into an adjacent wildfire front during India's burning season.

**Splitting is on 10 km `block_id`, never on `source_id`.** One physical flare fragments
into ~4 grid sources at any usable resolution, so fragments of the same site would
otherwise straddle a train/validation boundary and inflate scores.

**This is positive-unlabelled learning, not binary classification.** EOG covers gas flares
(>~1100 C) only. A brick kiln, cement plant or steel furnace is industrial but absent from
EOG, so an unmatched source is unlabelled, not negative.

## Known constraints

- **NOAA-20 is missing for Angola and Indonesia.** Raw cross-instrument detection ratios
  would encode country identity, which correlates with the label. Use a sensor-availability
  mask.
- **`facilities/` is India-only, and India is the untouched holdout.** Distance-to-infrastructure
  features therefore do not exist at training time and cannot be part of a training ablation.
- **Do not train on NASA `type`.** `type=2` is produced by NASA's own persistence mask;
  training persistence features against it reproduces the mask. `type=3` (offshore) is
  usable as a hard constraint.
- **Distance-to-nearest-EOG-flare is the label** and must never become a feature.
- ESA WorldCover and Sentinel-2/Landsat imagery are **not present** and must be sourced
  externally (GEE / Copernicus / Planetary Computer) before any CV branch.

## Country roles

Fixed by the data explainer; India is never used for fitting or model selection.

- Train (positives): Iraq, Algeria, Nigeria, Libya
- Train (background): Angola, Indonesia
- Holdout: India

## Layout

```
kaggle/kg_common.py               config, loaders, EOG label join
kaggle/kg_02_source_clustering.py detections -> sources, labels, CV blocks
kaggle/kg_03_features.py          FIRMS-only source feature construction
kaggle/kg_05_lightgbm.py          original full LightGBM and ablations
kaggle/kg_05b_robust_tabular.py   common-window, PU and corrected LOCO run
kaggle/kg_05c_balanced_tabular.py country-balanced weighting and threshold run
kaggle/kg_eval.py                 shared metrics, thresholds and split builders
src/common.py                     local-run equivalents
src/01_data_audit.py              streams 78 CSVs -> single parquet + audit table
src/01b_audit_report.py           audit summary
outputs/                          audit tables and source summaries
```

The completed robust run is documented in `KAGGLE_05B.md`. The next Kaggle
experiment is documented in `KAGGLE_05C.md`. Neither run loads or scores India.

## FIRMS gotchas

- `acq_time` is `"0136"` (HHMM). Read as string and zero-pad; reading as int corrupts it.
- NOAA-20 files are named `viirs-jpss1_*`; globbing `viirs-noaa20*` silently returns nothing.
- `confidence` is numeric 0–100 for MODIS and categorical `n`/`l`/`h` for VIIRS.
- MODIS `brightness`/`bright_t31` and VIIRS `bright_ti4`/`bright_ti5` unify to `t_mir`/`t_lwir`.
- VIIRS `bright_ti4` saturates hard at 367.0 K (2.0% MODIS, 3.7% N20, 6.1% S-NPP).
