"""Stage 02: collapse FIRMS detections into persistent thermal SOURCES, and attach
EOG labels at source level.

Unit of analysis = one physical thermal source, not one detection row.

Primary method is metric grid snapping (deterministic, order-independent, and it
CANNOT chain -- important, because 8-neighbour agglomeration merges a flare into an
adjacent wildfire front during India's burning season). DBSCAN is run as a
sensitivity check only.

Writes: cache/sources_<country>.parquet, cache/detections_<country>.parquet
"""
import sys, time, json
import numpy as np, pandas as pd
from kg_common import *

GRID_M      = 1000       # source cell size in metres (chosen by sensitivity sweep)
LABEL_R_M   = 1000       # EOG match radius, metres
BLOCK_M     = 10_000     # spatial block used as the CV grouping unit (anti-leakage)
R_EARTH     = 6_371_000.0

def to_local_m(lat, lon, lat0, lon0):
    """Equirectangular projection to metres about (lat0, lon0). Accurate to <0.1%
    over a single country, which is far below the 500 m grid."""
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y

def grid_ids(x, y, cell):
    ix = np.floor(x / cell).astype("int64")
    iy = np.floor(y / cell).astype("int64")
    return ix, iy, (ix + 2_000_000) * 4_000_000 + (iy + 2_000_000)

def attach_eog_labels(src, country, lat0, lon0, grid_m=GRID_M,
                      label_r=LABEL_R_M, active_years=range(2019, 2025)):
    """Attach nearest-site EOG labels to a source table.

    ``src`` must contain ``x`` and ``y`` coordinates in the local projection
    defined by ``lat0`` and ``lon0``. Restricting ``active_years`` is essential
    for common-window experiments: a flare active outside the observation
    window must not become a positive label inside it.
    """
    src = src.copy()
    e = eog_sites(active_years=active_years)
    e = e[e.country == country].reset_index(drop=True)
    if not len(e):
        src["eog_dist_m"] = np.inf
        src["is_eog_flare"] = np.int8(0)
        src["eog_flare_id"] = None
        src["eog_offshore"] = False
        return src

    ex, ey = to_local_m(e.lat.values, e.lon.values, lat0, lon0)
    eix, eiy, _ = grid_ids(ex, ey, grid_m)
    buckets = {}
    for j, (a, b) in enumerate(zip(eix, eiy)):
        buckets.setdefault((a, b), []).append(j)

    six = np.floor(src.x.values / grid_m).astype("int64")
    siy = np.floor(src.y.values / grid_m).astype("int64")
    best_d = np.full(len(src), np.inf)
    best_j = np.full(len(src), -1)
    rad = int(np.ceil(label_r / grid_m))
    for dx in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            cand = {}
            for i, (a, b) in enumerate(zip(six + dx, siy + dy)):
                js = buckets.get((a, b))
                if js:
                    cand[i] = js
            for i, js in cand.items():
                d = np.hypot(ex[js] - src.x.values[i], ey[js] - src.y.values[i])
                k = int(np.argmin(d))
                if d[k] < best_d[i]:
                    best_d[i] = d[k]
                    best_j[i] = js[k]

    src["eog_dist_m"] = best_d
    src["is_eog_flare"] = (best_d <= label_r).astype("int8")
    safe_j = np.clip(best_j, 0, None)
    src["eog_flare_id"] = np.where(
        best_j >= 0, e.flare_id.values[safe_j], None)
    src["eog_offshore"] = np.where(
        best_j >= 0, e.location.values[safe_j] == "OFFSHORE", False)
    src.loc[src.is_eog_flare == 0, "eog_flare_id"] = None
    return src

def build_country(country, grid_m=GRID_M, label_r=LABEL_R_M, verbose=True):
    t0 = time.time()
    df = load_firms(countries=[country])
    lat0, lon0 = df.latitude.mean(), df.longitude.mean()
    x, y = to_local_m(df.latitude.values, df.longitude.values, lat0, lon0)
    ix, iy, key = grid_ids(x, y, grid_m)
    codes, uniq = pd.factorize(key)
    df["source_idx"] = codes
    df["_x"], df["_y"] = x, y

    n_src = len(uniq)
    if verbose:
        print(f"{country}: {len(df):,} detections -> {n_src:,} sources "
              f"({len(df)/n_src:.1f} det/source)  [{time.time()-t0:.0f}s]", flush=True)

    # ---- source-level geometry (features come later, in stage 03) ----
    g = df.groupby("source_idx", observed=True)
    src = pd.DataFrame({
        "n_det":   g.size(),
        "lat":     g.latitude.mean(),
        "lon":     g.longitude.mean(),
        "x":       g._x.mean(),
        "y":       g._y.mean(),
        "date_min": g.acq_dt.min(),
        "date_max": g.acq_dt.max(),
        "n_days":  g.acq_dt.apply(lambda s: s.dt.normalize().nunique()),
    })
    src["span_days"] = (src.date_max - src.date_min).dt.days
    src["country"] = country
    # Grid snapping fragments ONE physical flare into ~4 sources (measured: 3.9-4.2x).
    # Fragments of the same site must never straddle a train/val boundary, so all
    # splitting is done on these coarse blocks, never on source_id.
    bix = np.floor(src.x.values / BLOCK_M).astype("int64")
    biy = np.floor(src.y.values / BLOCK_M).astype("int64")
    src["block_id"] = country + "_b" + pd.Series((bix + 100000) * 1000000 + (biy + 100000)).astype(str)
    src["source_id"] = country + "_" + src.index.astype(str)
    src = src.reset_index(drop=True)

    # ---- EOG labels: match each source to the nearest active flare site ----
    src = attach_eog_labels(src, country, lat0, lon0, grid_m, label_r)

    det = df[["source_idx", "latitude", "longitude", "t_mir", "t_lwir", "frp", "scan",
              "track", "acq_dt", "acq_time", "satellite", "confidence", "daynight",
              "type", "sensor", "year"]].copy()
    det["source_id"] = src.source_id.values[det.source_idx.values]
    det = det.drop(columns=["source_idx"])

    src.to_parquet(CACHE / f"sources_{country}.parquet", index=False)
    det.to_parquet(CACHE / f"detections_{country}.parquet", index=False)
    return src

def main(countries=None):
    countries = countries or (TRAIN_COUNTRIES + [HOLDOUT])
    rows = []
    for c in countries:
        s = build_country(c)
        matched = int(s.is_eog_flare.sum())
        e_n = int((eog_sites().country == c).sum())
        rows.append(dict(
            country=c, detections=int(s.n_det.sum()), sources=len(s),
            det_per_source=round(s.n_det.sum() / len(s), 1),
            src_1det=int((s.n_det == 1).sum()),
            src_1det_pct=round((s.n_det == 1).mean() * 100, 1),
            src_ge5days=int((s.n_days >= 5).sum()),
            src_ge30days=int((s.n_days >= 30).sum()),
            src_multiyear=int((s.span_days >= 365).sum()),
            eog_sites=e_n, sources_matched=matched,
            eog_recovered=int(s.loc[s.is_eog_flare == 1, "eog_flare_id"].nunique()),
            eog_recall_pct=round(s.loc[s.is_eog_flare == 1, "eog_flare_id"].nunique()
                                 / max(e_n, 1) * 100, 1),
            pos_rate_pct=round(matched / len(s) * 100, 3)))
        print(pd.DataFrame(rows[-1:]).to_string(index=False), flush=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "02_source_summary.csv", index=False)
    print("\n" + "=" * 120); print("SOURCE-LEVEL SUMMARY"); print("=" * 120)
    print(rep.to_string(index=False))
    print("\nNOTE: sources with is_eog_flare=0 are UNLABELLED, not negative. EOG covers"
          "\ngas flares only (>~1100 C); kilns, cement and steel are industrial but absent.")
    return rep

if __name__ == "__main__":
    main(sys.argv[1:] or None)
