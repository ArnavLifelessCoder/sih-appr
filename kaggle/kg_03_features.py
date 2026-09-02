"""Stage 03: source-level features from FIRMS only.

Leakage rules enforced here:
  * NO raw lat/lon         -- near-perfect country proxy, and country correlates with label
  * NO country column
  * NO NASA `type`         -- type=2 IS a persistence mask; using it reproduces the mask
  * NO eog_dist_m          -- that IS the label
  * NO NOAA-20             -- absent for Angola and Indonesia, so any count that includes it
                              encodes country identity. Model uses MODIS + VIIRS S-NPP only,
                              which every country has. N20 is kept for India-side evidence.

Writes cache/features_<country>.parquet
"""
import sys, time
import numpy as np, pandas as pd
from kg_common import *
from kg_02_source_clustering import attach_eog_labels, to_local_m

MODEL_SENSORS = ["MODIS", "VIIRS_SNPP"]      # uniform across all 7 countries
FEATURE_BLOCKLIST = {"lat", "lon", "x", "y", "country", "type", "eog_dist_m",
                     "is_eog_flare", "eog_flare_id", "eog_offshore", "block_id",
                     "source_id", "date_min", "date_max"}

def _circ(hours):
    """circular mean/std of an hour-of-day array, in hours"""
    a = hours.to_numpy() * (2 * np.pi / 24.0)
    s, c = np.sin(a).mean(), np.cos(a).mean()
    R = np.hypot(s, c)
    mean = (np.arctan2(s, c) % (2 * np.pi)) * 24 / (2 * np.pi)
    std = np.sqrt(max(-2 * np.log(max(R, 1e-12)), 0)) * 24 / (2 * np.pi)
    return mean, std

def build(country, sensors=MODEL_SENSORS, years=None, output_tag=None):
    """Build source features, optionally inside a fixed observation window.

    When ``years`` is provided, detections and EOG activity are restricted to
    exactly those years. Window-normalized persistence features are added and
    the output is written as ``features_<country>_<output_tag>.parquet``.
    """
    t0 = time.time()
    det = pd.read_parquet(CACHE / f"detections_{country}.parquet")
    det = det[det.sensor.isin(sensors)].copy()
    if years is not None:
        years = tuple(sorted(set(int(y) for y in years)))
        if not years:
            raise ValueError("years cannot be empty")
        det = det[det.year.isin(years)].copy()
    if det.empty:
        raise ValueError(
            f"no detections for {country} with sensors={sensors}, years={years}")

    det["day"]      = det.acq_dt.values.astype("datetime64[D]")
    det["doy"]      = det.acq_dt.dt.dayofyear
    det["month"]    = det.acq_dt.dt.month
    det["pix_km2"]  = det.scan * det.track
    det["frp_dens"] = det.frp / det.pix_km2
    det["dt_mir_lwir"] = det.t_mir - det.t_lwir          # separates flares from veg fires
    det["sat"]      = (det.t_mir >= 367.0).astype("int8")
    det["is_night"] = (det.daynight.astype(str) == "N").astype("int8")
    # local solar time = UTC hour + lon/15
    det["lst"] = ((det.acq_time // 100 + (det.acq_time % 100) / 60.0)
                  + det.longitude / 15.0) % 24.0

    g = det.groupby("source_id", observed=True)
    f = pd.DataFrame(index=g.size().index)
    f["n_det"] = g.size()

    # ---- intensity ----
    for col in ["frp", "frp_dens", "t_mir", "t_lwir", "dt_mir_lwir"]:
        a = g[col]
        f[f"{col}_mean"] = a.mean(); f[f"{col}_max"] = a.max()
        f[f"{col}_std"]  = a.std().fillna(0); f[f"{col}_med"] = a.median()
    f["frp_p90"] = g.frp.quantile(0.90)
    f["frp_cv"]  = f.frp_std / f.frp_mean.replace(0, np.nan)
    f["frp_sum"] = g.frp.sum()
    f["pix_km2_mean"] = g.pix_km2.mean()

    # ---- saturation / day-night ----
    f["sat_frac"]   = g.sat.mean()
    f["night_frac"] = g.is_night.mean()

    # ---- temporal persistence ----
    f["n_days"]   = g.day.nunique()
    f["n_months"] = g.acq_dt.apply(lambda s: s.dt.to_period("M").nunique())
    f["n_years"]  = g.year.nunique()
    dmin, dmax = g.day.min(), g.day.max()
    f["span_days"]  = (dmax - dmin).dt.days.astype("float32")
    f["duty_cycle"] = f.n_days / f.span_days.replace(0, np.nan)
    f["det_per_day"] = f.n_det / f.n_days

    # max gap between consecutive active days
    ud = det[["source_id", "day"]].drop_duplicates().sort_values(["source_id", "day"])
    gap = ud.groupby("source_id", observed=True).day.diff().dt.days
    f["max_gap_days"]  = gap.groupby(ud.source_id.values).max().reindex(f.index).fillna(0)
    f["mean_gap_days"] = gap.groupby(ud.source_id.values).mean().reindex(f.index).fillna(0)

    # ---- seasonality: entropy over months (flat = year-round = industrial-like) ----
    mh = (det.groupby(["source_id", "month"], observed=True).size()
             .unstack(fill_value=0).reindex(f.index, fill_value=0))
    p = mh.div(mh.sum(1).replace(0, np.nan), axis=0).fillna(0)
    f["month_entropy"] = -(p * np.log(p.where(p > 0, 1))).sum(1) / np.log(12)
    f["month_max_share"] = p.max(1)

    # ---- overpass timing ----
    lst = g.lst.apply(lambda s: pd.Series(_circ(s), index=["m", "s"]))
    f["lst_mean"] = lst.xs("m", level=1); f["lst_std"] = lst.xs("s", level=1)

    # ---- cross-instrument (MODIS vs S-NPP only; both exist for all countries) ----
    sc = (det.groupby(["source_id", "sensor"], observed=True).size()
             .unstack(fill_value=0).reindex(f.index, fill_value=0))
    for s in sensors:
        if s not in sc: sc[s] = 0
    f["n_modis"] = sc["MODIS"]; f["n_snpp"] = sc["VIIRS_SNPP"]
    # small source -> resolved by 375 m VIIRS, diluted below MODIS threshold -> high ratio
    f["snpp_modis_ratio"] = (f.n_snpp + 1) / (f.n_modis + 1)
    f["n_sensors"] = (sc[sensors] > 0).sum(1)

    # ---- confidence (two different vocabularies) ----
    cf = det.confidence.astype(str)
    mm = det.sensor.astype(str) == "MODIS"
    f["conf_modis_mean"] = (pd.to_numeric(cf.where(mm), errors="coerce")
                            .groupby(det.source_id.values).mean().reindex(f.index))
    for lv in ["n", "l", "h"]:
        f[f"conf_viirs_{lv}_frac"] = ((cf == lv).astype("int8")
                                      .groupby(det.source_id.values).mean().reindex(f.index))

    # ---- spatial spread within the source, in metres ----
    latm = g.latitude.std().fillna(0) * 111_320.0
    lonm = g.longitude.std().fillna(0) * 111_320.0 * np.cos(np.radians(g.latitude.mean()))
    f["spread_m"] = np.hypot(latm, lonm)
    f["n_pixels"] = det.groupby("source_id", observed=True).apply(
        lambda d: len(d[["latitude", "longitude"]].drop_duplicates()), include_groups=False)

    if years is not None:
        # Exposure-normalized alternatives to raw counts. The robust model
        # blocklists the raw versions and consumes these columns instead.
        n_window_years = float(len(years))
        window_start = pd.Timestamp(min(years), 1, 1)
        window_end = pd.Timestamp(max(years), 12, 31)
        window_days = float((window_end - window_start).days + 1)
        f["det_per_year"] = f.n_det / n_window_years
        f["active_days_per_year"] = f.n_days / n_window_years
        f["active_months_per_year"] = f.n_months / n_window_years
        f["frp_sum_per_year"] = f.frp_sum / n_window_years
        f["modis_per_year"] = f.n_modis / n_window_years
        f["snpp_per_year"] = f.n_snpp / n_window_years
        f["span_window_frac"] = f.span_days / max(window_days - 1.0, 1.0)

    f = f.replace([np.inf, -np.inf], np.nan).astype("float32").reset_index()

    src = pd.read_parquet(CACHE / f"sources_{country}.parquet")
    if years is None:
        out = f.merge(src[["source_id", "block_id", "country", "lat", "lon",
                           "is_eog_flare", "eog_flare_id", "eog_dist_m", "eog_offshore"]],
                      on="source_id", how="left")
    else:
        # Recompute source centroids and EOG labels inside the common window.
        # Reuse the original 10 km block_id so the spatial grouping remains
        # anchored to the Stage 02 grid.
        geo = (det.groupby("source_id", observed=True)
                 .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
                 .reset_index())
        geo = geo.merge(src[["source_id", "block_id"]], on="source_id", how="left")
        geo["country"] = country
        lat0, lon0 = float(det.latitude.mean()), float(det.longitude.mean())
        geo["x"], geo["y"] = to_local_m(
            geo.lat.values, geo.lon.values, lat0, lon0)
        geo = attach_eog_labels(
            geo, country, lat0, lon0, active_years=years)
        out = f.merge(
            geo[["source_id", "block_id", "country", "lat", "lon",
                 "is_eog_flare", "eog_flare_id", "eog_dist_m", "eog_offshore"]],
            on="source_id", how="left")

    suffix = f"_{output_tag}" if output_tag else ""
    out.to_parquet(CACHE / f"features_{country}{suffix}.parquet", index=False)
    print(f"{country}: {len(out):,} sources x {len(feature_cols(out))} features "
          f"({int(out.is_eog_flare.sum()):,} pos), years={years or 'all'} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return out

def feature_cols(df):
    return [c for c in df.columns if c not in FEATURE_BLOCKLIST]

def main(countries=None):
    for c in (countries or TRAIN_COUNTRIES + [HOLDOUT]):
        build(c)

if __name__ == "__main__":
    main(sys.argv[1:] or None)
