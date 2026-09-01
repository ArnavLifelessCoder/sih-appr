"""Stage 01: audit every FIRMS CSV and stream all 78 into one parquet.

Streams file-by-file through a single ParquetWriter, so peak memory is one CSV.
Outputs: outputs/01_file_audit.csv, cache/firms_all.parquet
"""
import sys, os, time, gc; sys.path.insert(0, "src")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from common import *

SCHEMA = pa.schema([
    ("latitude", pa.float64()), ("longitude", pa.float64()),
    ("t_mir", pa.float32()), ("t_lwir", pa.float32()), ("frp", pa.float32()),
    ("scan", pa.float32()), ("track", pa.float32()),
    ("acq_date", pa.date32()), ("acq_time", pa.int16()),
    ("satellite", pa.string()), ("confidence", pa.string()), ("version", pa.string()),
    ("daynight", pa.string()), ("type", pa.int8()),
    ("sensor", pa.string()), ("country", pa.string()), ("year", pa.int16()),
])
COLS = [f.name for f in SCHEMA]

def main():
    t0 = time.time(); recs, schemas = [], {}
    outp = CACHE / os.environ.get("FIRMS_PARQUET", "firms_all.parquet")
    w = pq.ParquetWriter(outp, SCHEMA, compression="zstd")
    for f in firms_files():
        df = read_firms_csv(f["path"])
        schemas.setdefault(tuple(df.columns), []).append(f["path"].name)
        dt = pd.to_datetime(df["acq_date"], format="%Y-%m-%d", errors="coerce")
        vc = df["type"].value_counts()
        recs.append(dict(
            file=f["path"].name, sensor=f["sensor"], country=f["country"], year=f["year"],
            rows=len(df), date_min=str(dt.min().date()), date_max=str(dt.max().date()),
            n_days=int(dt.dt.date.nunique()), bad_date=int(dt.isna().sum()),
            n_null=int(df.isna().sum().sum()), frp_nonpos=int((df.frp <= 0).sum()),
            dup_rows=int(df.duplicated().sum()),
            uniq_coord=int(df.groupby(["latitude", "longitude"]).ngroups),
            lat_min=df.latitude.min(), lat_max=df.latitude.max(),
            lon_min=df.longitude.min(), lon_max=df.longitude.max(),
            frp_med=float(df.frp.median()), frp_max=float(df.frp.max()),
            tmir_med=float(df.t_mir.median()), tmir_max=float(df.t_mir.max()),
            tmir_sat=int((df.t_mir >= 367.0).sum()),
            conf_vals="|".join(f"{k}:{v}" for k, v in
                               sorted(df.confidence.astype(str).value_counts().items())),
            sats="|".join(sorted(df.satellite.dropna().unique())),
            vers="|".join(sorted(df.version.dropna().unique().astype(str))),
            dn_N_pct=round((df.daynight == "N").mean() * 100, 2),
            **{f"type{int(k)}": int(v) for k, v in vc.items()}))
        df["acq_date"] = dt.dt.date
        df["acq_time"] = df["acq_time"].str.zfill(4).astype("int16")
        df["sensor"] = f["sensor"]; df["country"] = f["country"]
        df["year"] = np.int16(f["year"]); df["type"] = df["type"].astype("int8")
        w.write_table(pa.Table.from_pandas(df[COLS], schema=SCHEMA, preserve_index=False))
        print(f"  {f['path'].name:38s} {len(df):>9,}", flush=True)
        del df, dt; gc.collect()
    w.close()
    a = pd.DataFrame(recs)
    for c in [c for c in a.columns if c.startswith("type")]:
        a[c] = a[c].fillna(0).astype(int)
    a.to_csv(OUT / os.environ.get("FIRMS_AUDIT_CSV", "01_file_audit.csv"), index=False)
    print(f"\n{len(a)} files, {a.rows.sum():,} rows, {time.time()-t0:.0f}s")
    for c, fl in schemas.items():
        print(f"schema n={len(fl)}: {list(c)}")

if __name__ == "__main__":
    main()
