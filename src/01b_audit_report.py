"""Stage 01b: read cache/firms_all.parquet and answer the audit questions."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from common import *
pd.set_option("display.width", 200)

a = pd.read_csv(OUT / "01_file_audit.csv")
tc = sorted([c for c in a.columns if c.startswith("type")])

print("="*90); print("A. COVERAGE"); print("="*90)
print(a.pivot_table(index="country", columns="sensor", values="rows",
                    aggfunc="sum", margins=True, fill_value=0).astype(int).to_string())
print("\nyears per country:")
print(a.groupby("country").year.agg(lambda s: f"{s.min()}-{s.max()} ({s.nunique()}y)").to_string())
print("\nsensors per country:")
print(a.groupby("country").sensor.unique().to_string())

print("\n" + "="*90); print("B. DATA QUALITY"); print("="*90)
print(f"bad dates      : {a.bad_date.sum()}")
print(f"null cells     : {a.n_null.sum()}")
print(f"frp <= 0       : {a.frp_nonpos.sum()}")
print(f"exact dup rows : {a.dup_rows.sum():,}  ({a.dup_rows.sum()/a.rows.sum()*100:.3f}%)")
print(f"versions       : {sorted(set('|'.join(a.vers.astype(str)).split('|')))}")
print(f"satellites     : {sorted(set('|'.join(a.sats.astype(str)).split('|')))}")
print("\ncoordinate reuse (rows / unique lat-lon pairs):")
g = a.groupby("sensor")[["rows", "uniq_coord"]].sum()
g["reuse"] = (g.rows / g.uniq_coord).round(2)
print(g.to_string())

print("\n" + "="*90); print("C. NASA 'type' FIELD  (0=veg 1=active-volcano 2=other-static 3=offshore)"); print("="*90)
print(a[tc].sum().to_string())
gt = a.groupby("country")[tc].sum()
print("\nrow share by country (%):")
print((gt.div(gt.sum(1), 0) * 100).round(2).to_string())
print("\nby sensor (%):")
gs = a.groupby("sensor")[tc].sum()
print((gs.div(gs.sum(1), 0) * 100).round(2).to_string())

print("\n" + "="*90); print("D. CONFIDENCE VOCABULARY"); print("="*90)
for s, grp in a.groupby("sensor"):
    tot = {}
    for cv in grp.conf_vals:
        for kv in cv.split("|"):
            k, v = kv.rsplit(":", 1); tot[k] = tot.get(k, 0) + int(v)
    n = sum(tot.values())
    if len(tot) > 12:
        print(f"{s:11s} NUMERIC 0-100, {len(tot)} distinct values, n={n:,}")
    else:
        print(f"{s:11s} CATEGORICAL {  {k: f'{v/n*100:.1f}%' for k, v in sorted(tot.items())} }")

print("\n" + "="*90); print("E. VIIRS SATURATION + NIGHT SHARE"); print("="*90)
s = a.groupby("sensor")[["tmir_sat", "rows"]].sum()
print((s.tmir_sat / s.rows * 100).round(2).rename("t_mir>=367K %").to_string())
print("\nnight-detection share by country (%), row-weighted:")
print(a.groupby("country").apply(lambda d: np.average(d.dn_N_pct, weights=d.rows),
                                 include_groups=False).round(1).to_string())
print("\nmedian FRP by sensor (row-weighted over files):")
print(a.groupby("sensor").apply(lambda d: np.average(d.frp_med, weights=d.rows),
                                include_groups=False).round(2).to_string())
