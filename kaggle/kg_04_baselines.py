"""Stage 04: non-learned baselines. Anything later must beat these to justify itself."""
import sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from kg_common import *
from kg_eval import *
from kg_03_features import feature_cols

def run(train_countries=None):
    tr = load_features(train_countries or TRAIN_COUNTRIES)
    y = tr.is_eog_flare.values.astype(int)
    rows = []

    # --- 1. persistence rule: burns on many days, spread across the year ---
    for nd in [5, 10, 30, 60]:
        for me in [0.0, 0.5, 0.7]:
            p = ((tr.n_days >= nd) & (tr.month_entropy >= me)).astype(float).values
            rows.append(metrics(y, p, 0.5, name=f"rule n_days>={nd} & month_entropy>={me}"))

    # --- 2. continuous persistence score, threshold tuned for best F1 ---
    for col in ["n_days", "duty_cycle", "n_months", "frp_sum"]:
        s = tr[col].fillna(0).values.astype(float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        thr, _ = best_f1_threshold(y, s)
        rows.append(metrics(y, s, thr, name=f"score:{col}"))

    # --- 3. Isolation Forest (unsupervised anomaly ranking) ---
    fc = feature_cols(tr)
    X = StandardScaler().fit_transform(np.nan_to_num(tr[fc].values, nan=0.0))
    t0 = time.time()
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0, n_jobs=-1).fit(X)
    ts = time.time() - t0
    t0 = time.time(); s = -iso.score_samples(X); ti = time.time() - t0
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    thr, _ = best_f1_threshold(y, s)
    rows.append(metrics(y, s, thr, ts, ti, name="IsolationForest(all feats)"))

    df = log_experiment(rows, "04_baselines.csv")
    print(show(df.sort_values("f1", ascending=False)))
    print("\nNOTE: precision is a LOWER BOUND. Unlabelled sources include real industrial"
          "\nsites (kilns, cement, steel) that EOG does not cover, so some 'false positives'"
          "\nare probably correct detections.")
    return df

if __name__ == "__main__":
    run(sys.argv[1:] or None)
