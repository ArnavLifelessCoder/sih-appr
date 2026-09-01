"""Stage 05: LightGBM source-level baseline + feature-group ablations.

Validation is GroupKFold on block_id (never source_id) plus leave-country-out.
Labels are positive-unlabelled: unmatched sources are treated as negatives for
fitting, but reported precision is a LOWER BOUND.
"""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from kg_common import *
from kg_eval import *
from kg_03_features import feature_cols

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=-1, seed=0)
N_ROUNDS = 600

GROUPS = {
    "intensity":   ["frp_", "t_mir", "t_lwir", "dt_mir_lwir", "pix_km2", "sat_frac"],
    "temporal":    ["n_days", "n_months", "n_years", "span_days", "duty_cycle",
                    "det_per_day", "max_gap_days", "mean_gap_days", "n_det"],
    "seasonality": ["month_entropy", "month_max_share"],
    "timing":      ["lst_", "night_frac"],
    "cross_inst":  ["n_modis", "n_snpp", "snpp_modis_ratio", "n_sensors"],
    "confidence":  ["conf_"],
    "spatial":     ["spread_m", "n_pixels"],
}

def cols_for(all_cols, groups):
    keep = []
    for c in all_cols:
        for gname in groups:
            if any(c.startswith(p) for p in GROUPS[gname]):
                keep.append(c); break
    return sorted(set(keep))

def cv_predict(df, cols, params=PARAMS, rounds=N_ROUNDS, n_splits=5):
    X = df[cols].values.astype("float32"); y = df.is_eog_flare.values.astype(int)
    oof = np.zeros(len(df)); ts = ti = 0.0
    for tr_i, te_i in grouped_folds(df, n_splits):
        d = lgb.Dataset(X[tr_i], label=y[tr_i])
        t0 = time.time(); m = lgb.train(params, d, num_boost_round=rounds); ts += time.time() - t0
        t0 = time.time(); oof[te_i] = m.predict(X[te_i]); ti += time.time() - t0
    return oof, ts, ti

def run(train_countries=None, n_splits=5):
    tr = load_features(train_countries or TRAIN_COUNTRIES)
    y = tr.is_eog_flare.values.astype(int)
    allc = feature_cols(tr)
    rows = []

    # ---- full model ----
    oof, ts, ti = cv_predict(tr, allc, n_splits=n_splits)
    thr, _ = best_f1_threshold(y, oof)
    rows.append(metrics(y, oof, thr, ts, ti, name="LGBM full",
                        extra=dict(n_features=len(allc))))
    tr["oof_full"] = oof

    # ---- cumulative feature groups ----
    order = ["temporal", "intensity", "seasonality", "timing", "cross_inst",
             "confidence", "spatial"]
    cum = []
    for gname in order:
        cum.append(gname)
        cs = cols_for(allc, cum)
        o, a, b = cv_predict(tr, cs, n_splits=n_splits)
        t, _ = best_f1_threshold(y, o)
        rows.append(metrics(y, o, t, a, b, name="+".join(cum),
                            extra=dict(n_features=len(cs))))

    # ---- leave-one-group-out ----
    for gname in order:
        rest = [g for g in order if g != gname]
        cs = cols_for(allc, rest)
        o, a, b = cv_predict(tr, cs, n_splits=n_splits)
        t, _ = best_f1_threshold(y, o)
        rows.append(metrics(y, o, t, a, b, name=f"full minus {gname}",
                            extra=dict(n_features=len(cs))))

    df = log_experiment(rows, "05_lgbm_ablation.csv")
    print("=" * 130); print("LIGHTGBM  (GroupKFold on block_id)"); print("=" * 130)
    print(show(df, ["experiment", "n_features", "n_pos", "precision", "recall", "f1",
                    "pr_auc", "roc_auc", "mcc", "brier", "train_s", "infer_s"]))

    # ---- leave-country-out ----
    lco = []
    for c, tr_m, te_m in leave_country_out(tr):
        d = lgb.Dataset(tr.loc[tr_m, allc].values.astype("float32"),
                        label=y[tr_m])
        t0 = time.time(); m = lgb.train(PARAMS, d, num_boost_round=N_ROUNDS)
        ts = time.time() - t0
        t0 = time.time(); p = m.predict(tr.loc[te_m, allc].values.astype("float32"))
        ti = time.time() - t0
        yt = y[te_m]
        t, _ = best_f1_threshold(yt, p)
        lco.append(metrics(yt, p, t, ts, ti, name=f"LOCO holdout={c}"))
    ldf = log_experiment(lco, "05_lgbm_leave_country_out.csv")
    print("\n" + "=" * 130); print("LEAVE-ONE-COUNTRY-OUT"); print("=" * 130)
    print(show(ldf))

    # ---- importance + calibration on the full model ----
    d = lgb.Dataset(tr[allc].values.astype("float32"), label=y)
    m = lgb.train(PARAMS, d, num_boost_round=N_ROUNDS)
    imp = (pd.DataFrame({"feature": allc, "gain": m.feature_importance("gain")})
             .sort_values("gain", ascending=False))
    imp.to_csv(OUT / "05_feature_importance.csv", index=False)
    print("\n=== top 20 features by gain ===")
    print(imp.head(20).to_string(index=False))
    print("\n=== calibration of OOF predictions (10 bins) ===")
    print(calibration_table(y, tr.oof_full.values).round(4).to_string())
    tr[["source_id", "country", "block_id", "lat", "lon", "is_eog_flare",
        "oof_full"]].to_parquet(CACHE / "05_oof_predictions.parquet", index=False)
    return df, ldf

if __name__ == "__main__":
    run(sys.argv[1:] or None)
