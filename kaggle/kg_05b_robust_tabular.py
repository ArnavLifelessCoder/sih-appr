"""Stage 05b: robust common-window tabular evaluation.

This stage repairs the two main issues found after the full Stage 05 run:

1. Countries had different observation horizons. Features are rebuilt on the
   shared 2022-2024 window and EOG activity is restricted to the same years.
2. LOCO thresholds were selected on held-out labels. Thresholds are now learned
   from grouped OOF predictions on non-holdout countries only.

The stage compares a reduced binary LightGBM model with a bagged
positive-unlabelled LightGBM baseline. India is never loaded by ``run()``.

Writes:
  cache/features_<country>_2022_2024.parquet
  cache/05b_oof_predictions.parquet
  cache/05b_corrected_loco_predictions.parquet
  outputs/05b_robust_models.csv
  outputs/05b_corrected_loco.csv
  outputs/05b_feature_importance.csv
  outputs/05b_feature_manifest.json
"""
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from kg_common import *
from kg_eval import *
from kg_03_features import build as build_features, feature_cols


WINDOW_YEARS = (2022, 2023, 2024)
FEATURE_TAG = "2022_2024"

PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=-1,
    seed=17,
    feature_fraction_seed=17,
    bagging_seed=17,
    data_random_seed=17,
    force_col_wise=True,
)
N_ROUNDS = 500

# These columns either encode the old observation horizon directly or were
# harmful in the Stage 05 ablation. Their normalized replacements remain.
RAW_EXPOSURE_FEATURES = {
    "n_det", "n_days", "n_months", "n_years", "span_days", "frp_sum",
    "n_modis", "n_snpp", "n_pixels",
}
HARMFUL_FEATURES = {"month_entropy", "month_max_share"}


def build_common_features(countries=None):
    """Build common-window features for foreign countries only by default."""
    countries = countries or TRAIN_COUNTRIES
    rows = []
    for country in countries:
        d = build_features(
            country,
            years=WINDOW_YEARS,
            output_tag=FEATURE_TAG,
        )
        rows.append(dict(
            country=country,
            n=len(d),
            n_pos=int(d.is_eog_flare.sum()),
            eog_sites_recovered=int(
                d.loc[d.is_eog_flare == 1, "eog_flare_id"].nunique()
            ),
        ))
        del d
        gc.collect()
    report = pd.DataFrame(rows)
    report.to_csv(OUT / "05b_common_window_summary.csv", index=False)
    print("\nCOMMON-WINDOW FEATURE SUMMARY")
    print(report.to_string(index=False))
    return report


def robust_feature_cols(df):
    cols = [
        c for c in feature_cols(df)
        if c not in RAW_EXPOSURE_FEATURES | HARMFUL_FEATURES
    ]
    required = {
        "det_per_year", "active_days_per_year", "active_months_per_year",
        "frp_sum_per_year", "modis_per_year", "snpp_per_year",
        "span_window_frac",
    }
    missing = required - set(cols)
    if missing:
        raise ValueError(
            "common-window normalized features are missing: " +
            ", ".join(sorted(missing))
        )
    return sorted(cols)


def _train_binary(X, y, train_idx, params=None, rounds=N_ROUNDS):
    dtrain = lgb.Dataset(
        X[train_idx], label=y[train_idx], free_raw_data=True
    )
    return lgb.train(params or PARAMS, dtrain, num_boost_round=rounds)


def _predict_fold(X, y, train_idx, test_idx, mode="binary", pu_bags=3,
                  unlabeled_per_positive=10, seed=17, rounds=N_ROUNDS):
    """Fit one binary model or a bagged PU ensemble and predict one fold."""
    train_s = infer_s = 0.0
    if mode == "binary":
        t0 = time.time()
        model = _train_binary(X, y, train_idx, rounds=rounds)
        train_s += time.time() - t0
        t0 = time.time()
        pred = model.predict(X[test_idx])
        infer_s += time.time() - t0
        return pred, train_s, infer_s

    if mode != "pu_bagging":
        raise ValueError(f"unknown mode: {mode}")

    pos_idx = train_idx[y[train_idx] == 1]
    unl_idx = train_idx[y[train_idx] == 0]
    if not len(pos_idx):
        raise ValueError("PU fold has no labelled positives")
    n_unl = min(len(unl_idx), unlabeled_per_positive * len(pos_idx))
    pred = np.zeros(len(test_idx), dtype="float64")
    for bag in range(pu_bags):
        rng = np.random.default_rng(seed + bag)
        sampled_unl = rng.choice(unl_idx, size=n_unl, replace=False)
        sampled_train = np.concatenate([pos_idx, sampled_unl])
        rng.shuffle(sampled_train)
        bag_params = dict(PARAMS)
        bag_params.update(
            seed=seed + bag,
            feature_fraction_seed=seed + bag,
            bagging_seed=seed + bag,
            data_random_seed=seed + bag,
        )
        t0 = time.time()
        model = _train_binary(
            X, y, sampled_train, params=bag_params, rounds=rounds
        )
        train_s += time.time() - t0
        t0 = time.time()
        pred += model.predict(X[test_idx]) / pu_bags
        infer_s += time.time() - t0
    return pred, train_s, infer_s


def cv_predict(df, cols, mode="binary", n_splits=5, pu_bags=3,
               unlabeled_per_positive=10, seed=17, rounds=N_ROUNDS):
    X = df[cols].to_numpy(dtype="float32")
    y = df.is_eog_flare.to_numpy(dtype="int8")
    oof = np.zeros(len(df), dtype="float64")
    train_s = infer_s = 0.0
    for fold, (train_idx, test_idx) in enumerate(
            grouped_folds(df, n_splits=n_splits)):
        pred, ts, ins = _predict_fold(
            X, y, train_idx, test_idx, mode=mode, pu_bags=pu_bags,
            unlabeled_per_positive=unlabeled_per_positive,
            seed=seed + fold * 100,
            rounds=rounds,
        )
        oof[test_idx] = pred
        train_s += ts
        infer_s += ins
    return oof, train_s, infer_s


def fit_predict(train_df, test_df, cols, mode="binary", pu_bags=3,
                unlabeled_per_positive=10, seed=17, rounds=N_ROUNDS):
    joined = pd.concat([train_df, test_df], ignore_index=True)
    X = joined[cols].to_numpy(dtype="float32")
    y = joined.is_eog_flare.to_numpy(dtype="int8")
    train_idx = np.arange(len(train_df), dtype="int64")
    test_idx = np.arange(len(train_df), len(joined), dtype="int64")
    pred, train_s, infer_s = _predict_fold(
        X, y, train_idx, test_idx, mode=mode, pu_bags=pu_bags,
        unlabeled_per_positive=unlabeled_per_positive, seed=seed,
        rounds=rounds,
    )
    return pred, train_s, infer_s


def corrected_loco(df, cols, inner_splits=3, rounds=N_ROUNDS):
    """Strict nested LOCO with a threshold learned without the holdout."""
    rows = []
    predictions = []
    for country in sorted(df.country.unique()):
        train_df = df[df.country != country].reset_index(drop=True)
        test_df = df[df.country == country].reset_index(drop=True)

        inner_oof, inner_train_s, _ = cv_predict(
            train_df, cols, mode="binary", n_splits=inner_splits,
            seed=1000 + len(rows) * 100,
            rounds=rounds,
        )
        inner_y = train_df.is_eog_flare.to_numpy(dtype="int8")
        threshold, inner_f1 = best_f1_threshold(inner_y, inner_oof)

        pred, final_train_s, infer_s = fit_predict(
            train_df, test_df, cols, mode="binary",
            seed=2000 + len(rows) * 100,
            rounds=rounds,
        )
        test_y = test_df.is_eog_flare.to_numpy(dtype="int8")
        rows.append(metrics(
            test_y,
            pred,
            threshold,
            train_s=inner_train_s + final_train_s,
            infer_s=infer_s,
            name=f"corrected LOCO holdout={country}",
            extra=dict(
                threshold_source=(
                    f"{inner_splits}-fold grouped OOF on non-holdout countries"
                ),
                inner_oof_f1=inner_f1,
                inner_n=len(train_df),
                inner_n_pos=int(inner_y.sum()),
            ),
        ))
        predictions.append(pd.DataFrame({
            "source_id": test_df.source_id,
            "country": country,
            "is_eog_flare": test_y,
            "score": pred,
            "threshold": threshold,
            "predicted_positive": pred >= threshold,
        }))
        print(
            f"LOCO {country}: threshold={threshold:.4f}, "
            f"PR-AUC={rows[-1]['pr_auc']:.4f}, F1={rows[-1]['f1']:.4f}",
            flush=True,
        )
        del train_df, test_df, inner_oof, pred
        gc.collect()

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "05b_corrected_loco.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        CACHE / "05b_corrected_loco_predictions.parquet", index=False
    )
    return result


def run(n_splits=5, inner_splits=3, pu_bags=3,
        unlabeled_per_positive=10, rounds=N_ROUNDS):
    """Evaluate foreign countries only. India is deliberately not loaded."""
    df = load_features(TRAIN_COUNTRIES, tag=FEATURE_TAG)
    if HOLDOUT in set(df.country):
        raise AssertionError("India entered Stage 05b training data")
    cols = robust_feature_cols(df)
    y = df.is_eog_flare.to_numpy(dtype="int8")

    rows = []
    oof_frame = df[[
        "source_id", "country", "block_id", "lat", "lon", "is_eog_flare"
    ]].copy()
    thresholds = {}

    for mode, label in [
        ("binary", "reduced LightGBM"),
        ("pu_bagging", "bagged PU LightGBM"),
    ]:
        pred, train_s, infer_s = cv_predict(
            df,
            cols,
            mode=mode,
            n_splits=n_splits,
            pu_bags=pu_bags,
            unlabeled_per_positive=unlabeled_per_positive,
            rounds=rounds,
        )
        threshold, _ = best_f1_threshold(y, pred)
        thresholds[mode] = threshold
        rows.append(metrics(
            y,
            pred,
            threshold,
            train_s=train_s,
            infer_s=infer_s,
            name=label,
            extra=dict(
                n_features=len(cols),
                window="2022-2024",
                threshold_source=(
                    f"{n_splits}-fold grouped OOF on foreign countries"
                ),
                pu_bags=pu_bags if mode == "pu_bagging" else 0,
                unlabeled_per_positive=(
                    unlabeled_per_positive if mode == "pu_bagging" else 0
                ),
            ),
        ))
        oof_frame[f"score_{mode}"] = pred
        print(f"\n{label}\n{show(pd.DataFrame(rows[-1:]))}", flush=True)

    model_results = pd.DataFrame(rows)
    model_results.to_csv(OUT / "05b_robust_models.csv", index=False)
    oof_frame.to_parquet(CACHE / "05b_oof_predictions.parquet", index=False)

    loco = corrected_loco(
        df, cols, inner_splits=inner_splits, rounds=rounds
    )

    X = df[cols].to_numpy(dtype="float32")
    model = _train_binary(
        X, y, np.arange(len(df), dtype="int64"), rounds=rounds
    )
    importance = (pd.DataFrame({
        "feature": cols,
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False))
    importance.to_csv(OUT / "05b_feature_importance.csv", index=False)

    manifest = dict(
        stage="05b_robust_tabular",
        observation_years=list(WINDOW_YEARS),
        training_countries=TRAIN_COUNTRIES,
        holdout_country=HOLDOUT,
        holdout_loaded=False,
        model_sensors=["MODIS", "VIIRS_SNPP"],
        features=cols,
        excluded_raw_exposure_features=sorted(RAW_EXPOSURE_FEATURES),
        excluded_harmful_features=sorted(HARMFUL_FEATURES),
        lightgbm_params=PARAMS,
        num_boost_round=rounds,
        grouped_cv_folds=n_splits,
        corrected_loco_inner_folds=inner_splits,
        pu_bags=pu_bags,
        unlabeled_per_positive=unlabeled_per_positive,
        grouped_oof_thresholds=thresholds,
        india_policy=(
            "Do not build India common-window features or run India inference "
            "until a foreign-country model and threshold are selected."
        ),
    )
    with open(OUT / "05b_feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nCORRECTED LEAVE-ONE-COUNTRY-OUT")
    print(show(loco))
    print("\nTop 20 reduced-model features")
    print(importance.head(20).to_string(index=False))
    return model_results, loco


def main():
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if action in {"features", "all"}:
        build_common_features()
    if action in {"evaluate", "all"}:
        run()
    if action not in {"features", "evaluate", "all"}:
        raise SystemExit("usage: kg_05b_robust_tabular.py [features|evaluate|all]")


if __name__ == "__main__":
    main()
