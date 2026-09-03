"""Foreign-only domain-generalization revamp for SIH 26162.

India is forbidden. Variants are selected inside each outer country holdout.
Country-relative ranks use feature values only, never labels.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from kg_05b_robust_tabular import FEATURE_TAG, WINDOW_YEARS, robust_feature_cols
from kg_05c_balanced_tabular import (
    active_eog_counts, country_metrics, macro_f1_threshold, site_metrics,
)
from kg_05d_nested_tabular import site_safe_blocks, validate_features
from kg_common import CACHE, HOLDOUT, OUT, TRAIN_COUNTRIES
from kg_eval import metrics


PROTOCOL_VERSION = "05e-domain-revamp-v1"
RANK_BASES = {
    "frp_mean", "frp_max", "frp_med", "frp_p90", "frp_std",
    "frp_dens_mean", "frp_dens_max", "frp_dens_med", "frp_dens_std",
    "t_mir_mean", "t_mir_max", "t_mir_med", "t_mir_std",
    "t_lwir_mean", "t_lwir_max", "t_lwir_med", "t_lwir_std",
    "dt_mir_lwir_mean", "dt_mir_lwir_max", "dt_mir_lwir_med",
    "dt_mir_lwir_std", "frp_sum_per_year",
}
COMPACT_RAW = {
    "active_days_per_year", "active_months_per_year", "det_per_day",
    "det_per_year", "duty_cycle", "span_window_frac", "mean_gap_days",
    "max_gap_days", "modis_per_year", "snpp_per_year", "n_sensors",
    "snpp_modis_ratio", "night_frac", "sat_frac", "lst_mean", "lst_std",
    "frp_cv",
}
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
    "max_depth": 8, "min_data_in_leaf": 100, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 1.0,
    "lambda_l2": 5.0, "max_bin": 127, "verbose": -1,
    "num_threads": -1, "force_col_wise": True,
}
VARIANTS = [
    {"name": "regularized_raw", "schema": "raw", "persistent_train": False},
    {"name": "ranked_physics", "schema": "ranked", "persistent_train": False},
    {"name": "ranked_compact", "schema": "compact", "persistent_train": False},
    {"name": "ranked_compact_persistent", "schema": "compact", "persistent_train": True},
]


def add_country_ranks(df: pd.DataFrame, raw_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if HOLDOUT in set(df.country):
        raise ValueError("India is forbidden during model development")
    ranked = df.copy()
    rank_bases = sorted(set(raw_cols) & RANK_BASES)
    for column in rank_bases:
        ranked[f"{column}_country_pct"] = ranked.groupby(
            "country", observed=True
        )[column].rank(method="average", pct=True).astype("float32")
    return ranked, rank_bases


def feature_schemas(
    df: pd.DataFrame, raw_columns: list[str] | None = None
) -> dict[str, list[str]]:
    raw = list(raw_columns) if raw_columns is not None else robust_feature_cols(df)
    ranked_bases = sorted(set(raw) & RANK_BASES)
    ranked = sorted(
        (set(raw) - set(ranked_bases) - {"pix_km2_mean", "spread_m"})
        | {f"{column}_country_pct" for column in ranked_bases}
    )
    compact = sorted(
        (set(raw) & COMPACT_RAW)
        | {f"{column}_country_pct" for column in ranked_bases}
    )
    schemas = {"raw": raw, "ranked": ranked, "compact": compact}
    for name, columns in schemas.items():
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Invalid {name} schema")
        if {"country", "lat", "lon", "type", "eog_dist_m"} & set(columns):
            raise ValueError(f"Leakage in {name} schema")
    return schemas


def _folds(df: pd.DataFrame, n_splits: int):
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=n_splits).split(
        df, df.is_eog_flare.to_numpy(), groups=df.block_id.to_numpy()
    )


def _model(df, columns, train_idx, rounds, seed, persistent_train):
    train_idx = np.asarray(train_idx, dtype="int64")
    if persistent_train:
        train_idx = train_idx[df.iloc[train_idx].n_days.to_numpy() >= 2]
    y = df.is_eog_flare.to_numpy(dtype="int8")
    if y[train_idx].sum() == 0:
        raise ValueError("Training fold has no positives")
    params = dict(PARAMS)
    params.update({key: seed for key in [
        "seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"
    ]})
    dataset = lgb.Dataset(
        df.iloc[train_idx][columns].to_numpy(dtype="float32"),
        label=y[train_idx], free_raw_data=True,
    )
    return lgb.train(params, dataset, num_boost_round=rounds), len(train_idx)


def cv_predict(df, columns, variant, n_splits=3, rounds=400, seed=71):
    score = np.empty(len(df), dtype="float64")
    train_rows = []
    for fold, (train_idx, test_idx) in enumerate(_folds(df, n_splits)):
        model, used = _model(
            df, columns, train_idx, rounds, seed + fold * 100,
            variant["persistent_train"],
        )
        score[test_idx] = model.predict(
            df.iloc[test_idx][columns].to_numpy(dtype="float32")
        )
        train_rows.append(used)
    return score, train_rows


def ensemble_predict(train, test, columns, variant, rounds, seeds):
    score = np.zeros(len(test), dtype="float64")
    used = []
    for seed in seeds:
        model, count = _model(
            train, columns, np.arange(len(train)), rounds, seed,
            variant["persistent_train"],
        )
        score += model.predict(test[columns].to_numpy(dtype="float32")) / len(seeds)
        used.append(count)
    return score, used


def evaluate_variants(df, schemas, n_splits=3, rounds=400, seed=71):
    rows, predictions = [], {}
    y = df.is_eog_flare.to_numpy(dtype="int8")
    countries = df.country.to_numpy()
    for variant in VARIANTS:
        columns = schemas[variant["schema"]]
        started = time.time()
        score, train_rows = cv_predict(
            df, columns, variant, n_splits=n_splits, rounds=rounds, seed=seed
        )
        threshold, macro_f1 = macro_f1_threshold(y, score, countries)
        country = country_metrics(df, score, threshold, variant["name"])
        row = metrics(y, score, threshold, name=variant["name"], extra={
            "schema": variant["schema"],
            "persistent_train": variant["persistent_train"],
            "n_features": len(columns), "threshold_exact": threshold,
            "macro_country_f1": macro_f1,
            "macro_country_pr_auc": float(country.pr_auc.mean()),
            "worst_country_pr_auc": float(country.pr_auc.min()),
            "mean_training_rows": float(np.mean(train_rows)),
            "elapsed_s": time.time() - started,
        })
        rows.append(row)
        predictions[variant["name"]] = score
        print(
            f"{variant['name']}: macro AP={row['macro_country_pr_auc']:.4f}, "
            f"macro F1={macro_f1:.4f}, pooled AP={row['pr_auc']:.4f}", flush=True
        )
    summary = pd.DataFrame(rows).sort_values(
        ["macro_country_pr_auc", "macro_country_f1"], ascending=False
    ).reset_index(drop=True)
    winner = next(v for v in VARIANTS if v["name"] == summary.iloc[0].experiment)
    return summary, winner, predictions


def nested_loco(df, schemas, inner_splits=3, rounds=400, seed=71, checkpoint=None):
    results, selections, predictions = [], [], []
    totals = active_eog_counts(sorted(df.country.unique()))
    site_rows = []
    for index, country in enumerate(sorted(df.country.unique())):
        train = df[df.country.ne(country)].reset_index(drop=True)
        test = df[df.country.eq(country)].reset_index(drop=True)
        inner_seed = seed + 10_000 + index * 1_000
        summary, winner, _ = evaluate_variants(
            train, schemas, inner_splits, rounds, inner_seed
        )
        selected = summary.iloc[0]
        summary["held_out_country"] = country
        summary["selected"] = summary.experiment.eq(winner["name"])
        selections.append(summary)
        seeds = [seed + 20_000 + index * 1_000 + offset for offset in (0, 101, 202)]
        columns = schemas[winner["schema"]]
        score, used = ensemble_predict(train, test, columns, winner, rounds, seeds)
        threshold = float(selected.threshold_exact)
        results.append(metrics(
            test.is_eog_flare.to_numpy(dtype="int8"), score, threshold,
            name=f"05e holdout={country}", extra={
                "country": country, "model_variant": winner["name"],
                "threshold_exact": threshold, "n_features": len(columns),
                "ensemble_seeds": json.dumps(seeds),
                "training_rows_per_seed": json.dumps(used),
                "inner_macro_pr_auc": float(selected.macro_country_pr_auc),
                "selection_policy": "training-country macro AP then macro F1",
            }
        ))
        site_rows.append(site_metrics(test, score, threshold, winner["name"], totals))
        part = test[["source_id", "country", "block_id", "is_eog_flare", "eog_flare_id"]].copy()
        part["score"] = score
        part["threshold"] = threshold
        part["predicted_positive"] = score >= threshold
        part["model_variant"] = winner["name"]
        predictions.append(part)
        print(f"HOLDOUT {country}: {winner['name']}, F1={results[-1]['f1']:.4f}, AP={results[-1]['pr_auc']:.4f}", flush=True)
        if checkpoint:
            checkpoint(pd.DataFrame(results), pd.concat(selections), pd.concat(site_rows), pd.concat(predictions))
        del train, test, score
        gc.collect()
    return pd.DataFrame(results), pd.concat(selections), pd.concat(site_rows), pd.concat(predictions)


def run(input_dir="/kaggle/input", root="/kaggle/working/nb8_domain_revamp", inner_splits=3, rounds=400, seed=71):
    root = Path(root)
    output = root / "outputs"
    cache = root / "cache"
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "05e_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh output directory for a full run")
    input_dir = Path(input_dir)
    frames, hashes = [], {}
    for country in TRAIN_COUNTRIES:
        name = f"features_{country}_{FEATURE_TAG}.parquet"
        matches = list(input_dir.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Need exactly one {name}; found {matches}")
        with matches[0].open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        frames.append(pd.read_parquet(matches[0]))
    df = pd.concat(frames, ignore_index=True)
    raw = robust_feature_cols(df)
    validate_features(df, raw)
    df, mapping = site_safe_blocks(df)
    df, rank_bases = add_country_ranks(df, raw)
    schemas = feature_schemas(df, raw)
    mapping.to_csv(output / "05e_spatial_group_map.csv", index=False)
    manifest = {
        "protocol": PROTOCOL_VERSION, "status": "running", "holdout_country": HOLDOUT,
        "holdout_loaded": False, "countries": TRAIN_COUNTRIES, "years": list(WINDOW_YEARS),
        "n_sources": len(df), "n_positive": int(df.is_eog_flare.sum()),
        "rank_features": rank_bases, "schemas": schemas, "variants": VARIANTS,
        "params": PARAMS, "inner_splits": inner_splits, "rounds": rounds, "seed": seed,
        "outer_ensemble_seeds": 3, "selection": "nested macro-country AP then macro-country F1",
        "rank_policy": "label-free within-country empirical percentile, batch-country inference",
        "input_sha256": hashes, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow"]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    def save(loco, selections, sites, preds):
        loco.to_csv(output / "05e_nested_loco.csv", index=False)
        selections.to_csv(output / "05e_inner_selection.csv", index=False)
        sites.to_csv(output / "05e_loco_site_metrics.csv", index=False)
        preds.to_parquet(cache / "05e_loco_predictions.parquet", index=False)
    loco, selections, sites, preds = nested_loco(df, schemas, inner_splits, rounds, seed, save)
    development, winner, oof = evaluate_variants(df, schemas, 5, rounds, seed)
    development.to_csv(output / "05e_development_variants.csv", index=False)
    oof_frame = df[["source_id", "country", "block_id", "is_eog_flare", "eog_flare_id"]].copy()
    for name, score in oof.items():
        oof_frame[f"score_{name}"] = score
    oof_frame.to_parquet(cache / "05e_oof_predictions.parquet", index=False)
    columns = schemas[winner["schema"]]
    gains = np.zeros(len(columns))
    final_seeds = [seed + 30_000 + offset for offset in (0, 101, 202)]
    for model_index, final_seed in enumerate(final_seeds):
        model, _ = _model(df, columns, np.arange(len(df)), rounds, final_seed, winner["persistent_train"])
        model.save_model(str(cache / f"05e_final_model_{model_index}.txt"))
        gains += model.feature_importance("gain") / len(final_seeds)
    pd.DataFrame({"feature": columns, "mean_gain": gains}).sort_values("mean_gain", ascending=False).to_csv(output / "05e_feature_importance.csv", index=False)
    manifest.update({
        "status": "complete", "selected_variant": winner,
        "selected_features": columns, "final_seeds": final_seeds,
        "final_threshold": float(development.iloc[0].threshold_exact),
        "macro_loco_f1": float(loco.f1.mean()), "macro_loco_pr_auc": float(loco.pr_auc.mean()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return development, loco


if __name__ == "__main__":
    run()
