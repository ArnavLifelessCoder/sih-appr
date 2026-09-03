"""Final foreign-country imagery and FIRMS fusion experiment.

All branches use the same QA-approved sources. India is forbidden. The sampled
imagery cohort is deliberately enriched for EOG matches, so its metrics compare
branches but do not estimate population precision.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


PROTOCOL_VERSION = "08-fusion-country-loco-v1"
COUNTRIES = ["Algeria", "Angola", "Indonesia", "Iraq", "Libya", "Nigeria"]
HOLDOUT = "India"
FEATURE_TAG = "2022_2024"

THERMAL_RAW = [
    "active_days_per_year", "active_months_per_year", "det_per_day",
    "det_per_year", "duty_cycle", "span_window_frac", "mean_gap_days",
    "max_gap_days", "modis_per_year", "snpp_per_year", "n_sensors",
    "snpp_modis_ratio", "night_frac", "sat_frac", "lst_mean", "lst_std",
    "frp_cv",
]
THERMAL_RANK = [
    "frp_mean", "frp_max", "frp_med", "frp_p90", "frp_std",
    "frp_dens_mean", "frp_dens_max", "frp_dens_med", "frp_dens_std",
    "t_mir_mean", "t_mir_max", "t_mir_med", "t_mir_std",
    "t_lwir_mean", "t_lwir_max", "t_lwir_med", "t_lwir_std",
    "dt_mir_lwir_mean", "dt_mir_lwir_max", "dt_mir_lwir_med",
    "dt_mir_lwir_std", "frp_sum_per_year",
]
THERMAL_COLS = THERMAL_RAW + [f"{column}_country_pct" for column in THERMAL_RANK]

IMAGE_COLS = (
    [f"img_center_{band}_median" for band in [
        "blue", "green", "red", "nir", "swir16", "swir22"
    ]]
    + [f"img_center_{index}_{stat}" for index in ["ndvi", "ndbi", "mndwi"]
       for stat in ["p10", "median", "p90"]]
    + [f"img_full_{index}_median" for index in ["ndvi", "ndbi", "mndwi"]]
    + [f"img_center_wc_{code}_fraction" for code in [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100
    ]]
    + ["img_nearest_builtup_in_chip_m"]
)
BRANCHES = {
    "thermal_only": THERMAL_COLS,
    "image_only": IMAGE_COLS,
    "early_fusion": THERMAL_COLS + IMAGE_COLS,
}
PARAMS = {
    "objective": "binary", "learning_rate": 0.03, "num_leaves": 7,
    "max_depth": 4, "min_data_in_leaf": 10, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 2.0,
    "lambda_l2": 10.0, "max_bin": 63, "verbose": -1,
    "num_threads": -1, "force_col_wise": True,
}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Need exactly one {name}; found {matches}")
    return matches[0]


def review_mask(series: pd.Series) -> pd.Series:
    """Normalize CSV booleans without treating the string "False" as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def load_cohort(input_root: str | Path) -> tuple[pd.DataFrame, dict]:
    input_root = Path(input_root)
    paths = {name: find_unique(input_root, name) for name in [
        "pilot_sources.csv", "image_features.parquet", "image_quality.csv",
        "run_state.json", "feature_manifest.json",
    ]}
    state = json.loads(paths["run_state.json"].read_text(encoding="utf-8"))
    if state.get("holdout_loaded") is not False or state.get("protocol") != "nb6-context-v1":
        raise ValueError("NB6 state is incompatible or loaded the holdout")
    sample = pd.read_csv(paths["pilot_sources.csv"])
    images = pd.read_parquet(paths["image_features.parquet"])
    quality = pd.read_csv(paths["image_quality.csv"])
    if len(sample) != 600 or HOLDOUT in set(sample.country):
        raise ValueError("Expected the frozen 600-source foreign NB6 sample")
    for frame, name in [(sample, "sample"), (images, "images"), (quality, "quality")]:
        if not frame.source_id.is_unique:
            raise ValueError(f"Duplicate source IDs in {name}")
    missing_image = set(IMAGE_COLS) - set(images.columns)
    if missing_image:
        raise ValueError(f"Missing image features: {sorted(missing_image)}")
    cohort = sample.merge(images, on="source_id", how="left", validate="one_to_one")
    cohort = cohort.merge(
        quality[["source_id", "status", "review_reflectance_tail"]],
        on="source_id", how="left", validate="one_to_one",
    )
    review = review_mask(cohort.review_reflectance_tail)
    cohort = cohort.loc[cohort.status.eq("ok") & ~review].copy()
    counts = cohort.groupby("country").agg(
        sources=("source_id", "size"), positives=("is_eog_flare", "sum")
    )
    if set(counts.index) != set(COUNTRIES) or counts.sources.min() < 35 or counts.positives.min() < 5:
        raise ValueError(f"Insufficient country or label coverage:\n{counts}")

    thermal_parts = []
    thermal_hashes = {}
    wanted = ["source_id"] + THERMAL_RAW + THERMAL_RANK
    for country in COUNTRIES:
        path = find_unique(input_root, f"features_{country}_{FEATURE_TAG}.parquet")
        thermal_hashes[path.name] = file_hash(path)
        frame = pd.read_parquet(path, columns=wanted)
        if not frame.source_id.is_unique:
            raise ValueError(f"Duplicate thermal sources for {country}")
        for column in THERMAL_RANK:
            frame[f"{column}_country_pct"] = frame[column].rank(
                method="average", pct=True
            ).astype("float32")
        wanted_ids = set(cohort.loc[cohort.country.eq(country), "source_id"])
        thermal_parts.append(frame.loc[frame.source_id.isin(wanted_ids), ["source_id"] + THERMAL_COLS])
    thermal = pd.concat(thermal_parts, ignore_index=True)
    cohort = cohort.merge(thermal, on="source_id", how="left", validate="one_to_one")
    if cohort[THERMAL_COLS].isna().all(axis=1).any():
        raise ValueError("At least one image source has no matching thermal features")
    for columns in BRANCHES.values():
        values = cohort[columns].to_numpy(dtype="float32")
        if np.isinf(values).any():
            raise ValueError("Infinite model feature found")
    metadata = {
        "nb6_counts": state["counts"],
        "qa_sources": len(cohort),
        "qa_country_label_counts": counts.reset_index().to_dict("records"),
        "input_sha256": {
            **{name: file_hash(path) for name, path in paths.items()},
            **thermal_hashes,
        },
    }
    return cohort.reset_index(drop=True), metadata


def train_model(frame, columns, seed, rounds):
    params = dict(PARAMS)
    params.update({key: seed for key in [
        "seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"
    ]})
    return lgb.train(
        params,
        lgb.Dataset(
            frame[columns].to_numpy(dtype="float32"),
            label=frame.is_eog_flare.to_numpy(dtype="int8"),
            free_raw_data=True,
        ),
        num_boost_round=rounds,
    )


def fit_predict(train, test, columns, seeds, rounds):
    score = np.zeros(len(test), dtype="float64")
    for seed in seeds:
        model = train_model(train, columns, seed, rounds)
        score += model.predict(test[columns].to_numpy(dtype="float32")) / len(seeds)
    return score


def inner_country_oof(train, columns, seed, rounds):
    score = np.empty(len(train), dtype="float64")
    for index, country in enumerate(sorted(train.country.unique())):
        fit = train.loc[train.country.ne(country)]
        test_index = train.index[train.country.eq(country)]
        model = train_model(fit, columns, seed + index * 100, rounds)
        score[test_index] = model.predict(
            train.loc[test_index, columns].to_numpy(dtype="float32")
        )
    return score


def f1_arrays(y, score, thresholds):
    order = np.argsort(score)[::-1]
    y_sorted = y[order]
    score_sorted = score[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total = y.sum()
    positions = np.searchsorted(-score_sorted, -thresholds, side="right") - 1
    valid = positions >= 0
    out = np.zeros(len(thresholds))
    t = np.zeros(len(thresholds)); f = np.zeros(len(thresholds))
    t[valid] = tp[positions[valid]]; f[valid] = fp[positions[valid]]
    out = 2 * t / np.maximum(2 * t + f + total - t, 1)
    return out


def macro_f1_threshold(frame, score):
    thresholds = np.unique(score)
    values = np.zeros(len(thresholds))
    for country in sorted(frame.country.unique()):
        mask = frame.country.eq(country).to_numpy()
        values += f1_arrays(
            frame.loc[mask, "is_eog_flare"].to_numpy(dtype="int8"),
            score[mask], thresholds,
        ) / frame.country.nunique()
    best = int(np.argmax(values))
    return float(thresholds[best]), float(values[best])


def macro_ap(frame, score):
    return float(np.mean([
        average_precision_score(part.is_eog_flare, score[part.index])
        for _, part in frame.groupby("country")
    ]))


def metric_row(frame, score, threshold, branch, held_out):
    y = frame.is_eog_flare.to_numpy(dtype="int8")
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "country": held_out, "branch": branch, "n": len(frame),
        "n_positive": int(y.sum()), "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run(input_root="/kaggle/input", output_root="/kaggle/working/nb9_final_fusion", rounds=300, seed=131):
    started = time.time()
    output_root = Path(output_root); out = output_root / "outputs"; models = output_root / "models"
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "08_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh output directory for a complete fusion run")
    shutil.copy2(Path(__file__), output_root / "kg_08_fusion.py")
    cohort, metadata = load_cohort(input_root)
    rows, prediction_parts, selection_rows = [], [], []
    for outer_index, held_out in enumerate(COUNTRIES):
        train = cohort.loc[cohort.country.ne(held_out)].reset_index(drop=True)
        test = cohort.loc[cohort.country.eq(held_out)].reset_index(drop=True)
        inner_scores = {}
        outer_scores = {}
        for branch, columns in BRANCHES.items():
            inner_scores[branch] = inner_country_oof(
                train, columns, seed + outer_index * 1000, rounds
            )
            threshold, inner_f1 = macro_f1_threshold(train, inner_scores[branch])
            seeds = [seed + 20_000 + outer_index * 1000 + offset for offset in (0, 101, 202)]
            outer_scores[branch] = fit_predict(train, test, columns, seeds, rounds)
            rows.append(metric_row(test, outer_scores[branch], threshold, branch, held_out))
            selection_rows.append({
                "held_out_country": held_out, "branch": branch,
                "inner_macro_ap": macro_ap(train, inner_scores[branch]),
                "inner_macro_f1": inner_f1, "threshold": threshold,
            })
        weights = np.linspace(0, 1, 9)
        weight_ap = [macro_ap(
            train, weight * inner_scores["thermal_only"] + (1 - weight) * inner_scores["image_only"]
        ) for weight in weights]
        weight = float(weights[int(np.argmax(weight_ap))])
        inner_late = weight * inner_scores["thermal_only"] + (1 - weight) * inner_scores["image_only"]
        threshold, inner_f1 = macro_f1_threshold(train, inner_late)
        outer_late = weight * outer_scores["thermal_only"] + (1 - weight) * outer_scores["image_only"]
        rows.append(metric_row(test, outer_late, threshold, "late_fusion", held_out))
        selection_rows.append({
            "held_out_country": held_out, "branch": "late_fusion",
            "inner_macro_ap": max(weight_ap), "inner_macro_f1": inner_f1,
            "threshold": threshold, "thermal_weight": weight,
        })
        part = test[["source_id", "country", "is_eog_flare", "eog_flare_id", "block_id"]].copy()
        for branch, score_values in {**outer_scores, "late_fusion": outer_late}.items():
            part[f"score_{branch}"] = score_values
        prediction_parts.append(part)
        print(f"Completed holdout {held_out}", flush=True)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    summary = metrics.groupby("branch").agg(
        macro_f1=("f1", "mean"), macro_pr_auc=("pr_auc", "mean"),
        macro_roc_auc=("roc_auc", "mean"), worst_country_pr_auc=("pr_auc", "min"),
    ).reset_index().sort_values(["macro_pr_auc", "macro_f1"], ascending=False)
    selected = str(summary.iloc[0].branch)
    final_artifacts = []
    if selected == "late_fusion":
        ordered = predictions.set_index("source_id").loc[cohort.source_id]
        weights = np.linspace(0, 1, 9)
        weight_ap = [macro_ap(
            cohort,
            weight * ordered.score_thermal_only.to_numpy()
            + (1 - weight) * ordered.score_image_only.to_numpy(),
        ) for weight in weights]
        final_weight = float(weights[int(np.argmax(weight_ap))])
        final_score = (
            final_weight * ordered.score_thermal_only.to_numpy()
            + (1 - final_weight) * ordered.score_image_only.to_numpy()
        )
        final_branches = ["thermal_only", "image_only"]
    else:
        final_weight = None
        final_score = predictions.set_index("source_id").loc[
            cohort.source_id, f"score_{selected}"
        ].to_numpy()
        final_branches = [selected]
    final_threshold, _ = macro_f1_threshold(cohort, final_score)
    final_seeds = [seed + 30_000 + offset for offset in (0, 101, 202)]
    importance_parts = []
    for branch in final_branches:
        gain = np.zeros(len(BRANCHES[branch]), dtype="float64")
        for index, final_seed in enumerate(final_seeds):
            model = train_model(cohort, BRANCHES[branch], final_seed, rounds)
            path = models / f"{branch}_{index}.txt"; model.save_model(str(path)); final_artifacts.append(path.name)
            gain += model.feature_importance("gain") / len(final_seeds)
        importance_parts.append(pd.DataFrame({
            "branch": branch, "feature": BRANCHES[branch], "mean_gain": gain,
        }))
    metrics.to_csv(out / "08_country_metrics.csv", index=False)
    summary.to_csv(out / "08_branch_summary.csv", index=False)
    selections.to_csv(out / "08_inner_selection.csv", index=False)
    pd.concat(importance_parts, ignore_index=True).sort_values(
        ["branch", "mean_gain"], ascending=[True, False]
    ).to_csv(out / "08_final_feature_importance.csv", index=False)
    predictions.to_parquet(out / "08_loco_predictions.parquet", index=False)
    cohort[["source_id", "country", "is_eog_flare", "eog_flare_id", "block_id"]].to_csv(out / "08_qa_cohort.csv", index=False)
    manifest = {
        "protocol": PROTOCOL_VERSION, "status": "complete", "holdout_country": HOLDOUT,
        "holdout_loaded": False, "interpretation": "enriched foreign imagery pilot; branch comparison, not population precision",
        "branches": {key: value for key, value in BRANCHES.items()}, "params": PARAMS,
        "rounds": rounds, "seed": seed, "outer_ensemble_seeds": 3,
        "selected_branch": selected, "final_threshold": final_threshold,
        "final_late_thermal_weight": final_weight, "model_artifacts": final_artifacts,
        "selection_rule": "highest foreign macro held-out-country AP, then macro F1",
        "metadata": metadata, "elapsed_minutes": (time.time() - started) / 60,
        "python": platform.python_version(), "versions": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow"]
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, metrics, predictions


if __name__ == "__main__":
    run()
