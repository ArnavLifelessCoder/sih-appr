"""Three-seed stability confirmation for guarded Sentinel CV fusion.

NB12 seed 271 is discovery evidence and is not reused as confirmation. This
stage freezes three fresh seeds, ensembles their outer-country predictions,
and compares one eligible challenger against the protected compact baseline.
The all-77 branch remains diagnostic and cannot affect the decision. India is
never loaded.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from kg_08_fusion import (
    COUNTRIES,
    HOLDOUT,
    file_hash,
    macro_f1_threshold,
    metric_row,
)
from kg_12_cv_tabular import (
    ALPHAS,
    CHECKPOINT_SHA256,
    PROTOCOL as CACHE_PROTOCOL,
    apply_deployment_policy,
    run as run_single_seed,
)


PROTOCOL = "12b-guarded-confirmatory-ensemble-v1"
DISCOVERY_SEED = 271
CONFIRMATORY_SEEDS = (272, 273, 274)
BOOTSTRAP_SEED = 12012
ELIGIBLE_CHALLENGER = "guarded_cv_tabular"
BASELINE = "compact_tabular"
DIAGNOSTIC_BRANCH = "all77_tabular"
EXPECTED_SOURCES = 294
EXPECTED_POSITIVES = 87
EXPECTED_CACHE = {
    "cohort_sha256": (
        "fd0a4208bcdbd3ec374e877c63f078479b54fa6b100fe0d842e9cb5b14aa9526"
    ),
    "chip_inventory_sha256": (
        "6ebba3c0bb9fd2463f733426e20c0454c650d1249a7cb77a4d0ba58107ac5d60"
    ),
    "embedding_sha256": (
        "44b78d9137b0cebc3f07d6aba2322a0139d402f04c1329fbac84ecc43a74f00b"
    ),
    "morphology_sha256": (
        "b873bf53eaf9161f9d4f775bc84af9670a7d7d52124988a921347894e0127719"
    ),
}
REQUIRED_CACHE_FILES = (
    "12_cv_embeddings.parquet",
    "12_cv_embeddings.json",
    "12_morphology.parquet",
    "12_morphology.json",
)
REQUIRED_SCORE_COLUMNS = (
    "score_compact_tabular",
    "score_guarded_cv_tabular",
    "score_cv_multispectral",
    "score_all77_tabular",
)
ACCEPTANCE_THRESHOLDS = {
    "minimum_macro_pr_auc_gain": 0.010,
    "minimum_improved_countries": 4,
    "minimum_worst_country_pr_auc_delta": -0.020,
    "minimum_indonesia_pr_auc_delta": -0.020,
    "minimum_macro_f1_delta": -0.010,
    "minimum_bootstrap_lower_95": -0.010,
    "minimum_macro_recall_at_20_delta": -0.020,
    "minimum_seed_macro_pr_auc_gain": 0.0,
    "minimum_median_seed_macro_pr_auc_gain": 0.010,
    "maximum_positive_loss_per_country_at_20": 1,
    "minimum_nonzero_full_foreign_seed_alphas": 2,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_verified_cache(input_root: Path) -> Path:
    """Find the exact deterministic cache produced by the audited NB12 run."""
    candidates = []
    for embedding in input_root.rglob("12_cv_embeddings.parquet"):
        parent = embedding.parent
        if all((parent / name).is_file() for name in REQUIRED_CACHE_FILES):
            candidates.append(parent)
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Attach exactly one saved NB12 output with its unpacked cache files. "
            f"Found {len(candidates)} candidates: {candidates}"
        )
    cache = candidates[0]
    embedding_manifest = _read_json(cache / "12_cv_embeddings.json")
    morphology_manifest = _read_json(cache / "12_morphology.json")
    expected_embedding = {
        "protocol": CACHE_PROTOCOL,
        "cohort_sha256": EXPECTED_CACHE["cohort_sha256"],
        "chip_inventory_sha256": EXPECTED_CACHE["chip_inventory_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "embedding_sha256": EXPECTED_CACHE["embedding_sha256"],
        "n_sources": EXPECTED_SOURCES,
        "n_features": 1024,
        "tta": True,
    }
    for key, expected in expected_embedding.items():
        if embedding_manifest.get(key) != expected:
            raise ValueError(
                f"NB12 embedding cache mismatch for {key}: "
                f"{embedding_manifest.get(key)!r} != {expected!r}"
            )
    expected_morphology = {
        "protocol": CACHE_PROTOCOL,
        "cohort_sha256": EXPECTED_CACHE["cohort_sha256"],
        "chip_inventory_sha256": EXPECTED_CACHE["chip_inventory_sha256"],
        "sha256": EXPECTED_CACHE["morphology_sha256"],
        "n_sources": EXPECTED_SOURCES,
    }
    for key, expected in expected_morphology.items():
        if morphology_manifest.get(key) != expected:
            raise ValueError(
                f"NB12 morphology cache mismatch for {key}: "
                f"{morphology_manifest.get(key)!r} != {expected!r}"
            )
    if file_hash(cache / "12_cv_embeddings.parquet") != EXPECTED_CACHE["embedding_sha256"]:
        raise ValueError("NB12 embedding parquet SHA-256 mismatch")
    if file_hash(cache / "12_morphology.parquet") != EXPECTED_CACHE["morphology_sha256"]:
        raise ValueError("NB12 morphology parquet SHA-256 mismatch")
    return cache


def _copy_cache(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_CACHE_FILES:
        source_path = source / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing reusable cache file: {source_path}")
        shutil.copy2(source_path, destination / name)


def _stable_cohort_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_id", "country", "is_eog_flare", "eog_flare_id", "block_id"
    ]
    canonical = frame[columns].copy()
    for column in columns:
        canonical[column] = canonical[column].fillna("<NA>").astype(str)
    payload = canonical.sort_values("source_id").to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_order(part: pd.DataFrame, score_column: str) -> pd.DataFrame:
    ordered = part.copy()
    ordered["_source_tiebreak"] = ordered.source_id.astype(str)
    return ordered.sort_values(
        [score_column, "_source_tiebreak"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns="_source_tiebreak")


def paired_country_block_bootstrap(
    predictions: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap 10 km blocks within country while preserving positive blocks."""
    required = {
        "country", "block_id", "is_eog_flare", baseline_column, candidate_column
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Block bootstrap is missing columns: {sorted(missing)}")
    if predictions.block_id.isna().any() or predictions.duplicated("source_id").any():
        raise ValueError("Block bootstrap needs unique sources and non-null block IDs")
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype="float64")
    country_groups = []
    for country, part in predictions.groupby("country", sort=True):
        part = part.reset_index(drop=True)
        blocks = {
            block_id: index.to_numpy(dtype="int64")
            for block_id, index in part.groupby("block_id").groups.items()
        }
        positive_blocks = [
            block_id
            for block_id, index in blocks.items()
            if bool(part.loc[index, "is_eog_flare"].any())
        ]
        positive_block_set = set(positive_blocks)
        background_blocks = [
            block_id for block_id in blocks if block_id not in positive_block_set
        ]
        if not positive_blocks or not background_blocks:
            raise ValueError(f"Cannot stratify spatial blocks in {country}")
        country_groups.append((country, part, blocks, positive_blocks, background_blocks))
    for repeat in range(repeats):
        country_differences = []
        for _, part, blocks, positive_blocks, background_blocks in country_groups:
            sampled_block_ids = np.concatenate([
                rng.choice(positive_blocks, size=len(positive_blocks), replace=True),
                rng.choice(background_blocks, size=len(background_blocks), replace=True),
            ])
            sampled_index = np.concatenate([blocks[block_id] for block_id in sampled_block_ids])
            sampled = part.iloc[sampled_index]
            y = sampled.is_eog_flare.to_numpy(dtype="int8")
            baseline_ap = average_precision_score(
                y, sampled[baseline_column].to_numpy()
            )
            candidate_ap = average_precision_score(
                y, sampled[candidate_column].to_numpy()
            )
            country_differences.append(candidate_ap - baseline_ap)
        differences[repeat] = float(np.mean(country_differences))
    return {
        "mean": float(differences.mean()),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "repeats": repeats,
        "method": "positive-stratified 10 km block bootstrap within country",
    }


def review_budget_tables(
    predictions: pd.DataFrame,
    branches: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fixed review budgets with deterministic source-ID tie breaks."""
    metric_rows = []
    membership_rows = []
    for country, part in predictions.groupby("country", sort=True):
        positives = int(part.is_eog_flare.sum())
        if positives < 1:
            raise ValueError(f"No positives available for review budget in {country}")
        for branch in branches:
            score_column = f"score_{branch}"
            ordered = _deterministic_order(part, score_column).reset_index(drop=True)
            for fraction in (0.10, 0.20, 0.30):
                count = max(1, int(math.ceil(fraction * len(ordered))))
                selected = ordered.head(count)
                found = int(selected.is_eog_flare.sum())
                metric_rows.append({
                    "country": country,
                    "branch": branch,
                    "review_fraction": fraction,
                    "review_count": count,
                    "positive_found": found,
                    "precision_at_budget": found / count,
                    "recall_at_budget": found / positives,
                })
                for rank, row in selected.iterrows():
                    membership_rows.append({
                        "country": country,
                        "branch": branch,
                        "review_fraction": fraction,
                        "rank": rank + 1,
                        "source_id": row.source_id,
                        "is_eog_flare": int(row.is_eog_flare),
                        "score": float(row[score_column]),
                    })
    return pd.DataFrame(metric_rows), pd.DataFrame(membership_rows)


def apply_confirmatory_policy(
    frame: pd.DataFrame,
    compact_model_scores: np.ndarray,
    selected_branch: str,
    visual_model_scores: np.ndarray | None = None,
    seed_fusion_alphas: np.ndarray | None = None,
    compact_oof_reference: np.ndarray | None = None,
    visual_oof_reference: np.ndarray | None = None,
    image_available: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse each seed pipeline separately, then average the pipeline scores."""
    compact_matrix = np.asarray(compact_model_scores, dtype="float64")
    if (
        compact_matrix.ndim != 3
        or compact_matrix.shape[0] != len(frame)
        or compact_matrix.shape[1] != len(CONFIRMATORY_SEEDS)
    ):
        raise ValueError(
            "Compact scores must have shape (rows, three seeds, models per seed)"
        )
    if not np.isfinite(compact_matrix).all():
        raise ValueError("Compact model scores contain non-finite values")
    compact_by_seed = compact_matrix.mean(axis=2)
    if selected_branch == BASELINE:
        return compact_by_seed.mean(axis=1)
    if selected_branch != ELIGIBLE_CHALLENGER:
        raise ValueError(f"Unknown confirmatory deployment branch: {selected_branch}")
    if any(value is None for value in (
        visual_model_scores,
        seed_fusion_alphas,
        compact_oof_reference,
        visual_oof_reference,
    )):
        raise ValueError("Guarded deployment requires every seed fusion input")
    visual_matrix = np.asarray(visual_model_scores, dtype="float64")
    alphas = np.asarray(seed_fusion_alphas, dtype="float64")
    compact_reference = np.asarray(compact_oof_reference, dtype="float64")
    visual_reference = np.asarray(visual_oof_reference, dtype="float64")
    expected_visual = (len(frame), len(CONFIRMATORY_SEEDS))
    if visual_matrix.shape != expected_visual:
        raise ValueError("Visual model scores must have shape (rows, three seeds)")
    if alphas.shape != (len(CONFIRMATORY_SEEDS),):
        raise ValueError("Seed fusion alphas must contain exactly three values")
    if (
        compact_reference.ndim != 2
        or compact_reference.shape[1] != len(CONFIRMATORY_SEEDS)
        or visual_reference.shape != compact_reference.shape
    ):
        raise ValueError("OOF references must have shape (reference rows, three seeds)")
    if image_available is None:
        available = np.ones(len(frame), dtype=bool)
    else:
        available = np.asarray(image_available, dtype=bool)
        if available.shape != (len(frame),):
            raise ValueError("Image availability mask has the wrong shape")
    if not np.isfinite(visual_matrix[available]).all():
        raise ValueError("Available visual model scores contain non-finite values")
    if not all(np.isfinite(values).all() for values in (
        alphas, compact_reference, visual_reference
    )):
        raise ValueError("Guarded calibration inputs contain non-finite values")
    if any(float(alpha) not in ALPHAS for alpha in alphas):
        raise ValueError(f"Seed fusion alpha is outside the frozen grid {ALPHAS}")
    seed_outputs = []
    for index in range(len(CONFIRMATORY_SEEDS)):
        seed_outputs.append(apply_deployment_policy(
            frame,
            compact_by_seed[:, index],
            ELIGIBLE_CHALLENGER,
            fusion_alpha=float(alphas[index]),
            visual_score=visual_matrix[:, index],
            image_available=image_available,
            tabular_reference=compact_reference[:, index],
            visual_reference=visual_reference[:, index],
        ))
    return np.column_stack(seed_outputs).mean(axis=1)


def _validate_panel(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    run_manifests: dict[int, dict],
) -> dict:
    expected_countries = set(COUNTRIES)
    expected_branches = {
        BASELINE,
        DIAGNOSTIC_BRANCH,
        "handcrafted_image",
        "cnn_only",
        "cv_multispectral",
        ELIGIBLE_CHALLENGER,
        "nested_champion",
    }
    if predictions.duplicated(["seed", "source_id"]).any():
        raise ValueError("Duplicate seed and source ID in confirmatory predictions")
    if set(predictions.seed.unique()) != set(CONFIRMATORY_SEEDS):
        raise ValueError("Confirmatory prediction seeds do not match the frozen panel")
    reference_hash = None
    for seed in CONFIRMATORY_SEEDS:
        part = predictions.loc[predictions.seed.eq(seed)].copy()
        if len(part) != EXPECTED_SOURCES or part.source_id.nunique() != EXPECTED_SOURCES:
            raise ValueError(f"Seed {seed} does not contain 294 unique sources")
        if set(part.country) != expected_countries or HOLDOUT in set(part.country):
            raise ValueError(f"Seed {seed} has the wrong country cohort")
        if int(part.is_eog_flare.sum()) != EXPECTED_POSITIVES:
            raise ValueError(f"Seed {seed} has the wrong positive count")
        values = part[list(REQUIRED_SCORE_COLUMNS)].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Seed {seed} contains invalid predictions")
        cohort_hash = _stable_cohort_hash(part)
        if reference_hash is None:
            reference_hash = cohort_hash
        elif cohort_hash != reference_hash:
            raise ValueError("Source metadata or labels differ between seeds")
        seed_metrics = metrics.loc[metrics.seed.eq(seed)]
        if set(seed_metrics.country) != expected_countries:
            raise ValueError(f"Seed {seed} metrics have the wrong countries")
        if not expected_branches.issubset(set(seed_metrics.branch)):
            raise ValueError(f"Seed {seed} metrics are missing expected branches")
        expected_pairs = len(expected_countries) * len(expected_branches)
        if len(seed_metrics.loc[seed_metrics.branch.isin(expected_branches)]) != expected_pairs:
            raise ValueError(f"Seed {seed} metrics have duplicate branch-country rows")
        seed_selections = selections.loc[selections.seed.eq(seed)]
        if len(seed_selections) != expected_pairs:
            raise ValueError(f"Seed {seed} inner selections are incomplete")
        manifest = run_manifests[seed]
        checkpoint = manifest.get("checkpoint", {})
        if (
            manifest.get("holdout_loaded") is not False
            or checkpoint.get("cohort_sha256") != EXPECTED_CACHE["cohort_sha256"]
            or checkpoint.get("chip_inventory_sha256")
            != EXPECTED_CACHE["chip_inventory_sha256"]
            or checkpoint.get("embedding_sha256")
            != EXPECTED_CACHE["embedding_sha256"]
        ):
            raise ValueError(f"Seed {seed} run manifest failed cache integrity checks")
    return {
        "passed": True,
        "sources_per_seed": EXPECTED_SOURCES,
        "positives_per_seed": EXPECTED_POSITIVES,
        "countries": list(COUNTRIES),
        "holdout_loaded": False,
        "stable_panel_cohort_sha256": reference_hash,
        "validated_seed_source_rows": int(len(predictions)),
    }


def _ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions.seed.eq(CONFIRMATORY_SEEDS[0])].copy()
    primary = primary.drop(columns="seed").reset_index(drop=True)
    score_columns = sorted(
        column for column in predictions if column.startswith("score_")
    )
    for score_column in score_columns:
        pivot = predictions.pivot(index="source_id", columns="seed", values=score_column)
        if list(pivot.columns) != list(CONFIRMATORY_SEEDS) or pivot.isna().any().any():
            raise ValueError(f"Incomplete seed panel for {score_column}")
        primary[score_column] = primary.source_id.map(pivot.mean(axis=1))
    if not np.isfinite(primary[score_columns].to_numpy(dtype="float64")).all():
        raise ValueError("Non-finite ensemble prediction")
    return primary


def _ensemble_metrics(
    ensemble: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    threshold_rows = []
    branches = (BASELINE, ELIGIBLE_CHALLENGER, DIAGNOSTIC_BRANCH)
    for country in COUNTRIES:
        part = ensemble.loc[ensemble.country.eq(country)].reset_index(drop=True)
        for branch in branches:
            chosen = selections.loc[
                selections.held_out_country.eq(country)
                & selections.branch.eq(branch),
                ["seed", "threshold"],
            ].sort_values("seed")
            if list(chosen.seed) != list(CONFIRMATORY_SEEDS):
                raise ValueError(f"Missing thresholds for {country}, {branch}")
            threshold = float(chosen.threshold.median())
            metric_rows.append(metric_row(
                part,
                part[f"score_{branch}"].to_numpy(dtype="float64"),
                threshold,
                branch,
                country,
            ))
            threshold_rows.append({
                "country": country,
                "branch": branch,
                "threshold_policy": "median of three fold-local seed thresholds",
                "threshold": threshold,
                "seed_272_threshold": float(chosen.iloc[0].threshold),
                "seed_273_threshold": float(chosen.iloc[1].threshold),
                "seed_274_threshold": float(chosen.iloc[2].threshold),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows)


def _seed_summary(metrics: pd.DataFrame, run_manifests: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for seed in CONFIRMATORY_SEEDS:
        selected = metrics.loc[metrics.seed.eq(seed)]
        summary = selected.groupby("branch").agg(
            macro_pr_auc=("pr_auc", "mean"),
            macro_f1=("f1", "mean"),
            worst_country_pr_auc=("pr_auc", "min"),
        )
        baseline = summary.loc[BASELINE]
        candidate = summary.loc[ELIGIBLE_CHALLENGER]
        country_metrics = selected.pivot(
            index="country", columns="branch", values="pr_auc"
        )
        deltas = country_metrics[ELIGIBLE_CHALLENGER] - country_metrics[BASELINE]
        rows.append({
            "seed": seed,
            "compact_macro_pr_auc": baseline.macro_pr_auc,
            "guarded_macro_pr_auc": candidate.macro_pr_auc,
            "macro_pr_auc_gain": candidate.macro_pr_auc - baseline.macro_pr_auc,
            "compact_macro_f1": baseline.macro_f1,
            "guarded_macro_f1": candidate.macro_f1,
            "macro_f1_gain": candidate.macro_f1 - baseline.macro_f1,
            "improved_countries": int((deltas > 0).sum()),
            "worst_country_pr_auc_delta": float(deltas.min()),
            "original_nb12_guard_passed": bool(
                run_manifests[seed]["selection_results"][ELIGIBLE_CHALLENGER]["passed"]
            ),
            "diagnostic_nested_selection": run_manifests[seed]["selected_branch"],
            "elapsed_minutes": run_manifests[seed]["elapsed_minutes"],
        })
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _country_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics.branch.isin([BASELINE, ELIGIBLE_CHALLENGER]),
        ["seed", "country", "branch", "pr_auc"],
    ]
    comparison = selected.pivot(
        index=["seed", "country"], columns="branch", values="pr_auc"
    )
    comparison["pr_auc_delta"] = comparison[ELIGIBLE_CHALLENGER] - comparison[BASELINE]
    return comparison.reset_index().groupby("country").agg(
        mean_pr_auc_delta=("pr_auc_delta", "mean"),
        std_pr_auc_delta=("pr_auc_delta", "std"),
        min_pr_auc_delta=("pr_auc_delta", "min"),
        max_pr_auc_delta=("pr_auc_delta", "max"),
        positive_seeds=("pr_auc_delta", lambda values: int((values > 0).sum())),
    ).reset_index()


def _condition(name: str, observed, rule: str, passed: bool) -> dict:
    if isinstance(observed, np.generic):
        observed = observed.item()
    return {"name": name, "observed": observed, "rule": rule, "passed": bool(passed)}


def _acceptance_decision(
    ensemble: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    budget_metrics: pd.DataFrame,
    seed_summary: pd.DataFrame,
    deployment_alphas: dict[int, float],
    integrity: dict,
    bootstrap_repeats: int,
) -> dict:
    metric_table = ensemble_metrics.pivot(index="country", columns="branch", values="pr_auc")
    deltas = metric_table[ELIGIBLE_CHALLENGER] - metric_table[BASELINE]
    macro_gain = float(deltas.mean())
    f1_table = ensemble_metrics.groupby("branch").f1.mean()
    macro_f1_delta = float(f1_table[ELIGIBLE_CHALLENGER] - f1_table[BASELINE])
    bootstrap = paired_country_block_bootstrap(
        ensemble,
        "score_compact_tabular",
        "score_guarded_cv_tabular",
        repeats=bootstrap_repeats,
        seed=BOOTSTRAP_SEED,
    )
    review = budget_metrics.loc[
        budget_metrics.review_fraction.eq(0.20)
        & budget_metrics.branch.isin([BASELINE, ELIGIBLE_CHALLENGER])
    ]
    recall = review.pivot(index="country", columns="branch", values="recall_at_budget")
    found = review.pivot(index="country", columns="branch", values="positive_found")
    recall_delta = recall[ELIGIBLE_CHALLENGER] - recall[BASELINE]
    found_delta = found[ELIGIBLE_CHALLENGER] - found[BASELINE]
    seed_gain = seed_summary.macro_pr_auc_gain.to_numpy(dtype="float64")
    nonzero_alphas = sum(alpha > 0 for alpha in deployment_alphas.values())
    thresholds = ACCEPTANCE_THRESHOLDS
    conditions = [
        _condition("integrity_valid", integrity["passed"], "must be true", integrity["passed"]),
        _condition(
            "ensemble_macro_pr_auc_gain", macro_gain,
            f">= {thresholds['minimum_macro_pr_auc_gain']}",
            macro_gain >= thresholds["minimum_macro_pr_auc_gain"],
        ),
        _condition(
            "ensemble_improved_countries", int((deltas > 0).sum()),
            f">= {thresholds['minimum_improved_countries']}",
            int((deltas > 0).sum()) >= thresholds["minimum_improved_countries"],
        ),
        _condition(
            "ensemble_worst_country_pr_auc_delta", float(deltas.min()),
            f">= {thresholds['minimum_worst_country_pr_auc_delta']}",
            float(deltas.min()) >= thresholds["minimum_worst_country_pr_auc_delta"],
        ),
        _condition(
            "ensemble_indonesia_pr_auc_delta", float(deltas.loc["Indonesia"]),
            f">= {thresholds['minimum_indonesia_pr_auc_delta']}",
            float(deltas.loc["Indonesia"])
            >= thresholds["minimum_indonesia_pr_auc_delta"],
        ),
        _condition(
            "ensemble_macro_f1_delta", macro_f1_delta,
            f">= {thresholds['minimum_macro_f1_delta']}",
            macro_f1_delta >= thresholds["minimum_macro_f1_delta"],
        ),
        _condition(
            "bootstrap_lower_95", bootstrap["lower_95"],
            f">= {thresholds['minimum_bootstrap_lower_95']}",
            bootstrap["lower_95"] >= thresholds["minimum_bootstrap_lower_95"],
        ),
        _condition(
            "macro_recall_at_20_delta", float(recall_delta.mean()),
            f">= {thresholds['minimum_macro_recall_at_20_delta']}",
            float(recall_delta.mean())
            >= thresholds["minimum_macro_recall_at_20_delta"],
        ),
        _condition(
            "worst_positive_count_loss_at_20", int(found_delta.min()),
            f">= -{thresholds['maximum_positive_loss_per_country_at_20']}",
            int(found_delta.min())
            >= -thresholds["maximum_positive_loss_per_country_at_20"],
        ),
        _condition(
            "all_seed_macro_pr_auc_gains_positive", seed_gain.tolist(),
            "> 0 for every fresh seed",
            bool((seed_gain > thresholds["minimum_seed_macro_pr_auc_gain"]).all()),
        ),
        _condition(
            "median_seed_macro_pr_auc_gain", float(np.median(seed_gain)),
            f">= {thresholds['minimum_median_seed_macro_pr_auc_gain']}",
            float(np.median(seed_gain))
            >= thresholds["minimum_median_seed_macro_pr_auc_gain"],
        ),
        _condition(
            "nonzero_full_foreign_seed_alphas", nonzero_alphas,
            f">= {thresholds['minimum_nonzero_full_foreign_seed_alphas']}",
            nonzero_alphas
            >= thresholds["minimum_nonzero_full_foreign_seed_alphas"],
        ),
    ]
    return {
        "protocol": PROTOCOL,
        "eligible_challenger": ELIGIBLE_CHALLENGER,
        "baseline": BASELINE,
        "diagnostic_only": [DIAGNOSTIC_BRANCH, "nested_champion"],
        "conditions": conditions,
        "bootstrap": {**bootstrap, "seed": BOOTSTRAP_SEED},
        "full_foreign_seed_alphas": {
            str(seed): float(alpha) for seed, alpha in deployment_alphas.items()
        },
        "country_pr_auc_deltas": {str(key): float(value) for key, value in deltas.items()},
        "country_recall_at_20_deltas": {
            str(key): float(value) for key, value in recall_delta.items()
        },
        "country_positive_count_at_20_deltas": {
            str(key): int(value) for key, value in found_delta.items()
        },
        "passed_all_conditions": bool(
            all(condition["passed"] for condition in conditions)
        ),
    }


def _copy_ensemble_models(
    runs_root: Path,
    final_models: Path,
    ensemble: pd.DataFrame,
    seed_predictions: pd.DataFrame,
    run_schemas: dict[int, dict],
) -> dict:
    pipelines = []
    compact_reference_parts = []
    visual_reference_parts = []
    for seed in CONFIRMATORY_SEEDS:
        model_root = runs_root / f"seed_{seed}" / "models"
        compact_files = []
        for index in range(3):
            source = model_root / f"12_compact_fallback_{index}.txt"
            if not source.is_file():
                raise FileNotFoundError(f"Missing compact model: {source}")
            name = f"12b_compact_seed_{seed}_{index}.txt"
            shutil.copy2(source, final_models / name)
            compact_files.append(name)
        source = model_root / "12_visual_pu.joblib"
        if not source.is_file():
            raise FileNotFoundError(f"Missing visual model: {source}")
        name = f"12b_visual_seed_{seed}.joblib"
        shutil.copy2(source, final_models / name)
        aligned = seed_predictions.loc[seed_predictions.seed.eq(seed)].set_index(
            "source_id"
        ).loc[ensemble.source_id]
        compact_reference_parts.append(
            aligned.score_compact_tabular.to_numpy(dtype="float64")
        )
        visual_reference_parts.append(
            aligned.score_cv_multispectral.to_numpy(dtype="float64")
        )
        pipelines.append({
            "seed": seed,
            "compact_models": compact_files,
            "visual_model": name,
            "fusion_alpha": float(
                run_schemas[seed]["diagnostic_oof_best_alpha"]
            ),
        })
    calibration_name = "12b_seed_fusion_calibration.npz"
    np.savez_compressed(
        final_models / calibration_name,
        source_id=ensemble.source_id.astype(str).to_numpy(dtype="U"),
        seeds=np.asarray(CONFIRMATORY_SEEDS, dtype="int64"),
        compact_oof=np.column_stack(compact_reference_parts),
        visual_oof=np.column_stack(visual_reference_parts),
        fusion_alpha=np.asarray(
            [pipeline["fusion_alpha"] for pipeline in pipelines],
            dtype="float64",
        ),
    )
    return {
        "seed_pipelines": pipelines,
        "calibration_file": calibration_name,
        "compact_rule": "mean of all nine LightGBM probabilities",
        "guarded_rule": (
            "within each seed, mean its three compact models, fuse with that "
            "seed's visual model, OOF references, and alpha, then mean the "
            "three seed pipeline scores"
        ),
    }


def run_confirmatory(
    input_root: str | Path = "/kaggle/input",
    output_root: str | Path = "/kaggle/working/nb12b_confirmatory",
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    rounds: int = 300,
    pca_components: int = 12,
    per_seed_bootstrap_repeats: int = 1000,
    final_bootstrap_repeats: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the frozen guarded-CV confirmation panel and export an ensemble."""
    if tuple(seeds) != CONFIRMATORY_SEEDS:
        raise ValueError(
            f"Confirmatory seeds are frozen as {CONFIRMATORY_SEEDS}, got {seeds}"
        )
    if rounds != 300 or pca_components != 12:
        raise ValueError("NB12b keeps NB12 rounds=300 and pca_components=12")
    if final_bootstrap_repeats != 5000:
        raise ValueError("The final bootstrap is frozen at 5000 repeats")
    started = time.time()
    input_root = Path(input_root)
    output_root = Path(output_root)
    outputs = output_root / "outputs"
    cache = output_root / "cache"
    final_models = output_root / "final_models"
    runs_root = output_root / "runs"
    manifest_path = outputs / "12b_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh NB12b output directory")
    for directory in (outputs, cache, final_models, runs_root):
        directory.mkdir(parents=True, exist_ok=True)

    attached_cache = _find_verified_cache(input_root)
    _copy_cache(attached_cache, cache)
    print(f"Verified and copied deterministic NB12 cache: {attached_cache}", flush=True)

    metric_parts = []
    prediction_parts = []
    selection_parts = []
    run_manifests: dict[int, dict] = {}
    run_schemas: dict[int, dict] = {}
    for seed in CONFIRMATORY_SEEDS:
        run_root = runs_root / f"seed_{seed}"
        _copy_cache(cache, run_root / "cache")
        _, metrics, predictions = run_single_seed(
            input_root=input_root,
            output_root=run_root,
            rounds=rounds,
            pca_components=pca_components,
            bootstrap_repeats=per_seed_bootstrap_repeats,
            seed=seed,
        )
        metrics.insert(0, "seed", seed)
        predictions.insert(0, "seed", seed)
        selections = pd.read_csv(run_root / "outputs" / "12_inner_selection.csv")
        selections.insert(0, "seed", seed)
        metric_parts.append(metrics)
        prediction_parts.append(predictions)
        selection_parts.append(selections)
        run_manifests[seed] = _read_json(run_root / "outputs" / "12_manifest.json")
        run_schemas[seed] = _read_json(
            run_root / "outputs" / "12_selected_schema.json"
        )
        print(f"Completed fresh confirmatory seed {seed}", flush=True)

    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    all_selections = pd.concat(selection_parts, ignore_index=True)
    integrity = _validate_panel(all_predictions, all_metrics, all_selections, run_manifests)
    schema_keys = (
        "compact_fallback_columns",
        "image_columns",
        "morphology_columns",
        "embedding_columns",
        "pca_components",
        "fusion_formula",
    )
    reference_schema = run_schemas[CONFIRMATORY_SEEDS[0]]
    for seed in CONFIRMATORY_SEEDS[1:]:
        for key in schema_keys:
            if run_schemas[seed].get(key) != reference_schema.get(key):
                raise ValueError(f"Seed {seed} changed the frozen schema field {key}")
    ensemble = _ensemble_predictions(all_predictions)
    ensemble_metrics, ensemble_thresholds = _ensemble_metrics(ensemble, all_selections)
    seed_summary = _seed_summary(all_metrics, run_manifests)
    country_stability = _country_stability(all_metrics)
    budget, queue_membership = review_budget_tables(
        ensemble, (BASELINE, ELIGIBLE_CHALLENGER, DIAGNOSTIC_BRANCH)
    )
    deployment_alphas = {
        seed: float(run_schemas[seed]["diagnostic_oof_best_alpha"])
        for seed in CONFIRMATORY_SEEDS
    }
    if any(
        not np.isfinite(alpha) or alpha not in ALPHAS
        for alpha in deployment_alphas.values()
    ):
        raise ValueError(f"A full-foreign alpha is outside the frozen grid {ALPHAS}")
    acceptance = _acceptance_decision(
        ensemble,
        ensemble_metrics,
        budget,
        seed_summary,
        deployment_alphas,
        integrity,
        final_bootstrap_repeats,
    )

    ensemble_frame = ensemble.reset_index(drop=True)
    acceptance["deployment_alpha_grid"] = list(ALPHAS)
    acceptance["passed"] = acceptance["passed_all_conditions"]
    selected_branch = ELIGIBLE_CHALLENGER if acceptance["passed"] else BASELINE
    acceptance["final_decision"] = selected_branch
    selected_score = ensemble_frame[f"score_{selected_branch}"].to_numpy(dtype="float64")
    pilot_threshold, _ = macro_f1_threshold(ensemble_frame, selected_score)

    model_schema = _copy_ensemble_models(
        runs_root,
        final_models,
        ensemble_frame,
        all_predictions,
        run_schemas,
    )
    for name in ("kg_08_fusion.py", "kg_12_cv_tabular.py"):
        source = Path(__file__).with_name(name)
        if not source.is_file():
            raise FileNotFoundError(f"Missing source file: {source}")
        shutil.copy2(source, output_root / name)
    shutil.copy2(Path(__file__), output_root / "kg_12b_confirmatory.py")

    ensemble_summary = ensemble_metrics.groupby("branch").agg(
        macro_f1=("f1", "mean"),
        macro_pr_auc=("pr_auc", "mean"),
        macro_roc_auc=("roc_auc", "mean"),
        worst_country_pr_auc=("pr_auc", "min"),
    ).reset_index().sort_values("macro_pr_auc", ascending=False)
    seed_summary.to_csv(outputs / "12b_seed_summary.csv", index=False)
    all_metrics.to_csv(outputs / "12b_seed_country_metrics.csv", index=False)
    all_predictions.to_parquet(outputs / "12b_seed_predictions.parquet", index=False)
    all_selections.to_csv(outputs / "12b_inner_selection.csv", index=False)
    ensemble.to_parquet(outputs / "12b_ensemble_predictions.parquet", index=False)
    ensemble_metrics.to_csv(outputs / "12b_ensemble_country_metrics.csv", index=False)
    ensemble_summary.to_csv(outputs / "12b_ensemble_summary.csv", index=False)
    ensemble_thresholds.to_csv(outputs / "12b_ensemble_thresholds.csv", index=False)
    country_stability.to_csv(outputs / "12b_country_stability.csv", index=False)
    budget.to_csv(outputs / "12b_review_budget_metrics.csv", index=False)
    queue_membership.to_csv(outputs / "12b_review_queue_membership.csv", index=False)
    (outputs / "12b_acceptance.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )

    final_schema = {
        "protocol": PROTOCOL,
        "selected_branch": selected_branch,
        "deployment_branch": selected_branch,
        "eligible_challenger": ELIGIBLE_CHALLENGER,
        "diagnostic_only_branches": [DIAGNOSTIC_BRANCH, "nested_champion"],
        "discovery_seed_excluded": DISCOVERY_SEED,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "model_ensemble": model_schema,
        "tabular_columns": reference_schema["compact_fallback_columns"],
        "compact_fallback_columns": reference_schema["compact_fallback_columns"],
        "image_columns": reference_schema["image_columns"],
        "morphology_columns": reference_schema["morphology_columns"],
        "embedding_columns": reference_schema["embedding_columns"],
        "pca_components": reference_schema["pca_components"],
        "deployment_fusion_alphas_by_seed": (
            {str(seed): alpha for seed, alpha in deployment_alphas.items()}
            if selected_branch == ELIGIBLE_CHALLENGER
            else {str(seed): 0.0 for seed in CONFIRMATORY_SEEDS}
        ),
        "guarded_diagnostic_alphas_by_seed": {
            str(seed): alpha for seed, alpha in deployment_alphas.items()
        },
        "fusion_formula": (
            "calibrated_tabular + alpha * 4 * calibrated_tabular * "
            "(1-calibrated_tabular) * (calibrated_visual-calibrated_tabular)"
        ),
        "score_calibration": (
            "within each seed, logit z-score fitted on that seed's foreign OOF "
            "reference, followed by an average of the three fused pipelines"
        ),
        "missing_image_fallback": (
            "mean of three seed-specific calibrated compact pipelines"
            if selected_branch == ELIGIBLE_CHALLENGER
            else "raw compact ensemble"
        ),
        "outer_f1_threshold_policy": "median of the three seed-specific fold-local inner thresholds",
        "pilot_threshold": pilot_threshold,
        "threshold_is_deployment_calibrated": False,
        "all77_can_affect_selection": False,
    }
    (outputs / "12b_selected_schema.json").write_text(
        json.dumps(final_schema, indent=2), encoding="utf-8"
    )

    artifact_paths = [
        output_root / "kg_08_fusion.py",
        output_root / "kg_12_cv_tabular.py",
        output_root / "kg_12b_confirmatory.py",
        *[cache / name for name in REQUIRED_CACHE_FILES],
        *sorted(final_models.glob("*")),
        *sorted(path for path in outputs.glob("*") if path.name != "12b_manifest.json"),
    ]
    artifact_hashes = {
        path.relative_to(output_root).as_posix(): file_hash(path)
        for path in artifact_paths
        if path.is_file()
    }
    primary_manifest = run_manifests[CONFIRMATORY_SEEDS[0]]
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "training_countries": list(COUNTRIES),
        "holdout_country": HOLDOUT,
        "holdout_loaded": False,
        "interpretation": (
            "Seed-stability confirmation on the same enriched foreign cohort, "
            "not independent country validation or population precision"
        ),
        "selection_policy": (
            "one preregistered guarded CV challenger versus protected compact baseline"
        ),
        "eligible_challenger": ELIGIBLE_CHALLENGER,
        "diagnostic_branches": [DIAGNOSTIC_BRANCH, "nested_champion"],
        "discovery_seed_excluded": DISCOVERY_SEED,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "selected_branch": selected_branch,
        "deployment_branch": selected_branch,
        "acceptance": acceptance,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "integrity": integrity,
        "rounds": rounds,
        "pca_components": pca_components,
        "per_seed_bootstrap_repeats": per_seed_bootstrap_repeats,
        "final_bootstrap_repeats": final_bootstrap_repeats,
        "final_bootstrap_seed": BOOTSTRAP_SEED,
        "cache_protocol": CACHE_PROTOCOL,
        "cache_expected": EXPECTED_CACHE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input": primary_manifest["input"],
        "artifact_sha256": artifact_hashes,
        "elapsed_minutes": (time.time() - started) / 60,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                "torch", "pyarrow", "joblib",
            )
        },
        "limitations": [
            "The confirmation reuses the same enriched 294-source foreign cohort.",
            "Three seeds test algorithmic variance, not new-country sampling variance.",
            "EOG labels identify gas flares; an unlabeled source is not a verified negative.",
            "No India imagery or labels were loaded.",
            "The pilot threshold is not deployment calibrated.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NB12b final decision: {selected_branch}", flush=True)
    return seed_summary, ensemble_summary, manifest


if __name__ == "__main__":
    run_confirmatory()
