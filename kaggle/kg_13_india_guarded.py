"""Frozen NB12b transfer evaluation on a bounded India image panel.

This stage does not fit a model, select a threshold, choose a fusion weight, or
change the source sample after seeing India scores. EOG matches are positives
and every unmatched source remains unlabelled, not a confirmed negative.
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

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from kg_07_context import extract_features
from kg_08_fusion import (
    HOLDOUT,
    IMAGE_COLS,
    THERMAL_COLS,
    THERMAL_RANK,
    THERMAL_RAW,
    file_hash,
)
from kg_12_cv_tabular import extract_cv_embeddings, extract_morphology
from kg_12b_confirmatory import CONFIRMATORY_SEEDS, apply_confirmatory_policy
from kg_imagery_io import (
    BANDS,
    COLLECTION,
    DATES,
    MAX_SCENES,
    MIN_CLEAR,
    PIXEL_M,
    SIZE,
    acquire,
    load_worldcover_keys,
    make_session,
    validate_chip,
)


PROTOCOL = "13-india-guarded-transfer-v1"
PANEL_SEED = 13013
DEFAULT_PANEL_SIZE = 300
DEFAULT_POSITIVE_SITE_QUOTA = 96
MIN_FINAL_COVERAGE = 0.80
FEATURE_FILE = "features_India_2022_2024.parquet"
PANEL_FILE = "india_panel.parquet"
MODEL_PROTOCOL = "12b-guarded-confirmatory-ensemble-v1"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _find_unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name} below {root}; found {len(matches)}: {matches}"
        )
    return matches[0]


def _stable_key(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _stable_panel_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_id",
        "country",
        "block_id",
        "lat",
        "lon",
        "is_eog_flare",
        "eog_flare_id",
        "chip_id",
        "batch_order",
    ]
    canonical = frame[columns].copy()
    for column in columns:
        canonical[column] = canonical[column].fillna("<NA>").astype(str)
    payload = canonical.sort_values("batch_order").to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _review_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def _validate_india(frame: pd.DataFrame) -> None:
    required = {
        "source_id",
        "country",
        "block_id",
        "lat",
        "lon",
        "is_eog_flare",
        "eog_flare_id",
        *THERMAL_RAW,
        *THERMAL_RANK,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"India feature table is missing columns: {sorted(missing)}")
    if frame.empty or not frame.source_id.is_unique or frame.block_id.isna().any():
        raise ValueError("India features need unique sources and non-null 10 km blocks")
    if set(frame.country) != {HOLDOUT}:
        raise ValueError("The transfer feature table must contain India only")
    if not frame.is_eog_flare.isin([0, 1]).all():
        raise ValueError("EOG proxy labels must be binary")
    if frame.loc[frame.is_eog_flare.eq(1), "eog_flare_id"].isna().any():
        raise ValueError("Every positive source must identify its matched EOG site")
    coordinates = frame[["lat", "lon"]].to_numpy(dtype="float64")
    if not np.isfinite(coordinates).all():
        raise ValueError("India source coordinates contain non-finite values")
    if not frame.lat.between(-80, 84).all() or not frame.lon.between(
        -180, 180, inclusive="left"
    ).all():
        raise ValueError("India source coordinates are outside supported bounds")
    thermal = frame[THERMAL_RAW + THERMAL_RANK].to_numpy(dtype="float64")
    if np.isinf(thermal).any():
        raise ValueError("India thermal features contain infinite values")


def _interleave_panel(positive: pd.DataFrame, unlabelled: pd.DataFrame) -> pd.DataFrame:
    """Interleave classes so each bounded download batch is useful."""
    positive_rows = positive.to_dict("records")
    unlabelled_rows = unlabelled.to_dict("records")
    rows = []
    positive_index = 0
    unlabelled_index = 0
    while positive_index < len(positive_rows) or unlabelled_index < len(unlabelled_rows):
        if positive_index < len(positive_rows):
            rows.append(positive_rows[positive_index])
            positive_index += 1
        for _ in range(2):
            if unlabelled_index < len(unlabelled_rows):
                rows.append(unlabelled_rows[unlabelled_index])
                unlabelled_index += 1
        if unlabelled_index >= len(unlabelled_rows) and positive_index < len(positive_rows):
            rows.extend(positive_rows[positive_index:])
            break
    output = pd.DataFrame(rows)
    output["batch_order"] = np.arange(len(output), dtype="int64")
    return output


def select_panel(
    frame: pd.DataFrame,
    n_sources: int = DEFAULT_PANEL_SIZE,
    positive_site_quota: int = DEFAULT_POSITIVE_SITE_QUOTA,
    seed: int = PANEL_SEED,
) -> pd.DataFrame:
    """Freeze an EOG-enriched, spatially separated India transfer panel."""
    _validate_india(frame)
    if n_sources < 30 or not 1 <= positive_site_quota < n_sources:
        raise ValueError("Need n_sources >= 30 and 1 <= positive quota < n_sources")

    ordered = frame.copy()
    ordered["_stable_order"] = ordered.source_id.map(lambda value: _stable_key(value, seed))
    ordered = ordered.sort_values("_stable_order", kind="mergesort")
    positive = (
        ordered.loc[ordered.is_eog_flare.eq(1)]
        .drop_duplicates("eog_flare_id")
        .drop_duplicates("block_id")
        .head(positive_site_quota)
        .copy()
    )
    if len(positive) < min(20, positive_site_quota):
        raise ValueError(f"Only {len(positive)} distinct positive India sites are available")
    unlabelled = (
        ordered.loc[
            ordered.is_eog_flare.eq(0)
            & ~ordered.block_id.isin(set(positive.block_id))
        ]
        .drop_duplicates("block_id")
        .head(n_sources - len(positive))
        .copy()
    )
    if len(positive) + len(unlabelled) != n_sources:
        raise ValueError("Insufficient distinct 10 km blocks for the fixed India panel")

    positive = positive.sort_values("_stable_order", kind="mergesort")
    unlabelled = unlabelled.sort_values("_stable_order", kind="mergesort")
    panel = _interleave_panel(positive, unlabelled).drop(columns="_stable_order")
    panel["chip_id"] = panel.source_id.map(
        lambda value: hashlib.sha256(f"India:{value}".encode("utf-8")).hexdigest()[:20]
    )
    if not panel.source_id.is_unique or not panel.chip_id.is_unique:
        raise ValueError("Panel source and chip IDs must be unique")
    if panel.block_id.duplicated().any():
        raise ValueError("Panel sources must occupy distinct 10 km blocks")
    return panel.sort_values("batch_order").reset_index(drop=True)


def prepare(
    input_root: str | Path,
    output_root: str | Path,
    n_sources: int = DEFAULT_PANEL_SIZE,
    positive_site_quota: int = DEFAULT_POSITIVE_SITE_QUOTA,
    seed: int = PANEL_SEED,
) -> pd.DataFrame:
    """Load NB2 India features and freeze the panel before image acquisition."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    source_path = _find_unique(input_root, FEATURE_FILE)
    wanted = list(dict.fromkeys([
        "source_id",
        "country",
        "block_id",
        "lat",
        "lon",
        "is_eog_flare",
        "eog_flare_id",
        *THERMAL_RAW,
        *THERMAL_RANK,
    ]))
    frame = pd.read_parquet(source_path, columns=wanted)
    _validate_india(frame)
    for column in THERMAL_RANK:
        frame[f"{column}_country_pct"] = frame[column].rank(
            method="average", pct=True
        ).astype("float32")
    panel = select_panel(frame, n_sources, positive_site_quota, seed)
    panel = panel[[
        "source_id",
        "country",
        "block_id",
        "lat",
        "lon",
        "is_eog_flare",
        "eog_flare_id",
        "chip_id",
        "batch_order",
        *THERMAL_COLS,
    ]]
    config = {
        "protocol": PROTOCOL,
        "purpose": "frozen NB12b India transfer ranking on an EOG-enriched panel",
        "country": HOLDOUT,
        "years": [2022, 2023, 2024],
        "seed": seed,
        "n_sources": n_sources,
        "positive_site_quota": positive_site_quota,
        "actual_positive_sites": int(panel.is_eog_flare.sum()),
        "sample_is_eog_enriched": True,
        "unmatched_means_unlabelled": True,
        "model_fitting": False,
        "threshold_selection": False,
        "sentinel_collection": COLLECTION,
        "dates": DATES,
        "bands": BANDS,
        "shape": [6, SIZE, SIZE],
        "pixel_m": PIXEL_M,
        "minimum_clear_fraction": MIN_CLEAR,
        "max_scenes": MAX_SCENES,
        "feature_input_sha256": file_hash(source_path),
        "panel_sha256": _stable_panel_hash(panel),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Existing output root uses a different frozen India panel")
    _write_json(config_path, config)
    panel.to_parquet(output_root / PANEL_FILE, index=False)
    panel[[
        "source_id", "country", "block_id", "lat", "lon", "is_eog_flare",
        "eog_flare_id", "chip_id", "batch_order",
    ]].to_csv(output_root / "india_panel.csv", index=False)
    return panel


def _prior_roots(input_root: Path, expected_config: dict) -> list[Path]:
    roots = []
    for path in sorted(input_root.rglob("run_config.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            config.get("protocol") == PROTOCOL
            and config.get("panel_sha256") == expected_config["panel_sha256"]
            and config.get("feature_input_sha256")
            == expected_config["feature_input_sha256"]
        ):
            roots.append(path.parent)
    return roots


def _existing_record(
    row: dict,
    output_root: Path,
    prior_roots: list[Path],
) -> dict | None:
    cached_failure = None
    for directory in [output_root, *prior_roots]:
        sidecar = directory / f"{row['chip_id']}.json"
        chip = directory / f"{row['chip_id']}.npz"
        if not sidecar.is_file():
            continue
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(
            str(record.get(key)) != str(row[key])
            for key in ("source_id", "country", "chip_id")
        ):
            raise ValueError(f"Cached record identity mismatch: {sidecar}")
        if record.get("status") == "failed":
            cached_failure = record
            continue
        if record.get("status") != "ok":
            continue
        if not chip.is_file():
            raise FileNotFoundError(f"Successful record has no chip: {chip}")
        validate_chip(chip, row, record)
        if directory.resolve() != output_root.resolve():
            shutil.copy2(chip, output_root / chip.name)
            copied = dict(record)
            copied["reused_from"] = str(directory)
            _write_json(output_root / sidecar.name, copied)
            return copied
        return record
    if cached_failure is not None:
        _write_json(output_root / f"{row['chip_id']}.json", cached_failure)
    return cached_failure


def _save_download_manifest(
    output_root: Path,
    panel: pd.DataFrame,
    records: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    omitted = {"stac_item", "rejected_scenes", "worldcover_urls"}
    for row in panel.to_dict("records"):
        record = records.get(row["chip_id"], {})
        summary = {key: value for key, value in record.items() if key not in omitted}
        rows.append({
            "source_id": row["source_id"],
            "country": row["country"],
            "chip_id": row["chip_id"],
            "is_eog_flare": int(row["is_eog_flare"]),
            "status": "pending",
            **summary,
        })
    manifest = pd.DataFrame(rows)
    temporary = output_root / "download_manifest.tmp.csv"
    manifest.to_csv(temporary, index=False)
    temporary.replace(output_root / "download_manifest.csv")
    return manifest


def run_batch(
    output_root: str | Path,
    input_root: str | Path = "/kaggle/input",
    max_new: int = 100,
    max_minutes: float = 55,
    retry_failed: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Reuse prior saved outputs, then acquire one bounded image batch."""
    output_root = Path(output_root)
    input_root = Path(input_root)
    if max_new < 0 or max_minutes <= 0:
        raise ValueError("max_new must be nonnegative and max_minutes positive")
    config = json.loads((output_root / "run_config.json").read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Output root does not use the India transfer protocol")
    panel = pd.read_parquet(output_root / PANEL_FILE)
    if _stable_panel_hash(panel) != config["panel_sha256"]:
        raise ValueError("Frozen India panel changed after preparation")
    prior_roots = _prior_roots(input_root, config)
    records: dict[str, dict] = {}
    for row in panel.to_dict("records"):
        record = _existing_record(row, output_root, prior_roots)
        if record is not None:
            records[row["chip_id"]] = record
    manifest = _save_download_manifest(output_root, panel, records)

    attempts = 0
    consecutive_failures = 0
    started = time.monotonic()
    session = None
    available = None
    stop_reason = "all eligible sources processed"
    try:
        for row in panel.sort_values("batch_order").to_dict("records"):
            old = records.get(row["chip_id"], {})
            if old.get("status") == "ok" or (
                old.get("status") == "failed" and not retry_failed
            ):
                continue
            elapsed = (time.monotonic() - started) / 60
            if offline or attempts >= max_new or elapsed >= max_minutes:
                stop_reason = "offline or per-version attempt/time limit"
                break
            if shutil.disk_usage(output_root).free < 1024**3:
                stop_reason = "less than 1 GiB free disk space"
                break
            if session is None:
                session = make_session()
                available = load_worldcover_keys(
                    session, output_root / "worldcover_keys.json"
                )
            record = acquire(row, output_root, session, available)
            record.update(
                requested_lon=float(row["lon"]),
                requested_lat=float(row["lat"]),
            )
            _write_json(output_root / f"{row['chip_id']}.json", record)
            records[row["chip_id"]] = record
            attempts += 1
            manifest = _save_download_manifest(output_root, panel, records)
            print(
                attempts,
                HOLDOUT,
                record["status"],
                record.get("error", "")[:180],
                flush=True,
            )
            consecutive_failures = (
                consecutive_failures + 1 if record["status"] == "failed" else 0
            )
            if consecutive_failures >= 6:
                stop_reason = "six consecutive failures, inspect errors before retrying"
                break
    finally:
        if session is not None:
            session.close()

    counts = manifest.status.value_counts().to_dict()
    state = {
        "protocol": PROTOCOL,
        "counts": {key: int(value) for key, value in counts.items()},
        "new_attempts": attempts,
        "elapsed_minutes": round((time.monotonic() - started) / 60, 2),
        "stop_reason": stop_reason,
        "panel_complete": int(counts.get("pending", 0)) == 0,
        "model_fitting": False,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "requests", "pyproj")
        },
    }
    _write_json(output_root / "run_state.json", state)
    return manifest


def export_image_features(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract image summaries and QA without using labels in feature creation."""
    output_root = Path(output_root)
    panel = pd.read_parquet(output_root / PANEL_FILE)
    manifest = pd.read_csv(output_root / "download_manifest.csv")
    if not manifest.chip_id.is_unique or set(manifest.chip_id) != set(panel.chip_id):
        raise ValueError("Download manifest does not match the frozen India panel")
    status = manifest.set_index("chip_id").status
    feature_rows = []
    quality_rows = []
    for row in panel.to_dict("records"):
        row_status = status.loc[row["chip_id"]]
        features: dict[str, float] = {}
        quality: dict[str, float | bool] = {}
        if row_status == "ok":
            sidecar = output_root / f"{row['chip_id']}.json"
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            chip = output_root / f"{row['chip_id']}.npz"
            validate_chip(chip, row, record)
            features, quality = extract_features(chip)
        feature_rows.append({"source_id": row["source_id"], **features})
        quality_rows.append({
            "source_id": row["source_id"],
            "country": HOLDOUT,
            "status": row_status,
            **quality,
        })
    image = pd.DataFrame(feature_rows)
    quality_frame = pd.DataFrame(quality_rows)
    if "review_reflectance_tail" not in quality_frame:
        quality_frame["review_reflectance_tail"] = False
    feature_columns = [column for column in image if column.startswith("img_")]
    if feature_columns and np.isinf(
        image[feature_columns].to_numpy(dtype="float64")
    ).any():
        raise ValueError("Extracted image features contain infinite values")
    image.to_parquet(output_root / "image_features.parquet", index=False)
    quality_frame.to_csv(output_root / "image_quality.csv", index=False)
    coverage = panel[["source_id", "is_eog_flare"]].merge(
        quality_frame[["source_id", "status", "review_reflectance_tail"]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    coverage["review_reflectance_tail"] = _review_mask(
        coverage["review_reflectance_tail"]
    )
    coverage.groupby(
        ["is_eog_flare", "status", "review_reflectance_tail"], observed=True
    ).size().rename("n").reset_index().to_csv(
        output_root / "coverage_by_label.csv", index=False
    )
    return image, quality_frame


def _chip_inventory(
    output_root: Path,
    cohort: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[dict[object, Path], str]:
    chip_ids = panel.set_index("source_id").chip_id
    paths = {
        source_id: output_root / f"{chip_ids.loc[source_id]}.npz"
        for source_id in cohort.source_id
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing QA-approved image chip: {missing[0]}")
    digest = hashlib.sha256()
    for source_id in sorted(paths, key=str):
        digest.update(str(source_id).encode("utf-8"))
        digest.update(file_hash(paths[source_id]).encode("utf-8"))
    return paths, digest.hexdigest()


def _verified_model_inputs(repo_root: Path) -> tuple[Path, dict, dict]:
    manifest_path = repo_root / "results/nb12b_confirmatory/12b_manifest.json"
    schema_path = repo_root / "results/nb12b_confirmatory/12b_selected_schema.json"
    models = repo_root / "artifacts/nb12b/final_models"
    if not manifest_path.is_file() or not schema_path.is_file() or not models.is_dir():
        raise FileNotFoundError("The repository is missing frozen NB12b artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != MODEL_PROTOCOL
        or manifest.get("status") != "complete"
        or manifest.get("selected_branch") != "guarded_cv_tabular"
        or schema.get("selected_branch") != "guarded_cv_tabular"
        or schema.get("threshold_is_deployment_calibrated") is not False
    ):
        raise ValueError("Frozen NB12b manifest or deployment schema is incompatible")
    hashes = manifest.get("artifact_sha256", {})
    required = []
    for pipeline in schema["model_ensemble"]["seed_pipelines"]:
        required.extend(pipeline["compact_models"])
        required.append(pipeline["visual_model"])
    required.append(schema["model_ensemble"]["calibration_file"])
    for name in required:
        path = models / name
        expected = hashes.get(f"final_models/{name}")
        if not path.is_file() or expected is None or file_hash(path) != expected:
            raise ValueError(f"Frozen artifact failed SHA-256 verification: {name}")
    expected_schema_hash = hashes.get("outputs/12b_selected_schema.json")
    if expected_schema_hash is None or file_hash(schema_path) != expected_schema_hash:
        raise ValueError("Frozen deployment schema failed SHA-256 verification")
    return models, manifest, schema


def _review_budget_metrics(
    predictions: pd.DataFrame,
    score_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    positives = int(predictions.is_eog_flare.sum())
    positive_sites = int(
        predictions.loc[predictions.is_eog_flare.eq(1), "eog_flare_id"].nunique()
    )
    for score_column in score_columns:
        ordered = predictions.assign(
            _source_tiebreak=predictions.source_id.astype(str)
        ).sort_values(
            [score_column, "_source_tiebreak"],
            ascending=[False, True],
            kind="mergesort",
        )
        for fraction in (0.10, 0.20, 0.30):
            count = max(1, int(math.ceil(fraction * len(ordered))))
            reviewed = ordered.head(count)
            found = int(reviewed.is_eog_flare.sum())
            sites_found = int(
                reviewed.loc[reviewed.is_eog_flare.eq(1), "eog_flare_id"].nunique()
            )
            rows.append({
                "branch": score_column.removeprefix("score_"),
                "review_fraction": fraction,
                "reviewed_sources": count,
                "positive_rows_found": found,
                "positive_row_recall": found / max(positives, 1),
                "positive_sites_found": sites_found,
                "positive_site_recall": sites_found / max(positive_sites, 1),
            })
    return pd.DataFrame(rows)


def _block_bootstrap(
    predictions: pd.DataFrame,
    repeats: int = 2000,
    seed: int = 13014,
) -> dict:
    """Estimate guarded minus compact AP with positive-stratified block resampling."""
    groups = {
        block_id: index.to_numpy(dtype="int64")
        for block_id, index in predictions.groupby("block_id").groups.items()
    }
    positive_blocks = [
        block_id
        for block_id, index in groups.items()
        if bool(predictions.loc[index, "is_eog_flare"].any())
    ]
    background_blocks = [
        block_id for block_id in groups if block_id not in set(positive_blocks)
    ]
    if not positive_blocks or not background_blocks:
        raise ValueError("Bootstrap requires positive and unlabelled 10 km blocks")
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype="float64")
    for repeat in range(repeats):
        sampled_blocks = np.concatenate([
            rng.choice(positive_blocks, size=len(positive_blocks), replace=True),
            rng.choice(background_blocks, size=len(background_blocks), replace=True),
        ])
        sampled_index = np.concatenate([groups[block] for block in sampled_blocks])
        sampled = predictions.iloc[sampled_index]
        y = sampled.is_eog_flare.to_numpy(dtype="int8")
        differences[repeat] = average_precision_score(
            y, sampled.score_guarded_cv_tabular
        ) - average_precision_score(y, sampled.score_compact_tabular)
    return {
        "metric": "EOG-proxy PR-AUC difference, guarded minus compact",
        "mean": float(differences.mean()),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "repeats": repeats,
        "seed": seed,
        "resampling_unit": "10 km block, stratified by EOG-positive status",
    }


def _india_decision(metrics: pd.DataFrame, review: pd.DataFrame) -> dict:
    """Apply the precommitted India ranking guard without threshold tuning."""
    by_branch = metrics.set_index("branch")
    required = {"compact_tabular", "guarded_cv_tabular"}
    if set(by_branch.index) != required:
        raise ValueError("India decision requires exactly compact and guarded metrics")
    at_twenty = review.loc[review.review_fraction.eq(0.20)].set_index("branch")
    if set(at_twenty.index) != required:
        raise ValueError("India decision requires both branches at a 20 percent budget")
    pr_auc_delta = float(
        by_branch.loc["guarded_cv_tabular", "eog_proxy_pr_auc"]
        - by_branch.loc["compact_tabular", "eog_proxy_pr_auc"]
    )
    site_delta = int(
        at_twenty.loc["guarded_cv_tabular", "positive_sites_found"]
        - at_twenty.loc["compact_tabular", "positive_sites_found"]
    )
    conditions = {
        "pr_auc_delta_at_least_minus_0_01": pr_auc_delta >= -0.01,
        "positive_site_loss_at_20_no_more_than_one": site_delta >= -1,
    }
    passed = all(conditions.values())
    return {
        "protocol": PROTOCOL,
        "rule_written_before_results": True,
        "eog_proxy_pr_auc_delta_guarded_minus_compact": pr_auc_delta,
        "positive_sites_at_20_delta_guarded_minus_compact": site_delta,
        "conditions": conditions,
        "passed": passed,
        "india_ranking_branch": (
            "guarded_cv_tabular" if passed else "compact_tabular"
        ),
        "interpretation": (
            "Panel ranking decision only; not population precision or a deployment threshold"
        ),
    }


def score_frozen_model(
    output_root: str | Path,
    repo_root: str | Path,
    checkpoint_path: str | Path | None = None,
    bootstrap_repeats: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the exact frozen NB12b ensemble, then evaluate India rankings."""
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    state = json.loads((output_root / "run_state.json").read_text(encoding="utf-8"))
    pending = int(state.get("counts", {}).get("pending", 0))
    if pending:
        raise RuntimeError(
            f"India panel still has {pending} pending chips. Save this version, attach "
            "its output to the next run, and rerun before scoring."
        )
    panel = pd.read_parquet(output_root / PANEL_FILE)
    image = pd.read_parquet(output_root / "image_features.parquet")
    quality = pd.read_csv(output_root / "image_quality.csv")
    if not image.source_id.is_unique or not quality.source_id.is_unique:
        raise ValueError("Image feature and quality tables need unique source IDs")
    quality["review_reflectance_tail"] = _review_mask(
        quality["review_reflectance_tail"]
    )
    cohort = panel.merge(image, on="source_id", how="left", validate="one_to_one")
    cohort = cohort.merge(
        quality[["source_id", "status", "review_reflectance_tail"]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    cohort = cohort.loc[
        cohort.status.eq("ok") & ~cohort.review_reflectance_tail
    ].copy().sort_values("batch_order").reset_index(drop=True)
    if len(cohort) < math.ceil(MIN_FINAL_COVERAGE * len(panel)):
        raise ValueError(
            f"Only {len(cohort)}/{len(panel)} panel sources passed image QA, below "
            f"the frozen {MIN_FINAL_COVERAGE:.0%} coverage requirement"
        )
    if cohort.is_eog_flare.sum() < 20 or cohort.is_eog_flare.eq(0).sum() < 40:
        raise ValueError("QA-approved India panel has insufficient label coverage")

    models_root, model_manifest, schema = _verified_model_inputs(repo_root)
    expected_versions = model_manifest.get("versions", {})
    for package in ("scikit-learn", "lightgbm", "joblib"):
        expected = expected_versions.get(package)
        observed = importlib.metadata.version(package)
        if expected is None or observed != expected:
            raise RuntimeError(
                f"Frozen NB12b requires {package}=={expected}, found {observed}. "
                "Use the version-pinning setup cell before importing the scorer."
            )
    missing_tabular = set(schema["tabular_columns"]) - set(cohort.columns)
    if missing_tabular:
        raise ValueError(f"India panel is missing model columns: {sorted(missing_tabular)}")
    inventory = _chip_inventory(output_root, cohort, panel)
    embeddings, embedding_manifest = extract_cv_embeddings(
        output_root,
        cohort,
        output_root / "13_cv_embeddings.parquet",
        checkpoint_path=checkpoint_path,
        batch_size=32,
        tta=True,
        chip_inventory=inventory,
    )
    morphology, morphology_manifest = extract_morphology(
        output_root,
        cohort,
        output_root / "13_morphology.parquet",
        chip_inventory=inventory,
    )
    cohort = cohort.merge(embeddings, on="source_id", validate="one_to_one")
    cohort = cohort.merge(morphology, on="source_id", validate="one_to_one")
    tabular_columns = schema["tabular_columns"]
    embedding_columns = schema["embedding_columns"]
    auxiliary_columns = schema["image_columns"] + schema["morphology_columns"]
    for columns, name in (
        (tabular_columns, "tabular"),
        (embedding_columns, "embedding"),
        (auxiliary_columns, "visual auxiliary"),
    ):
        missing = set(columns) - set(cohort.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        values = cohort[columns].to_numpy(dtype="float32")
        if np.isinf(values).any():
            raise ValueError(f"{name.title()} matrix contains infinite values")

    compact_scores = np.empty(
        (len(cohort), len(CONFIRMATORY_SEEDS), 3), dtype="float64"
    )
    visual_scores = np.empty(
        (len(cohort), len(CONFIRMATORY_SEEDS)), dtype="float64"
    )
    tabular_matrix = cohort[tabular_columns].to_numpy(dtype="float32")
    embedding_matrix = cohort[embedding_columns].to_numpy(dtype="float32")
    auxiliary_matrix = cohort[auxiliary_columns].to_numpy(dtype="float32")
    pipelines = schema["model_ensemble"]["seed_pipelines"]
    if [int(item["seed"]) for item in pipelines] != list(CONFIRMATORY_SEEDS):
        raise ValueError("Deployment pipeline seeds do not match the frozen protocol")
    for seed_index, pipeline in enumerate(pipelines):
        for model_index, name in enumerate(pipeline["compact_models"]):
            model = lgb.Booster(model_file=str(models_root / name))
            observed_names = model.feature_name()
            generic_names = [f"Column_{index}" for index in range(len(tabular_columns))]
            if observed_names not in (tabular_columns, generic_names):
                raise ValueError(f"LightGBM feature order mismatch: {name}")
            compact_scores[:, seed_index, model_index] = model.predict(tabular_matrix)
        visual_model = joblib.load(models_root / pipeline["visual_model"])
        visual_scores[:, seed_index] = visual_model.predict(
            embedding_matrix, auxiliary_matrix
        )

    calibration_path = models_root / schema["model_ensemble"]["calibration_file"]
    with np.load(calibration_path, allow_pickle=False) as calibration:
        seeds = calibration["seeds"].astype("int64")
        compact_reference = calibration["compact_oof"].astype("float64")
        visual_reference = calibration["visual_oof"].astype("float64")
        alphas = calibration["fusion_alpha"].astype("float64")
    if seeds.tolist() != list(CONFIRMATORY_SEEDS):
        raise ValueError("Calibration seed order does not match the frozen ensemble")
    inference_frame = cohort[["source_id", "country"]].copy()
    score_compact = apply_confirmatory_policy(
        inference_frame, compact_scores, "compact_tabular"
    )
    score_guarded = apply_confirmatory_policy(
        inference_frame,
        compact_scores,
        "guarded_cv_tabular",
        visual_model_scores=visual_scores,
        seed_fusion_alphas=alphas,
        compact_oof_reference=compact_reference,
        visual_oof_reference=visual_reference,
        image_available=np.ones(len(cohort), dtype=bool),
    )
    if not np.isfinite(np.column_stack([score_compact, score_guarded])).all():
        raise ValueError("Frozen ensemble produced non-finite India scores")

    predictions = cohort[[
        "source_id",
        "block_id",
        "country",
        "lat",
        "lon",
        "is_eog_flare",
        "eog_flare_id",
        "chip_id",
    ]].copy()
    predictions["score_compact_tabular"] = score_compact
    predictions["score_guarded_cv_tabular"] = score_guarded
    for score_column in ("score_compact_tabular", "score_guarded_cv_tabular"):
        predictions[f"rank_{score_column.removeprefix('score_')}"] = (
            predictions[score_column]
            .rank(method="first", ascending=False)
            .astype("int64")
        )

    y = predictions.is_eog_flare.to_numpy(dtype="int8")
    metric_rows = []
    for score_column in ("score_compact_tabular", "score_guarded_cv_tabular"):
        score = predictions[score_column].to_numpy(dtype="float64")
        metric_rows.append({
            "branch": score_column.removeprefix("score_"),
            "sources": len(predictions),
            "eog_positive_rows": int(y.sum()),
            "unlabelled_rows": int((y == 0).sum()),
            "eog_proxy_pr_auc": float(average_precision_score(y, score)),
            "eog_proxy_roc_auc": float(roc_auc_score(y, score)),
            "f1": np.nan,
            "threshold": np.nan,
            "threshold_note": "not reported because NB12b has no deployment-calibrated threshold",
        })
    metrics = pd.DataFrame(metric_rows)
    review = _review_budget_metrics(
        predictions, ("score_compact_tabular", "score_guarded_cv_tabular")
    )
    bootstrap = _block_bootstrap(predictions, bootstrap_repeats)
    decision = _india_decision(metrics, review)

    predictions.to_parquet(output_root / "13_india_predictions.parquet", index=False)
    metrics.to_csv(output_root / "13_india_ranking_metrics.csv", index=False)
    review.to_csv(output_root / "13_india_review_budgets.csv", index=False)
    _write_json(output_root / "13_india_block_bootstrap.json", bootstrap)
    _write_json(output_root / "13_india_decision.json", decision)
    predictions.sort_values(
        ["score_guarded_cv_tabular", "source_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(100).to_csv(output_root / "13_india_review_top100.csv", index=False)

    output_names = [
        "13_india_predictions.parquet",
        "13_india_ranking_metrics.csv",
        "13_india_review_budgets.csv",
        "13_india_block_bootstrap.json",
        "13_india_decision.json",
        "13_india_review_top100.csv",
        "13_cv_embeddings.parquet",
        "13_cv_embeddings.json",
        "13_morphology.parquet",
        "13_morphology.json",
    ]
    run_manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "country": HOLDOUT,
        "evaluation_role": "final-model transfer audit, not a first untouched holdout",
        "model_action": "loaded frozen NB12b models and calibration; no fitting",
        "selection_action": "no feature, model, threshold, alpha, or sample selection from India scores",
        "selected_branch": schema["selected_branch"],
        "panel_sources": len(panel),
        "qa_scored_sources": len(predictions),
        "qa_coverage": len(predictions) / len(panel),
        "panel_positive_sites": int(panel.is_eog_flare.sum()),
        "scored_positive_sites": int(y.sum()),
        "sample_is_eog_enriched": True,
        "population_precision_valid": False,
        "unmatched_means_unlabelled": True,
        "threshold_reported": False,
        "model_manifest_sha256": file_hash(
            repo_root / "results/nb12b_confirmatory/12b_manifest.json"
        ),
        "deployment_schema_sha256": file_hash(
            repo_root / "results/nb12b_confirmatory/12b_selected_schema.json"
        ),
        "panel_sha256": _stable_panel_hash(panel),
        "embedding_manifest": embedding_manifest,
        "morphology_manifest": morphology_manifest,
        "output_sha256": {
            name: file_hash(output_root / name) for name in output_names
        },
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy", "pandas", "scikit-learn", "lightgbm", "torch",
                "pyarrow", "joblib",
            )
        },
        "limitations": [
            "India was previously evaluated with a superseded model, so this is not a pristine project-level holdout.",
            "The fixed panel is EOG-enriched and cannot estimate population precision.",
            "EOG labels cover gas flares, while unmatched sources remain unlabelled.",
            "The panel tests ranking transfer on sources with QA-approved imagery, not all India candidates.",
            "No F1 is reported because the frozen NB12b threshold is not deployment-calibrated.",
        ],
        "training_countries": model_manifest["training_countries"],
    }
    _write_json(output_root / "13_manifest.json", run_manifest)
    return metrics, review, predictions


def bundle(output_root: str | Path) -> Path:
    """Bundle small audit outputs. Chips remain in the saved notebook output for resume."""
    import zipfile

    output_root = Path(output_root)
    output = output_root.parent / "nb13_india_transfer_results.zip"
    allowed_suffixes = {".csv", ".json", ".parquet"}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.iterdir()):
            if (
                path.is_file()
                and path.suffix in allowed_suffixes
                and path.suffix != ".npz"
                and not path.name.startswith("worldcover_keys")
            ):
                archive.write(path, arcname=f"nb13_india_transfer/{path.name}")
        code_path = Path(__file__)
        archive.write(code_path, arcname=f"code/{code_path.name}")
    return output
