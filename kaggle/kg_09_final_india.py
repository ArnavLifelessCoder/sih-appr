"""Stage 09: frozen NB5 model inference and evaluation on India.

This module never fits a model or selects a threshold. It builds India's feature
table on the same 2022 to 2024 observation window, applies the saved NB5 model,
and then evaluates the frozen predictions against EOG proxy labels.
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

from kg_03_features import FEATURE_BLOCKLIST, build as build_features
from kg_05b_robust_tabular import FEATURE_TAG, WINDOW_YEARS
from kg_common import CACHE, HOLDOUT, eog_sites
from kg_eval import metrics


PROTOCOL_VERSION = "09-frozen-india-v1"
REQUIRED_NB5_PROTOCOL = "05d-nested-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name} below {root}; found {len(matches)}: {matches}"
        )
    return matches[0]


def validate_nb5_manifest(manifest: dict) -> list[str]:
    if manifest.get("protocol") != REQUIRED_NB5_PROTOCOL:
        raise ValueError("The model must come from the completed NB5 nested protocol")
    if manifest.get("status") != "complete":
        raise ValueError("NB5 did not finish successfully")
    if manifest.get("holdout_country") != HOLDOUT or manifest.get("holdout_loaded") is not False:
        raise ValueError("NB5 does not document India as an untouched holdout")
    if manifest.get("selected_variant", {}).get("name") != "unweighted":
        raise ValueError("Unexpected NB5 selected model variant")
    features = manifest.get("features")
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise ValueError("NB5 feature schema is empty or duplicated")
    forbidden = set(features) & FEATURE_BLOCKLIST
    if forbidden:
        raise ValueError(f"Forbidden model features found: {sorted(forbidden)}")
    threshold = manifest.get("final_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("NB5 threshold is invalid")
    return features


def validate_india_features(frame: pd.DataFrame, feature_names: list[str]) -> None:
    required = {
        "source_id", "country", "block_id", "lat", "lon",
        "is_eog_flare", "eog_flare_id",
    }
    missing = (required | set(feature_names)) - set(frame.columns)
    if missing:
        raise ValueError(f"India feature table is missing columns: {sorted(missing)}")
    if frame.empty or not frame.source_id.is_unique:
        raise ValueError("India feature rows must be nonempty with unique source IDs")
    if set(frame.country) != {HOLDOUT}:
        raise ValueError("India feature table contains another country")
    if not frame.is_eog_flare.isin([0, 1]).all():
        raise ValueError("EOG match labels must be binary")
    matrix = frame[feature_names].to_numpy(dtype="float32")
    if np.isinf(matrix).any():
        raise ValueError("Model features contain infinite values")


def score_features(
    frame: pd.DataFrame,
    model: lgb.Booster,
    nb5_manifest: dict,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """Score India without exposing its labels to model or threshold selection."""
    feature_names = validate_nb5_manifest(nb5_manifest)
    validate_india_features(frame, feature_names)
    if model.num_feature() != len(feature_names):
        raise ValueError(
            f"Model expects {model.num_feature()} features, manifest lists {len(feature_names)}"
        )
    scores = np.empty(len(frame), dtype="float64")
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        matrix = frame.iloc[start:stop][feature_names].to_numpy(dtype="float32")
        scores[start:stop] = model.predict(matrix)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Model returned invalid scores")

    threshold = float(nb5_manifest["final_threshold"])
    output = frame[[
        "source_id", "block_id", "country", "lat", "lon",
        "is_eog_flare", "eog_flare_id",
    ]].copy()
    output["eog_like_score"] = scores
    output["predicted_eog_like"] = scores >= threshold
    output["score_rank"] = pd.Series(scores).rank(method="first", ascending=False).astype("int64")
    return output


def evaluate_frozen_predictions(
    predictions: pd.DataFrame,
    threshold: float,
    eog_sites_total: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = predictions.is_eog_flare.to_numpy(dtype="int8")
    score = predictions.eog_like_score.to_numpy(dtype="float64")
    row = metrics(
        y,
        score,
        threshold,
        name="NB5 frozen model on India",
        extra={
            "country": HOLDOUT,
            "threshold_source": "NB5 foreign-country grouped OOF",
            "labels": "EOG matches are positive; unmatched sources are unlabelled",
        },
    )
    labelled = predictions.loc[
        predictions.is_eog_flare.eq(1) & predictions.eog_flare_id.notna()
    ]
    recoverable = int(labelled.eog_flare_id.nunique())
    detected = int(
        labelled.loc[labelled.predicted_eog_like, "eog_flare_id"].nunique()
    )
    sites = pd.DataFrame([{
        "country": HOLDOUT,
        "eog_sites_total": int(eog_sites_total),
        "eog_sites_recoverable": recoverable,
        "eog_sites_detected": detected,
        "source_construction_recall": recoverable / max(int(eog_sites_total), 1),
        "model_recall_of_recoverable": detected / max(recoverable, 1),
        "end_to_end_recall_of_all_active": detected / max(int(eog_sites_total), 1),
    }])
    return pd.DataFrame([row]), sites


def _copy_india_inputs(input_root: Path) -> dict[str, str]:
    copied = {}
    for name in ["detections_India.parquet", "sources_India.parquet"]:
        source = find_unique(input_root, name)
        destination = CACHE / name
        shutil.copy2(source, destination)
        copied[name] = sha256(destination)
    return copied


def run(
    input_root: str | Path = "/kaggle/input",
    output_dir: str | Path = "/kaggle/working/nb7_final_india",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output_dir / "09_manifest.json"
    if final_manifest_path.exists():
        raise FileExistsError("India has already been scored in this output directory")

    nb5_manifest_path = find_unique(input_root, "05d_manifest.json")
    model_path = find_unique(input_root, "05d_final_foreign_model.txt")
    nb5_manifest = json.loads(nb5_manifest_path.read_text(encoding="utf-8"))
    feature_names = validate_nb5_manifest(nb5_manifest)
    copied_hashes = _copy_india_inputs(input_root)

    started = time.time()
    india = build_features(HOLDOUT, years=WINDOW_YEARS, output_tag=FEATURE_TAG)
    model = lgb.Booster(model_file=str(model_path))
    predictions = score_features(india, model, nb5_manifest)

    total_sites = int(
        eog_sites(active_years=WINDOW_YEARS).country.eq(HOLDOUT).sum()
    )
    summary, site_summary = evaluate_frozen_predictions(
        predictions, float(nb5_manifest["final_threshold"]), total_sites
    )
    predictions.to_parquet(output_dir / "09_india_predictions.parquet", index=False)
    predictions.sort_values(
        ["predicted_eog_like", "eog_like_score"], ascending=False
    ).head(1000).to_csv(output_dir / "09_review_top1000.csv", index=False)
    summary.to_csv(output_dir / "09_india_eog_proxy_metrics.csv", index=False)
    site_summary.to_csv(output_dir / "09_india_site_recall.csv", index=False)

    output_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.glob("09_*.csv"))
    }
    output_hashes["09_india_predictions.parquet"] = sha256(
        output_dir / "09_india_predictions.parquet"
    )
    run_manifest = {
        "protocol": PROTOCOL_VERSION,
        "status": "complete",
        "model_action": "loaded frozen NB5 model; no fitting or tuning",
        "country": HOLDOUT,
        "years": list(WINDOW_YEARS),
        "n_sources": len(india),
        "n_features": len(feature_names),
        "features": feature_names,
        "threshold": float(nb5_manifest["final_threshold"]),
        "threshold_source": "NB5 foreign-country grouped OOF",
        "nb5_manifest_sha256": sha256(nb5_manifest_path),
        "nb5_model_sha256": sha256(model_path),
        "india_input_sha256": copied_hashes,
        "output_sha256": output_hashes,
        "elapsed_minutes": (time.time() - started) / 60,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "lightgbm", "pyarrow"]
        },
        "interpretation": (
            "Scores estimate similarity to EOG-labelled gas flares. Unmatched "
            "sources are unlabelled, so nominal precision and F1 are proxy metrics."
        ),
    }
    final_manifest_path.write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    return summary, site_summary, predictions


if __name__ == "__main__":
    run()
