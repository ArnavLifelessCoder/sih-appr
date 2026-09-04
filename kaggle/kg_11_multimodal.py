"""Leakage-safe Sentinel-2, temporal, and FIRMS fusion on foreign countries.

The labelled imagery cohort is too small for end-to-end neural fine-tuning.
This stage therefore freezes a Sentinel-2-pretrained image encoder, uses a
population-trained TCN only through nested country-held-out scores, and keeps
the proven NB9 compact LightGBM branch as the mandatory baseline.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import urllib.error
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from kg_08_fusion import (
    BRANCHES as NB9_BRANCHES,
    COUNTRIES,
    HOLDOUT,
    PARAMS as NB9_PARAMS,
    find_unique,
    load_cohort,
    macro_ap,
    macro_f1_threshold,
    metric_row,
)
from kg_10_temporal_tcn import (
    fit_final_tcn,
    precompute_nested_scores,
    prepare_temporal_data,
)


PROTOCOL = "11-multimodal-country-loco-v1"
BASE_COLUMNS = NB9_BRANCHES["early_fusion"]
TCN_STRUCTURED_WEIGHT = 0.80
STAGE1_GATE_FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.02)
SSL_WEIGHT_NAME = "SENTINEL2_RGB_MOCO"
SSL_WEIGHT_URL = (
    "https://hf.co/torchgeo/resnet18_sentinel2_rgb_moco/resolve/"
    "e1c032e7785fd0625224cdb6699aa138bb304eec/"
    "resnet18_sentinel2_rgb_moco-e3a335e3.pth"
)
VARIANTS = {
    "nb9_baseline": {"temporal": False, "ssl": False},
    "temporal_features": {"temporal": True, "ssl": False},
    "ssl_features": {"temporal": False, "ssl": True},
    "structured_full": {"temporal": True, "ssl": True},
}
FORBIDDEN_FEATURE_COLUMNS = {
    "country", "latitude", "longitude", "lat", "lon", "type",
    "eog_dist_m", "eog_flare_id", "is_eog_flare", "block_id",
}
# Keep the reference branch exactly aligned with the completed NB9 experiment.
PARAMS = dict(NB9_PARAMS)


class OptionalSSLUnavailable(RuntimeError):
    """Raised only when the optional SSL dependency or checkpoint is unavailable."""


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cohort_hash(cohort: pd.DataFrame) -> str:
    value = cohort[["source_id", "country", "is_eog_flare"]].sort_values(
        "source_id"
    ).to_csv(index=False)
    return hashlib.sha256(value.encode()).hexdigest()


def _fill_rgb(image: np.ndarray) -> np.ndarray:
    """Return finite Sentinel-2 RGB reflectance in B4, B3, B2 order."""
    rgb = image[[2, 1, 0]].astype("float32", copy=True)
    for band in range(3):
        finite = np.isfinite(rgb[band])
        fill = float(np.median(rgb[band, finite])) if finite.any() else 0.0
        rgb[band, ~finite] = fill
    return np.clip(rgb, -0.05, 1.0)


def extract_ssl_embeddings(
    input_root: str | Path,
    cohort: pd.DataFrame,
    cache_path: str | Path,
    batch_size: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """Extract frozen full-chip and 1 km center embeddings on GPU when present."""
    import torch

    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".json")
    expected = {
        "cohort_sha256": _cohort_hash(cohort),
        "weight_name": SSL_WEIGHT_NAME,
        "weight_url": SSL_WEIGHT_URL,
        "views": ["full_2km", "center_1km"],
        "bands": ["B4", "B3", "B2"],
    }
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            frame = pd.read_parquet(cache_path)
            if frame.source_id.is_unique and set(frame.source_id) == set(cohort.source_id):
                return frame, manifest

    try:
        from torchgeo.models import ResNet18_Weights, resnet18
    except (ImportError, OSError) as error:
        raise OptionalSSLUnavailable(
            "Install torchgeo==0.7.1 and timm before extracting SSL embeddings"
        ) from error

    weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
    weight_bands = [
        str(band).upper().split(".")[-1].replace("B0", "B")
        for band in weights.meta.get("bands", [])
    ]
    if weight_bands != ["B4", "B3", "B2"]:
        raise ValueError(f"Unexpected pretrained band order: {weights.meta}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = resnet18(weights=weights)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise OptionalSSLUnavailable(
            "The optional TorchGeo checkpoint could not be downloaded"
        ) from error
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = weights.transforms()
    with torch.inference_mode():
        probe = transform(torch.zeros((3, 200, 200), dtype=torch.float32))
        probe_embedding = model(probe.unsqueeze(0).to(device))
    if tuple(probe_embedding.shape) != (1, 512):
        raise ValueError(
            f"TorchGeo encoder smoke test returned {tuple(probe_embedding.shape)}"
        )
    del probe, probe_embedding

    sample_path = find_unique(Path(input_root), "pilot_sources.csv")
    chip_root = sample_path.parent
    sample = pd.read_csv(sample_path, usecols=["source_id", "chip_id"])
    mapping = sample.set_index("source_id").chip_id
    if not set(cohort.source_id).issubset(mapping.index):
        raise ValueError("Cohort source is absent from the frozen chip manifest")

    rows = []
    tensors = []
    source_ids = []

    def flush() -> None:
        if not tensors:
            return
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embedding = model(batch)
            else:
                embedding = model(batch)
        values = embedding.float().cpu().numpy()
        for source_id, value in zip(source_ids, values):
            rows.append((source_id, value))
        tensors.clear()
        source_ids.clear()

    for source_id in cohort.source_id:
        chip_id = mapping.loc[source_id]
        path = chip_root / f"{chip_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing successful chip: {path}")
        with np.load(path, allow_pickle=False) as data:
            rgb = _fill_rgb(data["reflectance"])
        full = transform(torch.from_numpy(rgb * 10_000.0))
        center = transform(torch.from_numpy(rgb[:, 50:150, 50:150] * 10_000.0))
        tensors.extend([full, center])
        source_ids.extend([f"{source_id}|full", f"{source_id}|center"])
        if len(tensors) >= 2 * batch_size:
            flush()
    flush()

    by_key = {key: value for key, value in rows}
    matrix = []
    for source_id in cohort.source_id:
        matrix.append(np.concatenate([
            by_key[f"{source_id}|full"], by_key[f"{source_id}|center"]
        ]).astype("float32"))
    matrix = np.stack(matrix)
    if not np.isfinite(matrix).all() or matrix.shape[1] != 1024:
        raise ValueError(f"Invalid SSL embedding matrix: {matrix.shape}")
    columns = [f"ssl_{index:04d}" for index in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "source_id", cohort.source_id.to_numpy())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    manifest = {
        **expected,
        "n_sources": len(frame),
        "n_features": len(columns),
        "device": str(device),
        "torch": torch.__version__,
        "torchgeo": importlib.metadata.version("torchgeo"),
        "embedding_sha256": file_hash(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def _aligned_extra(
    cohort: pd.DataFrame,
    descriptors: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    descriptor_columns = [column for column in descriptors if column.startswith("ts_")]
    if not descriptor_columns:
        raise ValueError("No explicit ts_* temporal descriptors were supplied")
    forbidden = FORBIDDEN_FEATURE_COLUMNS & set(descriptor_columns)
    eog_columns = [column for column in descriptor_columns if "eog" in column.lower()]
    if forbidden or eog_columns:
        raise ValueError(
            f"Forbidden temporal descriptor: {sorted(forbidden | set(eog_columns))}"
        )
    descriptor_frame = cohort[["source_id"]].merge(
        descriptors[["source_id"] + descriptor_columns],
        on="source_id", how="left", validate="one_to_one",
    )
    if descriptor_frame[descriptor_columns].isna().all(axis=1).any():
        raise ValueError("At least one cohort row has no temporal descriptors")
    embedding_columns = [column for column in embeddings if column.startswith("ssl_")]
    embedding_frame = cohort[["source_id"]].merge(
        embeddings, on="source_id", how="left", validate="one_to_one"
    )
    embedding_matrix = embedding_frame[embedding_columns].to_numpy(dtype="float32")
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("Non-finite frozen image embedding")
    return descriptor_frame, embedding_matrix, descriptor_columns


def make_matrices(
    cohort: pd.DataFrame,
    descriptor_frame: pd.DataFrame,
    descriptor_columns: list[str],
    embedding_matrix: np.ndarray,
    fit_index: np.ndarray,
    test_index: np.ndarray,
    variant: str,
    pca_components: int,
):
    config = VARIANTS[variant]
    names = list(BASE_COLUMNS)
    fit_parts = [cohort.iloc[fit_index][BASE_COLUMNS].to_numpy(dtype="float32")]
    test_parts = [cohort.iloc[test_index][BASE_COLUMNS].to_numpy(dtype="float32")]
    if config["temporal"]:
        fit_parts.append(
            descriptor_frame.iloc[fit_index][descriptor_columns].to_numpy(dtype="float32")
        )
        test_parts.append(
            descriptor_frame.iloc[test_index][descriptor_columns].to_numpy(dtype="float32")
        )
        names.extend(descriptor_columns)
    transformer = None
    if config["ssl"]:
        components = min(pca_components, len(fit_index) - 1, embedding_matrix.shape[1])
        if components < 2:
            raise ValueError("Insufficient training rows for fold-local SSL PCA")
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(embedding_matrix[fit_index])
        pca = PCA(n_components=components, whiten=True, random_state=0)
        fit_parts.append(pca.fit_transform(fit_scaled).astype("float32"))
        test_parts.append(
            pca.transform(scaler.transform(embedding_matrix[test_index])).astype("float32")
        )
        names.extend([f"ssl_pc_{index:02d}" for index in range(components)])
        transformer = {"scaler": scaler, "pca": pca}
    return np.concatenate(fit_parts, axis=1), np.concatenate(test_parts, axis=1), names, transformer


def train_model(x, y, seed, rounds):
    params = dict(PARAMS)
    params.update({key: seed for key in [
        "seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"
    ]})
    return lgb.train(
        params,
        lgb.Dataset(x, label=y, free_raw_data=True),
        num_boost_round=rounds,
    )


def structured_inner_oof(
    train: pd.DataFrame,
    descriptor_frame: pd.DataFrame,
    descriptor_columns: list[str],
    embedding_matrix: np.ndarray,
    variant: str,
    seed: int,
    rounds: int,
    pca_components: int,
) -> np.ndarray:
    score = np.empty(len(train), dtype="float64")
    for inner_index, country in enumerate(sorted(train.country.unique())):
        fit_index = np.flatnonzero(train.country.ne(country).to_numpy())
        test_index = np.flatnonzero(train.country.eq(country).to_numpy())
        x_fit, x_test, _, _ = make_matrices(
            train, descriptor_frame, descriptor_columns, embedding_matrix,
            fit_index, test_index, variant, pca_components,
        )
        model = train_model(
            x_fit,
            train.iloc[fit_index].is_eog_flare.to_numpy(dtype="int8"),
            seed + inner_index * 100,
            rounds,
        )
        score[test_index] = model.predict(x_test)
    return score


def structured_outer_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_descriptors: pd.DataFrame,
    test_descriptors: pd.DataFrame,
    descriptor_columns: list[str],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    variant: str,
    seeds: list[int],
    rounds: int,
    pca_components: int,
) -> np.ndarray:
    combined = pd.concat([train, test], ignore_index=True)
    combined_descriptors = pd.concat(
        [train_descriptors, test_descriptors], ignore_index=True
    )
    combined_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    fit_index = np.arange(len(train))
    test_index = np.arange(len(train), len(combined))
    x_fit, x_test, _, _ = make_matrices(
        combined, combined_descriptors, descriptor_columns, combined_embeddings,
        fit_index, test_index, variant, pca_components,
    )
    score = np.zeros(len(test), dtype="float64")
    y = train.is_eog_flare.to_numpy(dtype="int8")
    for model_seed in seeds:
        score += train_model(x_fit, y, model_seed, rounds).predict(x_test) / len(seeds)
    return score


def _lookup_score(mapping, country: str, source_ids: pd.Series) -> np.ndarray:
    values = mapping[country]
    if isinstance(values, pd.DataFrame):
        values = values.set_index("source_id").score
    elif isinstance(values, dict):
        values = pd.Series(values)
    if not isinstance(values, pd.Series):
        raise TypeError(f"Unexpected TCN score mapping for {country}: {type(values)}")
    missing = set(source_ids) - set(values.index)
    if missing:
        raise ValueError(f"Missing {len(missing)} TCN scores for {country}")
    score = values.loc[source_ids].to_numpy(dtype="float64")
    if not np.isfinite(score).all():
        raise ValueError("Non-finite TCN score")
    return score


def _lookup_flat_score(mapping, source_ids: pd.Series) -> np.ndarray:
    if isinstance(mapping, pd.DataFrame):
        values = mapping.set_index("source_id").score
    elif isinstance(mapping, dict):
        values = pd.Series(mapping, dtype="float64")
    elif isinstance(mapping, pd.Series):
        values = mapping
    else:
        raise TypeError(f"Unexpected flat TCN score mapping: {type(mapping)}")
    missing = set(source_ids) - set(values.index)
    if missing:
        raise ValueError(f"Missing {len(missing)} nested TCN scores")
    score = values.loc[source_ids].to_numpy(dtype="float64")
    if not np.isfinite(score).all():
        raise ValueError("Non-finite nested TCN score")
    return score


def _pair_mapping(pair_scores, first: str, second: str):
    key = tuple(sorted((first, second)))
    if key not in pair_scores:
        raise KeyError(f"Missing nested TCN pair scores for {key}")
    return pair_scores[key]


def _country_ap_values(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    values = {}
    for country, part in frame.groupby("country"):
        values[country] = macro_ap(part.reset_index(drop=True), score[part.index])
    return values


def country_percentile(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Convert scores to within-country percentile ranks without labels."""
    if len(frame) != len(score):
        raise ValueError("Frame and score lengths differ")
    ranked = np.empty(len(frame), dtype="float64")
    for _, part in frame.groupby("country", sort=False):
        indices = part.index.to_numpy(dtype="int64")
        ranked[indices] = pd.Series(score[indices]).rank(
            method="average", pct=True
        ).to_numpy(dtype="float64")
    return ranked


def candidate_scores(
    frame: pd.DataFrame,
    structured_scores: dict[str, np.ndarray],
    tcn_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build fixed candidates, including conservative rank-level blends."""
    candidates = {key: np.asarray(value) for key, value in structured_scores.items()}
    candidates["tcn_only"] = np.asarray(tcn_score)
    tcn_rank = country_percentile(frame, tcn_score)
    for variant, score in structured_scores.items():
        key = f"rank_blend_{variant}"
        candidates[key] = (
            TCN_STRUCTURED_WEIGHT * country_percentile(frame, score)
            + (1.0 - TCN_STRUCTURED_WEIGHT) * tcn_rank
        )
    return candidates


def choose_inner_candidate(
    frame: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    minimum_improved_countries: int,
    maximum_worst_drop: float,
) -> tuple[str, np.ndarray, list[dict]]:
    """Apply a predeclared baseline guard using training countries only."""
    baseline = "nb9_baseline"
    baseline_country = _country_ap_values(frame, candidates[baseline])
    baseline_ap = float(np.mean(list(baseline_country.values())))
    rows = []
    for branch, score in candidates.items():
        by_country = _country_ap_values(frame, score)
        deltas = np.asarray([
            by_country[country] - baseline_country[country]
            for country in sorted(by_country)
        ])
        rows.append({
            "branch": branch,
            "macro_ap": float(np.mean(list(by_country.values()))),
            "ap_gain_vs_baseline": float(np.mean(deltas)),
            "improved_countries": int((deltas > 0).sum()),
            "worst_ap_delta": float(deltas.min()),
            "eligible": bool(
                branch == baseline
                or (
                    np.mean(list(by_country.values())) >= baseline_ap + 0.005
                    and (deltas > 0).sum() >= minimum_improved_countries
                    and deltas.min() >= -maximum_worst_drop
                )
            ),
        })
    table = pd.DataFrame(rows)
    selected = table.loc[table.eligible].sort_values(
        ["macro_ap", "branch"], ascending=[False, True]
    ).iloc[0]
    name = str(selected.branch)
    return name, candidates[name], table.sort_values(
        ["eligible", "macro_ap"], ascending=False
    ).to_dict("records")


def _global_branch_selection(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    summary = metrics.groupby("branch").agg(
        macro_f1=("f1", "mean"),
        macro_pr_auc=("pr_auc", "mean"),
        macro_roc_auc=("roc_auc", "mean"),
        worst_country_pr_auc=("pr_auc", "min"),
    ).reset_index()
    baseline = metrics.loc[metrics.branch.eq("nb9_baseline")].set_index("country")
    checks = []
    baseline_ap = float(baseline.pr_auc.mean())
    for branch in summary.branch:
        current = metrics.loc[metrics.branch.eq(branch)].set_index("country")
        delta = current.pr_auc - baseline.pr_auc
        checks.append({
            "branch": branch,
            "ap_gain_vs_baseline": float(current.pr_auc.mean() - baseline_ap),
            "improved_countries": int((delta > 0).sum()),
            "worst_ap_delta": float(delta.min()),
            "eligible": bool(
                branch == "nb9_baseline"
                or (
                    current.pr_auc.mean() >= baseline_ap + 0.005
                    and (delta > 0).sum() >= 4
                    and delta.min() >= -0.03
                )
            ),
        })
    checks = pd.DataFrame(checks)
    summary = summary.merge(checks, on="branch", validate="one_to_one")
    eligible = summary.loc[summary.eligible].sort_values(
        ["macro_pr_auc", "macro_f1"], ascending=False
    )
    return str(eligible.iloc[0].branch), summary.sort_values(
        ["eligible", "macro_pr_auc"], ascending=False
    )


def load_stage1(input_root: str | Path) -> tuple[pd.DataFrame, dict, list[Path]]:
    """Load the frozen NB8 population branch without retraining it."""
    root = Path(input_root)
    manifest_path = find_unique(root, "05e_manifest.json")
    prediction_path = find_unique(root, "05e_loco_predictions.parquet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != "05e-domain-revamp-v1"
        or manifest.get("status") != "complete"
        or manifest.get("holdout_country") != HOLDOUT
        or manifest.get("holdout_loaded") is not False
    ):
        raise ValueError("NB8 population manifest is incompatible or contaminated")
    columns = [
        "source_id", "country", "block_id", "is_eog_flare",
        "eog_flare_id", "score", "model_variant",
    ]
    frame = pd.read_parquet(prediction_path, columns=columns)
    frame["source_id"] = frame.source_id.astype(str)
    frame["country"] = frame.country.astype(str)
    if (
        len(frame) != int(manifest.get("n_sources", -1))
        or not frame.source_id.is_unique
        or set(frame.country) != set(COUNTRIES)
        or HOLDOUT in set(frame.country)
    ):
        raise ValueError("NB8 population predictions are incomplete or invalid")
    model_paths = sorted(manifest_path.parent.parent.rglob("05e_final_model_*.txt"))
    if len(model_paths) != 3:
        raise FileNotFoundError(f"Expected three frozen NB8 models; found {model_paths}")
    metadata = {
        "manifest_path": str(manifest_path),
        "prediction_path": str(prediction_path),
        "manifest_sha256": file_hash(manifest_path),
        "prediction_sha256": file_hash(prediction_path),
        "protocol": manifest["protocol"],
        "n_sources": len(frame),
        "n_positive": int(frame.is_eog_flare.sum()),
        "original_macro_pr_auc": manifest.get("macro_loco_pr_auc"),
        "original_macro_f1": manifest.get("macro_loco_f1"),
        "selected_features": manifest.get("selected_features"),
        "selected_variant": manifest.get("selected_variant"),
        "final_threshold": manifest.get("final_threshold"),
        "input_sha256": manifest.get("input_sha256", {}),
    }
    return frame, metadata, model_paths


def evaluate_stage1(
    nb8: pd.DataFrame,
    temporal_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, dict]:
    """Evaluate the population TCN and a fixed rank blend at real prevalence."""
    required_temporal = {
        "source_id", "country", "is_eog_flare", "temporal_score"
    }
    if not required_temporal.issubset(temporal_frame):
        raise ValueError("Full-population temporal scores are required for Stage A")
    temporal = temporal_frame[list(required_temporal)].copy()
    frame = nb8.merge(
        temporal,
        on=["source_id", "country"],
        how="inner",
        validate="one_to_one",
        suffixes=("_nb8", "_tcn"),
    )
    if len(frame) != len(nb8):
        raise ValueError("TCN scores do not cover the complete NB8 population")
    if not frame.is_eog_flare_nb8.astype("int8").equals(
        frame.is_eog_flare_tcn.astype("int8")
    ):
        raise ValueError("NB8 and TCN population labels disagree")
    frame = frame.rename(columns={
        "is_eog_flare_nb8": "is_eog_flare",
        "score": "nb8_score",
    }).drop(columns=["is_eog_flare_tcn"])
    frame["nb8_rank"] = frame.groupby("country", observed=True).nb8_score.rank(
        method="average", pct=True
    )
    frame["tcn_rank"] = frame.groupby("country", observed=True).temporal_score.rank(
        method="average", pct=True
    )
    frame["rank_blend_score"] = (
        TCN_STRUCTURED_WEIGHT * frame.nb8_rank
        + (1.0 - TCN_STRUCTURED_WEIGHT) * frame.tcn_rank
    )
    branches = {
        "nb8_population": "nb8_score",
        "tcn_population": "temporal_score",
        "fixed_rank_blend": "rank_blend_score",
    }
    metric_rows = []
    gate_rows = []
    for country, part in frame.groupby("country", observed=True):
        y = part.is_eog_flare.to_numpy(dtype="int8")
        total_sites = part.loc[part.is_eog_flare.eq(1), "eog_flare_id"].nunique()
        for branch, column in branches.items():
            score = part[column].to_numpy(dtype="float64")
            metric_rows.append({
                "country": country,
                "branch": branch,
                "n": len(part),
                "n_positive": int(y.sum()),
                "pr_auc": average_precision_score(y, score),
                "roc_auc": roc_auc_score(y, score),
            })
            order = np.argsort(score)[::-1]
            for fraction in STAGE1_GATE_FRACTIONS:
                count = max(1, int(np.ceil(fraction * len(part))))
                selected = part.iloc[order[:count]]
                true_positive = int(selected.is_eog_flare.sum())
                hit_sites = selected.loc[
                    selected.is_eog_flare.eq(1), "eog_flare_id"
                ].nunique()
                gate_rows.append({
                    "country": country,
                    "branch": branch,
                    "top_fraction": fraction,
                    "candidates": count,
                    "precision_at_gate": true_positive / count,
                    "source_recall_at_gate": true_positive / max(int(y.sum()), 1),
                    "site_recall_at_gate": hit_sites / max(total_sites, 1),
                })
    metrics = pd.DataFrame(metric_rows)
    gates = pd.DataFrame(gate_rows)
    baseline = metrics.loc[metrics.branch.eq("nb8_population")].set_index("country")
    blend = metrics.loc[metrics.branch.eq("fixed_rank_blend")].set_index("country")
    delta = blend.pr_auc - baseline.pr_auc
    guard = {
        "macro_ap_gain": float(delta.mean()),
        "improved_countries": int((delta > 0).sum()),
        "worst_country_ap_delta": float(delta.min()),
        "minimum_macro_ap_gain": 0.005,
        "minimum_improved_countries": 4,
        "maximum_worst_country_ap_drop": 0.03,
    }
    selected_branch = "fixed_rank_blend" if (
        guard["macro_ap_gain"] >= guard["minimum_macro_ap_gain"]
        and guard["improved_countries"] >= guard["minimum_improved_countries"]
        and guard["worst_country_ap_delta"]
        >= -guard["maximum_worst_country_ap_drop"]
    ) else "nb8_population"
    frame["selected_stage1_score"] = frame[
        branches[selected_branch]
    ]
    return frame, metrics, gates, selected_branch, guard


def validate_stage1_feature_hashes(stage1_metadata: dict, temporal_metadata: dict) -> None:
    """Require NB8 predictions and the current NB2 features to share a lineage."""
    for country in COUNTRIES:
        name = f"features_{country}_2022_2024.parquet"
        nb8_hash = stage1_metadata.get("input_sha256", {}).get(name)
        current_hash = temporal_metadata.get("input_sha256", {}).get(name)
        if not nb8_hash or nb8_hash != current_hash:
            raise ValueError(f"NB8 and NB2 use different feature data for {name}")


def run(
    input_root="/kaggle/input",
    output_root="/kaggle/working/nb11_multimodal",
    rounds=300,
    tcn_epochs=8,
    negative_ratio=10,
    pu_bags=2,
    tcn_batch_size=512,
    pca_components=16,
    seed=131,
    include_ssl=True,
):
    started = time.time()
    input_root = Path(input_root)
    output_root = Path(output_root)
    outputs = output_root / "outputs"
    cache = output_root / "cache"
    models = output_root / "models"
    for directory in [outputs, cache, models]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs / "11_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh output directory for a complete run")
    shutil.copy2(Path(__file__), output_root / Path(__file__).name)
    for dependency in ["kg_08_fusion.py", "kg_10_temporal_tcn.py"]:
        source = Path(__file__).with_name(dependency)
        if not source.exists():
            raise FileNotFoundError(f"Missing bundled implementation: {source}")
        shutil.copy2(source, output_root / dependency)

    cohort, nb9_metadata = load_cohort(input_root)
    if HOLDOUT in set(cohort.country):
        raise ValueError("India is forbidden during multimodal model selection")
    nb8_predictions, stage1_metadata, stage1_model_paths = load_stage1(input_root)
    available_variants = list(VARIANTS)
    if include_ssl:
        try:
            embeddings, ssl_metadata = extract_ssl_embeddings(
                input_root, cohort, cache / "11_ssl_embeddings.parquet"
            )
            ssl_metadata["status"] = "available"
        except OptionalSSLUnavailable as error:
            embeddings = pd.DataFrame({"source_id": cohort.source_id})
            available_variants = [
                variant for variant in available_variants
                if not VARIANTS[variant]["ssl"]
            ]
            ssl_metadata = {
                "status": "disabled",
                "reason": f"{type(error).__name__}: {error}",
                "weight_name": SSL_WEIGHT_NAME,
                "weight_url": SSL_WEIGHT_URL,
            }
            print(f"Optional SSL branch disabled: {ssl_metadata['reason']}", flush=True)
    else:
        embeddings = pd.DataFrame({"source_id": cohort.source_id})
        available_variants = [
            variant for variant in available_variants
            if not VARIANTS[variant]["ssl"]
        ]
        ssl_metadata = {
            "status": "disabled",
            "reason": "disabled by configuration",
            "weight_name": SSL_WEIGHT_NAME,
            "weight_url": SSL_WEIGHT_URL,
        }
    temporal_meta, sequences, descriptors, temporal_metadata = prepare_temporal_data(
        input_root,
        cohort,
        negative_ratio=negative_ratio,
        seed=seed,
        include_population=True,
    )
    validate_stage1_feature_hashes(stage1_metadata, temporal_metadata)
    descriptor_frame, embedding_matrix, descriptor_columns = _aligned_extra(
        cohort, descriptors, embeddings
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    temporal_scores = precompute_nested_scores(
        temporal_meta, sequences, cohort,
        epochs=tcn_epochs,
        seed=seed,
        device=device,
        batch_size=tcn_batch_size,
        pu_bags=pu_bags,
        unlabeled_keep_fraction=0.8,
    )
    single_scores = temporal_scores["single_scores"]
    pair_scores = temporal_scores["pair_scores"]
    stage1_predictions, stage1_metrics, stage1_gates, stage1_selected, stage1_guard = (
        evaluate_stage1(
            nb8_predictions,
            temporal_scores["single_score_frame"],
        )
    )

    metric_rows = []
    prediction_parts = []
    selection_rows = []
    for outer_index, held_out in enumerate(COUNTRIES):
        train_mask = cohort.country.ne(held_out).to_numpy()
        test_mask = ~train_mask
        train = cohort.loc[train_mask].reset_index(drop=True)
        test = cohort.loc[test_mask].reset_index(drop=True)
        train_descriptors = descriptor_frame.loc[train_mask].reset_index(drop=True)
        test_descriptors = descriptor_frame.loc[test_mask].reset_index(drop=True)
        train_embeddings = embedding_matrix[train_mask]
        test_embeddings = embedding_matrix[test_mask]

        inner_scores = {}
        outer_scores = {}
        for variant in available_variants:
            inner_scores[variant] = structured_inner_oof(
                train, train_descriptors, descriptor_columns, train_embeddings,
                variant, seed + outer_index * 1000, rounds, pca_components,
            )
            threshold, inner_f1 = macro_f1_threshold(train, inner_scores[variant])
            outer_scores[variant] = structured_outer_predict(
                train, test, train_descriptors, test_descriptors,
                descriptor_columns, train_embeddings, test_embeddings,
                variant,
                [seed + 20_000 + outer_index * 1000 + offset for offset in (0, 101, 202)],
                rounds, pca_components,
            )
            metric_rows.append(metric_row(
                test, outer_scores[variant], threshold, variant, held_out
            ))
            selection_rows.append({
                "held_out_country": held_out,
                "branch": variant,
                "inner_macro_ap": macro_ap(train, inner_scores[variant]),
                "inner_macro_f1": inner_f1,
                "threshold": threshold,
            })

        tcn_inner = np.empty(len(train), dtype="float64")
        for inner_country in sorted(train.country.unique()):
            mask = train.country.eq(inner_country)
            mapping = _pair_mapping(pair_scores, held_out, inner_country)
            tcn_inner[mask] = _lookup_flat_score(
                mapping, train.loc[mask, "source_id"]
            )
        tcn_outer = _lookup_score(single_scores, held_out, test.source_id)
        inner_candidates = candidate_scores(train, inner_scores, tcn_inner)
        outer_candidates = candidate_scores(test, outer_scores, tcn_outer)
        for branch in sorted(set(inner_candidates) - set(VARIANTS)):
            threshold, inner_f1 = macro_f1_threshold(train, inner_candidates[branch])
            metric_rows.append(metric_row(
                test, outer_candidates[branch], threshold, branch, held_out
            ))
            selection_rows.append({
                "held_out_country": held_out,
                "branch": branch,
                "inner_macro_ap": macro_ap(train, inner_candidates[branch]),
                "inner_macro_f1": inner_f1,
                "threshold": threshold,
            })

        champion, champion_inner, diagnostics = choose_inner_candidate(
            train,
            inner_candidates,
            minimum_improved_countries=3,
            maximum_worst_drop=0.04,
        )
        champion_threshold, champion_inner_f1 = macro_f1_threshold(
            train, champion_inner
        )
        champion_outer = outer_candidates[champion]
        metric_rows.append(metric_row(
            test, champion_outer, champion_threshold, "nested_champion", held_out
        ))
        selection_rows.append({
            "held_out_country": held_out,
            "branch": "nested_champion",
            "selected_candidate": champion,
            "inner_macro_ap": macro_ap(train, champion_inner),
            "inner_macro_f1": champion_inner_f1,
            "threshold": champion_threshold,
            "candidate_diagnostics": json.dumps(diagnostics),
        })

        part = test[[
            "source_id", "country", "is_eog_flare", "eog_flare_id", "block_id"
        ]].copy()
        for branch, score in outer_candidates.items():
            part[f"score_{branch}"] = score
        part["score_nested_champion"] = champion_outer
        part["nested_selected_candidate"] = champion
        prediction_parts.append(part)
        print(f"Completed multimodal holdout {held_out}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    summary = metrics.groupby("branch").agg(
        macro_f1=("f1", "mean"),
        macro_pr_auc=("pr_auc", "mean"),
        macro_roc_auc=("roc_auc", "mean"),
        worst_country_pr_auc=("pr_auc", "min"),
    ).reset_index().sort_values(
        ["macro_pr_auc", "macro_f1"], ascending=False
    )
    ordered = predictions.set_index("source_id").loc[cohort.source_id]
    tcn_oof = ordered.score_tcn_only.to_numpy()
    structured_oof = {
        variant: ordered[f"score_{variant}"].to_numpy()
        for variant in available_variants
    }
    global_candidates = candidate_scores(cohort, structured_oof, tcn_oof)
    selected, final_score, final_diagnostics = choose_inner_candidate(
        cohort,
        global_candidates,
        minimum_improved_countries=4,
        maximum_worst_drop=0.03,
    )
    if selected == "tcn_only":
        final_weight = 0.0
        model_variant = None
    elif selected.startswith("rank_blend_"):
        model_variant = selected.removeprefix("rank_blend_")
        final_weight = TCN_STRUCTURED_WEIGHT
    else:
        final_weight = None
        model_variant = selected
    final_threshold, _ = macro_f1_threshold(cohort, final_score)
    final_score_lookup = dict(zip(cohort.source_id, final_score))
    predictions["score_final_selected"] = predictions.source_id.map(final_score_lookup)

    model_artifacts = []
    importance = pd.DataFrame()
    feature_names: list[str] = []
    if model_variant is not None:
        all_index = np.arange(len(cohort))
        x, _, feature_names, transformer = make_matrices(
            cohort, descriptor_frame, descriptor_columns, embedding_matrix,
            all_index, all_index, model_variant, pca_components,
        )
        gain = np.zeros(len(feature_names))
        for model_index, final_seed in enumerate(
            [seed + 30_000 + offset for offset in (0, 101, 202)]
        ):
            model = train_model(
                x, cohort.is_eog_flare.to_numpy(dtype="int8"), final_seed, rounds
            )
            path = models / f"{model_variant}_{model_index}.txt"
            model.save_model(str(path))
            model_artifacts.append(path.name)
            gain += model.feature_importance("gain") / 3
        importance = pd.DataFrame({"feature": feature_names, "mean_gain": gain})
        importance.sort_values("mean_gain", ascending=False).to_csv(
            outputs / "11_final_feature_importance.csv", index=False
        )
        if transformer is not None:
            joblib.dump(transformer, models / "ssl_transformer.joblib")
            model_artifacts.append("ssl_transformer.joblib")

    tcn_artifact = fit_final_tcn(
        temporal_meta, sequences, models,
        epochs=tcn_epochs,
        seed=seed + 40_000,
        device=device,
        batch_size=tcn_batch_size,
        pu_bags=pu_bags,
        unlabeled_keep_fraction=0.8,
    )
    tcn_artifact["artifact_path"] = Path(tcn_artifact["artifact_path"]).name
    tcn_artifact["artifact_paths"] = [
        Path(path).name for path in tcn_artifact["artifact_paths"]
    ]
    tcn_artifact["config_path"] = Path(tcn_artifact["config_path"]).name
    stage1_model_dir = models / "stage1_nb8"
    stage1_model_dir.mkdir(parents=True, exist_ok=True)
    copied_stage1_models = []
    for source_path in stage1_model_paths:
        target = stage1_model_dir / source_path.name
        shutil.copy2(source_path, target)
        copied_stage1_models.append(str(target.relative_to(models)))
    metrics.to_csv(outputs / "11_country_metrics.csv", index=False)
    summary.to_csv(outputs / "11_branch_summary.csv", index=False)
    selections.to_csv(outputs / "11_inner_selection.csv", index=False)
    predictions.to_parquet(outputs / "11_loco_predictions.parquet", index=False)
    stage1_predictions.to_parquet(
        outputs / "11_stage1_population_predictions.parquet", index=False
    )
    stage1_metrics.to_csv(outputs / "11_stage1_country_metrics.csv", index=False)
    stage1_gates.to_csv(outputs / "11_stage1_gate_metrics.csv", index=False)
    descriptor_keep = temporal_meta.train_selected | temporal_meta.is_cohort
    descriptors.loc[descriptor_keep].to_parquet(
        cache / "11_temporal_descriptors.parquet", index=False
    )
    pd.DataFrame(temporal_scores["diagnostics"]).to_csv(
        outputs / "11_tcn_training_diagnostics.csv", index=False
    )
    selected_schema = {
        "stage1_branch": stage1_selected,
        "stage1_rank_weights": {
            "nb8": TCN_STRUCTURED_WEIGHT,
            "tcn": 1.0 - TCN_STRUCTURED_WEIGHT,
        } if stage1_selected == "fixed_rank_blend" else None,
        "stage2_candidate": selected,
        "stage2_structured_variant": model_variant,
        "stage2_structured_features": feature_names if model_variant is not None else [],
        "stage2_structured_weight": final_weight,
        "stage2_tcn_weight": (
            1.0 - final_weight if final_weight is not None else None
        ),
        "stage2_pilot_threshold": final_threshold,
        "threshold_is_deployment_calibrated": False,
    }
    (outputs / "11_selected_schema.json").write_text(
        json.dumps(selected_schema, indent=2), encoding="utf-8"
    )

    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "holdout_country": HOLDOUT,
        "holdout_loaded": False,
        "selection_population": "six foreign countries only",
        "stage1_selected_branch": stage1_selected,
        "stage1_guard": stage1_guard,
        "stage1": stage1_metadata,
        "stage1_model_artifacts": copied_stage1_models,
        "stage2_selected_candidate": selected,
        "stage2_nested_outer_branch": "nested_champion",
        "stage2_available_variants": available_variants,
        "stage2_final_selection_diagnostics": final_diagnostics,
        "final_structured_variant": model_variant,
        "final_structured_weight": final_weight,
        "stage2_pilot_threshold": final_threshold,
        "threshold_is_deployment_calibrated": False,
        "guard": {
            "minimum_macro_ap_gain": 0.005,
            "minimum_improved_countries": 4,
            "maximum_worst_country_ap_drop": 0.03,
        },
        "model_artifacts": model_artifacts,
        "tcn_artifact": tcn_artifact,
        "params": PARAMS,
        "rounds": rounds,
        "tcn_epochs": tcn_epochs,
        "negative_ratio": negative_ratio,
        "pu_bags": pu_bags,
        "tcn_batch_size": tcn_batch_size,
        "pca_components": pca_components,
        "seed": seed,
        "ssl": ssl_metadata,
        "temporal": temporal_metadata,
        "nb9": nb9_metadata,
        "limitations": [
            "Stage B imagery is an enriched foreign pilot and does not estimate population precision.",
            "India imagery is unavailable, so India inference remains Stage A only.",
            "Unmatched sources are positive-unlabeled examples, not verified negatives.",
            "The Stage B pilot threshold is not a deployment threshold.",
        ],
        "elapsed_minutes": (time.time() - started) / 60,
        "python": platform.python_version(),
        "versions": {
            package: package_version(package)
            for package in [
                "numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow",
                "torch", "torchgeo", "timm",
            ]
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, metrics, predictions


if __name__ == "__main__":
    run()
