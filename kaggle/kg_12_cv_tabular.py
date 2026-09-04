"""Leakage-safe Sentinel-2 computer vision and FIRMS tabular fusion.

This stage treats imagery as a residual reranker over the proven NB9 model.
The image cohort is enriched for EOG matches, so all metrics are branch
comparisons on that cohort and are not population precision estimates.
India is never loaded, no target-derived NASA type field is used, and every
learned transformation is fitted inside the current country training fold.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from kg_08_fusion import (
    BRANCHES as NB9_BRANCHES,
    COUNTRIES,
    HOLDOUT,
    THERMAL_COLS,
    file_hash,
    find_unique,
    fit_predict,
    inner_country_oof,
    load_cohort,
    macro_ap,
    macro_f1_threshold,
    metric_row,
    train_model,
)


PROTOCOL = "12-cv-tabular-nested-loco-v1"
CHECKPOINT_NAME = "resnet18_sentinel2_rgb_moco-e3a335e3.pth"
CHECKPOINT_URL = (
    "https://hf.co/torchgeo/resnet18_sentinel2_rgb_moco/resolve/"
    "e1c032e7785fd0625224cdb6699aa138bb304eec/"
    f"{CHECKPOINT_NAME}"
)
CHECKPOINT_SHA256 = (
    "e3a335e38d1d189ad3b0eba4be4004a9c52c5e846317b6737ac9f0fac57e1ac8"
)
TABULAR_COLUMNS = list(NB9_BRANCHES["early_fusion"])
ALPHAS = (0.0, 0.15, 0.30, 0.50, 0.70)
FORBIDDEN_FEATURES = {
    "country", "latitude", "longitude", "lat", "lon", "type",
    "eog_dist_m", "eog_flare_id", "is_eog_flare", "block_id",
}
SELECTION_GUARD = {
    "minimum_macro_ap_gain": 0.01,
    "minimum_improved_countries": 4,
    "minimum_worst_country_delta": -0.02,
    "minimum_indonesia_delta": -0.02,
    "minimum_bootstrap_lower_95": -0.01,
    "minimum_macro_f1_delta": -0.01,
}
INNER_SELECTION_GUARD = {
    "minimum_macro_ap_gain": 0.005,
    "minimum_improved_countries": 3,
    "minimum_worst_country_delta": -0.03,
    "minimum_macro_f1_delta": -0.015,
}


def _cohort_hash(cohort: pd.DataFrame) -> str:
    payload = cohort[["source_id", "country", "is_eog_flare"]].sort_values(
        "source_id"
    ).to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cohort_chip_inventory(
    input_root: str | Path,
    cohort: pd.DataFrame,
) -> tuple[dict[object, Path], str]:
    """Resolve and hash the exact QA chip inputs used by both CV extractors."""
    sample_path = find_unique(Path(input_root), "pilot_sources.csv")
    sample = pd.read_csv(sample_path, usecols=["source_id", "chip_id"])
    if not sample.source_id.is_unique or not sample.chip_id.is_unique:
        raise ValueError("Pilot source or chip IDs are not unique")
    mapping = sample.set_index("source_id").chip_id
    missing_ids = set(cohort.source_id) - set(mapping.index)
    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} cohort sources in chip manifest")
    chip_paths = {
        source_id: sample_path.parent / f"{mapping.loc[source_id]}.npz"
        for source_id in cohort.source_id
    }
    missing_paths = [path for path in chip_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image chip: {missing_paths[0]}")
    digest = hashlib.sha256()
    for source_id in sorted(chip_paths, key=str):
        path = chip_paths[source_id]
        digest.update(str(source_id).encode())
        digest.update(file_hash(path).encode())
    return chip_paths, digest.hexdigest()


def _finite_rgb(reflectance: np.ndarray) -> np.ndarray:
    """Return finite Sentinel-2 B4, B3, B2 reflectance."""
    if reflectance.shape[0] != 6:
        raise ValueError(f"Expected six reflectance bands, got {reflectance.shape}")
    rgb = reflectance[[2, 1, 0]].astype("float32", copy=True)
    for band in range(3):
        finite = np.isfinite(rgb[band])
        fill = float(np.median(rgb[band, finite])) if finite.any() else 0.0
        rgb[band, ~finite] = fill
    if not np.isfinite(rgb).all():
        raise ValueError("RGB fill produced a non-finite value")
    median = float(np.median(rgb))
    if median < -1.0 or median > 2.0:
        raise ValueError(
            "Expected reflectance scaled near 0 to 1. The chip may still be in "
            "integer Sentinel-2 units."
        )
    return np.clip(rgb, -0.05, 1.5)


def _build_resnet18():
    """Build the exact ResNet-18 backbone used by the TorchGeo checkpoint."""
    import torch

    nn = torch.nn

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes: int, planes: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                inplanes, planes, 3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            if stride != 1 or inplanes != planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            else:
                self.downsample = None

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(BasicBlock, 64, 2)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
            layers = [block(self.inplanes, planes, stride)]
            self.inplanes = planes
            layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return self.global_pool(x).flatten(1)

    return ResNet18()


def _load_checkpoint(checkpoint_path: str | Path | None = None):
    import torch

    if checkpoint_path is None:
        try:
            state = torch.hub.load_state_dict_from_url(
                CHECKPOINT_URL,
                map_location="cpu",
                progress=True,
                check_hash=True,
                file_name=CHECKPOINT_NAME,
                weights_only=True,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            raise RuntimeError(
                "Could not download the frozen Sentinel-2 checkpoint. "
                "Enable Kaggle Internet and rerun."
            ) from error
        cached = Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_NAME
    else:
        cached = Path(checkpoint_path)
        if not cached.is_file():
            raise FileNotFoundError(f"Sentinel-2 checkpoint not found: {cached}")
        state = torch.load(cached, map_location="cpu", weights_only=True)
    observed_hash = file_hash(cached)
    if observed_hash.lower() != CHECKPOINT_SHA256:
        raise ValueError(
            "Sentinel-2 checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, observed {observed_hash}"
        )
    if not isinstance(state, dict) or "conv1.weight" not in state:
        raise ValueError("Unexpected Sentinel-2 checkpoint structure")
    if tuple(state["conv1.weight"].shape) != (64, 3, 7, 7):
        raise ValueError("Checkpoint is not the expected RGB ResNet-18")
    return state, cached


def _prepare_view(rgb: np.ndarray):
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(rgb).unsqueeze(0)
    tensor = functional.interpolate(
        tensor, size=(256, 256), mode="bilinear", align_corners=False
    )
    return tensor[0, :, 16:240, 16:240].contiguous()


def extract_cv_embeddings(
    input_root: str | Path,
    cohort: pd.DataFrame,
    cache_path: str | Path,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 32,
    tta: bool = True,
    chip_inventory: tuple[dict[object, Path], str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Extract rotation-averaged full-chip and center Sentinel-2 embeddings."""
    import torch

    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".json")
    if checkpoint_path is not None:
        supplied_checkpoint = Path(checkpoint_path)
        if not supplied_checkpoint.is_file():
            raise FileNotFoundError(
                f"Sentinel-2 checkpoint not found: {supplied_checkpoint}"
            )
        supplied_hash = file_hash(supplied_checkpoint)
        if supplied_hash.lower() != CHECKPOINT_SHA256:
            raise ValueError(
                "Supplied Sentinel-2 checkpoint does not match the pinned SHA-256"
            )
    if chip_inventory is None:
        chip_inventory = _cohort_chip_inventory(input_root, cohort)
    chip_paths, chip_inventory_sha256 = chip_inventory
    expected = {
        "protocol": PROTOCOL,
        "cohort_sha256": _cohort_hash(cohort),
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "views": ["full_2km", "center_1km"],
        "bands": ["B4", "B3", "B2"],
        "tta": bool(tta),
        "chip_inventory_sha256": chip_inventory_sha256,
    }
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            frame = pd.read_parquet(cache_path)
            cache_hash_ok = (
                manifest.get("embedding_sha256") == file_hash(cache_path)
            )
            if (
                cache_hash_ok
                and frame.source_id.is_unique
                and set(frame.source_id) == set(cohort.source_id)
            ):
                return frame, manifest

    state, cached_checkpoint = _load_checkpoint(checkpoint_path)
    model = _build_resnet18()
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        probe = model(torch.zeros((1, 3, 224, 224), device=device))
    if tuple(probe.shape) != (1, 512):
        raise ValueError(f"Unexpected encoder output: {tuple(probe.shape)}")

    augmentations = (
        (lambda x: x),
        (lambda x: torch.flip(x, dims=(-1,))),
        (lambda x: torch.flip(x, dims=(-2,))),
        (lambda x: torch.rot90(x, 1, dims=(-2, -1))),
    ) if tta else ((lambda x: x),)
    pending_tensors: list = []
    pending_keys: list[tuple[str, str]] = []
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}

    def flush() -> None:
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    values = model(batch)
            else:
                values = model(batch)
        matrix = values.float().cpu().numpy()
        for key, value in zip(pending_keys, matrix):
            sums[key] = sums.get(key, np.zeros(512, dtype="float64")) + value
            counts[key] = counts.get(key, 0) + 1
        pending_tensors.clear()
        pending_keys.clear()

    for source_id in cohort.source_id:
        chip_path = chip_paths[source_id]
        with np.load(chip_path, allow_pickle=False) as data:
            rgb = _finite_rgb(data["reflectance"])
        for view_name, view in (
            ("full", rgb),
            ("center", rgb[:, 50:150, 50:150]),
        ):
            prepared = _prepare_view(view)
            for augmentation in augmentations:
                pending_tensors.append(augmentation(prepared))
                pending_keys.append((source_id, view_name))
                if len(pending_tensors) >= batch_size:
                    flush()
    flush()

    rows = []
    for source_id in cohort.source_id:
        full_key = (source_id, "full")
        center_key = (source_id, "center")
        full = sums[full_key] / counts[full_key]
        center = sums[center_key] / counts[center_key]
        rows.append(np.concatenate([full, center]).astype("float32"))
    matrix = np.stack(rows)
    if matrix.shape != (len(cohort), 1024) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid frozen embedding matrix: {matrix.shape}")
    columns = [f"cv_{index:04d}" for index in range(matrix.shape[1])]
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
        "checkpoint_path": str(cached_checkpoint),
        "checkpoint_sha256": file_hash(cached_checkpoint),
        "embedding_sha256": file_hash(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    output = np.full_like(denominator, np.nan, dtype="float32")
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > 1e-6)
    output[valid] = (a[valid] - b[valid]) / denominator[valid]
    return output


def _masked_stat(values: np.ndarray, mask: np.ndarray, function: str) -> float:
    selected = values[mask & np.isfinite(values)]
    if not len(selected):
        return float("nan")
    if function == "median":
        return float(np.median(selected))
    if function == "std":
        return float(np.std(selected))
    if function == "mean":
        return float(np.mean(selected))
    if function == "p90":
        return float(np.quantile(selected, 0.9))
    raise KeyError(function)


def morphology_features_from_chip(
    reflectance: np.ndarray,
    valid_mask: np.ndarray,
    worldcover: np.ndarray,
    worldcover_valid: np.ndarray,
) -> dict[str, float]:
    """Extract spatial and multispectral morphology without coordinates."""
    image = reflectance.astype("float32", copy=True)
    valid = valid_mask.astype(bool) & np.isfinite(image).all(axis=0)
    for band in range(image.shape[0]):
        finite = valid & np.isfinite(image[band])
        fill = float(np.median(image[band, finite])) if finite.any() else 0.0
        image[band, ~finite] = fill
    blue, green, red, nir, swir16, swir22 = image
    arrays = {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir16": swir16,
        "swir22": swir22,
        "ndvi": _safe_index(nir, red),
        "ndbi": _safe_index(swir16, nir),
        "mndwi": _safe_index(green, swir16),
        "bsi": _safe_index(swir16 + red, nir + blue),
    }
    height, width = valid.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2)
    center = valid & (radius <= 25)
    ring = valid & (radius > 25) & (radius <= 50)
    full = valid
    features: dict[str, float] = {}
    for name, values in arrays.items():
        center_median = _masked_stat(values, center, "median")
        ring_median = _masked_stat(values, ring, "median")
        features[f"morph_{name}_center_minus_ring"] = center_median - ring_median
        features[f"morph_{name}_center_std"] = _masked_stat(values, center, "std")
    for name in ("red", "nir", "swir16", "ndvi", "ndbi", "bsi"):
        values = np.nan_to_num(arrays[name], nan=0.0)
        gy, gx = np.gradient(values)
        gradient = np.hypot(gx, gy)
        features[f"morph_{name}_gradient_mean"] = _masked_stat(
            gradient, full, "mean"
        )
        features[f"morph_{name}_gradient_p90"] = _masked_stat(
            gradient, center, "p90"
        )

    known_wc = worldcover_valid.astype(bool)
    built = (worldcover == 50) & known_wc
    for pixels in (10, 25, 50, 100):
        region = known_wc & (radius <= pixels)
        denominator = int(region.sum())
        features[f"morph_built_fraction_r{pixels}"] = (
            float((built & region).sum() / denominator) if denominator else float("nan")
        )
    horizontal = built[:, 1:] != built[:, :-1]
    vertical = built[1:, :] != built[:-1, :]
    features["morph_built_edge_density"] = float(
        (horizontal.sum() + vertical.sum()) / (horizontal.size + vertical.size)
    )
    labels, components = ndimage.label(built, structure=np.ones((3, 3), dtype="int8"))
    if components:
        sizes = np.bincount(labels.ravel())[1:]
        features["morph_built_components_per_10k"] = float(
            components * 10_000 / max(known_wc.sum(), 1)
        )
        features["morph_largest_built_component_fraction"] = float(
            sizes.max() / max(built.sum(), 1)
        )
    else:
        features["morph_built_components_per_10k"] = 0.0
        features["morph_largest_built_component_fraction"] = 0.0
    features["morph_high_ndbi_center_fraction"] = float(
        np.mean(arrays["ndbi"][center] > 0) if center.any() else np.nan
    )
    features["morph_high_ndvi_center_fraction"] = float(
        np.mean(arrays["ndvi"][center] > 0.4) if center.any() else np.nan
    )
    return features


def extract_morphology(
    input_root: str | Path,
    cohort: pd.DataFrame,
    cache_path: str | Path,
    chip_inventory: tuple[dict[object, Path], str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".json")
    if chip_inventory is None:
        chip_inventory = _cohort_chip_inventory(input_root, cohort)
    chip_paths, chip_inventory_sha256 = chip_inventory
    expected = {
        "protocol": PROTOCOL,
        "cohort_sha256": _cohort_hash(cohort),
        "chip_inventory_sha256": chip_inventory_sha256,
    }
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            frame = pd.read_parquet(cache_path)
            cache_hash_ok = manifest.get("sha256") == file_hash(cache_path)
            if (
                cache_hash_ok
                and frame.source_id.is_unique
                and set(frame.source_id) == set(cohort.source_id)
            ):
                return frame, manifest
    rows = []
    for source_id in cohort.source_id:
        path = chip_paths[source_id]
        with np.load(path, allow_pickle=False) as data:
            row = morphology_features_from_chip(
                data["reflectance"], data["valid_mask"],
                data["worldcover"], data["worldcover_valid"],
            )
        rows.append({"source_id": source_id, **row})
    frame = pd.DataFrame(rows)
    feature_columns = [column for column in frame if column.startswith("morph_")]
    values = frame[feature_columns].to_numpy(dtype="float64")
    if np.isinf(values).any() or not frame.source_id.is_unique:
        raise ValueError("Invalid morphology features")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    manifest = {
        **expected,
        "n_sources": len(frame),
        "features": feature_columns,
        "sha256": file_hash(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def _balanced_country_class_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = list(zip(frame.country, frame.is_eog_flare))
    counts = pd.Series(keys).value_counts()
    raw = np.array([1.0 / counts[key] for key in keys], dtype="float64")
    raw /= raw.mean()
    raw = np.minimum(raw, 5.0)
    return raw / raw.mean()


class VisualPUModel:
    """Fold-local PCA plus regularized PU-bagged logistic regression."""

    def __init__(
        self,
        pca_components: int = 12,
        c: float = 0.10,
        pu_bags: int = 5,
        unlabelled_per_positive: float = 2.0,
        seed: int = 42,
    ):
        self.pca_components = pca_components
        self.c = c
        self.pu_bags = pu_bags
        self.unlabelled_per_positive = unlabelled_per_positive
        self.seed = seed
        self.embedding_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.aux_medians: np.ndarray | None = None
        self.aux_scaler = StandardScaler()
        self.models: list[LogisticRegression] = []

    @staticmethod
    def _normalise_embedding_views(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalise each 512-dimensional view before fold-local PCA."""
        values = np.asarray(embeddings, dtype="float32")
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError(f"Invalid embedding matrix: {values.shape}")
        view_width = 512 if values.shape[1] % 512 == 0 else values.shape[1]
        views = values.reshape(len(values), -1, view_width)
        norms = np.linalg.norm(views, axis=2, keepdims=True)
        normalised = views / np.maximum(norms, 1e-12)
        return normalised.reshape(values.shape)

    def _fit_transform(self, embeddings: np.ndarray, aux: np.ndarray) -> np.ndarray:
        normalised = self._normalise_embedding_views(embeddings)
        scaled = self.embedding_scaler.fit_transform(normalised)
        components = min(self.pca_components, len(embeddings) - 2, embeddings.shape[1])
        if components < 2:
            raise ValueError("Insufficient rows for fold-local image PCA")
        self.pca = PCA(n_components=components, whiten=False, random_state=self.seed)
        parts = [self.pca.fit_transform(scaled)]
        if aux.shape[1]:
            self.aux_medians = np.nanmedian(aux, axis=0)
            self.aux_medians = np.where(np.isfinite(self.aux_medians), self.aux_medians, 0)
            filled = np.where(np.isfinite(aux), aux, self.aux_medians)
            parts.append(self.aux_scaler.fit_transform(filled))
        return np.concatenate(parts, axis=1).astype("float32")

    def _transform(self, embeddings: np.ndarray, aux: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Visual model has not been fitted")
        normalised = self._normalise_embedding_views(embeddings)
        parts = [self.pca.transform(self.embedding_scaler.transform(normalised))]
        if aux.shape[1]:
            if self.aux_medians is None:
                raise RuntimeError("Visual auxiliary transformer is missing")
            filled = np.where(np.isfinite(aux), aux, self.aux_medians)
            parts.append(self.aux_scaler.transform(filled))
        return np.concatenate(parts, axis=1).astype("float32")

    def fit(self, frame: pd.DataFrame, embeddings: np.ndarray, aux: np.ndarray):
        x = self._fit_transform(embeddings, aux)
        y = frame.is_eog_flare.to_numpy(dtype="int8")
        positive = np.flatnonzero(y == 1)
        unlabelled = np.flatnonzero(y == 0)
        if len(positive) < 5 or len(unlabelled) < 5:
            raise ValueError("Visual PU model needs both label classes")
        keep_unlabelled = min(
            len(unlabelled),
            max(5, int(math.ceil(self.unlabelled_per_positive * len(positive)))),
        )
        weights = _balanced_country_class_weights(frame)
        self.models = []
        for bag in range(self.pu_bags):
            rng = np.random.default_rng(self.seed + bag * 1009)
            chosen_unlabelled = rng.choice(
                unlabelled, size=keep_unlabelled, replace=False
            )
            chosen = np.concatenate([positive, chosen_unlabelled])
            model = LogisticRegression(
                C=self.c,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=self.seed + bag,
            )
            model.fit(x[chosen], y[chosen], sample_weight=weights[chosen])
            self.models.append(model)
        return self

    def predict(self, embeddings: np.ndarray, aux: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Visual model has not been fitted")
        x = self._transform(embeddings, aux)
        return np.mean([model.predict_proba(x)[:, 1] for model in self.models], axis=0)


def visual_country_oof(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    aux: np.ndarray,
    seed: int,
    pca_components: int,
) -> np.ndarray:
    score = np.empty(len(frame), dtype="float64")
    for index, country in enumerate(sorted(frame.country.unique())):
        fit_index = np.flatnonzero(frame.country.ne(country).to_numpy())
        test_index = np.flatnonzero(frame.country.eq(country).to_numpy())
        model = VisualPUModel(
            pca_components=pca_components, seed=seed + index * 100
        ).fit(frame.iloc[fit_index], embeddings[fit_index], aux[fit_index])
        score[test_index] = model.predict(embeddings[test_index], aux[test_index])
    return score


def visual_outer_predict(
    train: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_aux: np.ndarray,
    test_aux: np.ndarray,
    seed: int,
    pca_components: int,
) -> tuple[np.ndarray, VisualPUModel]:
    model = VisualPUModel(pca_components=pca_components, seed=seed).fit(
        train, train_embeddings, train_aux
    )
    return model.predict(test_embeddings, test_aux), model


def country_percentile(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    if len(frame) != len(score):
        raise ValueError("Score length does not match frame")
    series = pd.Series(score, index=frame.index, dtype="float64")
    ranked = series.groupby(frame.country).rank(method="average", pct=True)
    return ranked.to_numpy(dtype="float64")


def guarded_rank_fusion(
    frame: pd.DataFrame,
    tabular_score: np.ndarray,
    visual_score: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Cross-fit calibration, then apply visual evidence to ambiguous scores."""
    countries = sorted(frame.country.unique())
    if len(countries) < 2:
        raise ValueError("Cross-fitted fusion needs at least two countries")
    output = np.empty(len(frame), dtype="float64")
    for country in countries:
        test = frame.country.eq(country).to_numpy()
        reference = ~test
        output[test] = calibrated_residual_fusion(
            np.asarray(tabular_score)[reference],
            np.asarray(visual_score)[reference],
            np.asarray(tabular_score)[test],
            np.asarray(visual_score)[test],
            alpha,
        )
    return output


def _logit_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("Score calibration received non-finite values")
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def _calibrate_score(reference: np.ndarray, score: np.ndarray) -> np.ndarray:
    reference_logit = _logit_score(reference)
    scale = max(float(reference_logit.std()), 1e-6)
    z_score = (_logit_score(score) - float(reference_logit.mean())) / scale
    return expit(z_score)


def calibrated_residual_fusion(
    reference_tabular: np.ndarray,
    reference_visual: np.ndarray,
    tabular_score: np.ndarray,
    visual_score: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse scores using calibration fitted only on training-country scores."""
    tabular = _calibrate_score(reference_tabular, tabular_score)
    visual = _calibrate_score(reference_visual, visual_score)
    uncertainty = 4.0 * tabular * (1.0 - tabular)
    fused = tabular + alpha * uncertainty * (visual - tabular)
    return np.clip(fused, 0.0, 1.0)


def apply_deployment_policy(
    frame: pd.DataFrame,
    compact_score: np.ndarray,
    deployment_branch: str,
    fusion_alpha: float = 0.0,
    all77_score: np.ndarray | None = None,
    visual_score: np.ndarray | None = None,
    image_available: np.ndarray | None = None,
    tabular_reference: np.ndarray | None = None,
    visual_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the exported stage-B policy with compact fallback."""
    compact = np.asarray(compact_score, dtype="float64")
    if len(frame) != len(compact) or not np.isfinite(compact).all():
        raise ValueError("Invalid compact score array")
    if image_available is None:
        available = np.ones(len(frame), dtype=bool)
    else:
        available = np.asarray(image_available, dtype=bool)
        if available.shape != (len(frame),):
            raise ValueError("Image availability mask has the wrong shape")
    output = compact.copy()
    if deployment_branch == "compact_tabular":
        return output
    if deployment_branch == "all77_tabular":
        if all77_score is None:
            raise ValueError("all77_tabular requires all77_score")
        candidate = np.asarray(all77_score, dtype="float64")
        if candidate.shape != output.shape or not np.isfinite(candidate[available]).all():
            raise ValueError("Invalid all77 score array")
        for index in frame.groupby("country", sort=False).indices.values():
            index = np.asarray(index, dtype="int64")
            if available[index].all():
                output[index] = candidate[index]
        return output
    if deployment_branch == "guarded_cv_tabular":
        if visual_score is None:
            raise ValueError("guarded_cv_tabular requires visual_score")
        visual = np.asarray(visual_score, dtype="float64")
        if visual.shape != output.shape or not np.isfinite(visual[available]).all():
            raise ValueError("Invalid visual score array")
        if tabular_reference is None or visual_reference is None:
            raise ValueError("guarded fusion requires saved calibration references")
        calibrated_compact = _calibrate_score(tabular_reference, compact)
        output = calibrated_compact
        available_index = np.flatnonzero(available)
        if len(available_index):
            output[available_index] = calibrated_residual_fusion(
                tabular_reference,
                visual_reference,
                compact[available_index],
                visual[available_index],
                fusion_alpha,
            )
        return output
    raise ValueError(f"Unknown deployment branch: {deployment_branch}")


def country_ap_deltas(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    deltas = {}
    for country, part in frame.groupby("country"):
        index = part.index.to_numpy()
        y = part.is_eog_flare.to_numpy(dtype="int8")
        deltas[str(country)] = float(
            average_precision_score(y, candidate[index])
            - average_precision_score(y, baseline[index])
        )
    return deltas


def choose_alpha(
    frame: pd.DataFrame,
    tabular_score: np.ndarray,
    visual_score: np.ndarray,
) -> tuple[float, np.ndarray, list[dict]]:
    baseline_ap = macro_ap(frame, tabular_score)
    diagnostics = []
    candidates = []
    for alpha in ALPHAS:
        score = guarded_rank_fusion(frame, tabular_score, visual_score, alpha)
        deltas = country_ap_deltas(frame, tabular_score, score)
        macro_value = macro_ap(frame, score)
        eligible = (
            alpha == 0
            or (
                macro_value - baseline_ap >= 0.002
                and sum(delta > 0 for delta in deltas.values()) >= 3
                and min(deltas.values()) >= -0.03
            )
        )
        row = {
            "alpha": alpha,
            "macro_ap": macro_value,
            "gain": macro_value - baseline_ap,
            "improved_countries": sum(delta > 0 for delta in deltas.values()),
            "worst_delta": min(deltas.values()),
            "eligible": eligible,
        }
        diagnostics.append(row)
        if eligible:
            candidates.append((macro_value, -alpha, alpha, score))
    _, _, selected_alpha, selected_score = max(candidates)
    return float(selected_alpha), selected_score, diagnostics


def choose_inner_branch(
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
) -> tuple[str, list[dict]]:
    """Choose a challenger using only inner country-held-out predictions."""
    required = {"compact_tabular", "all77_tabular", "guarded_cv_tabular"}
    if set(scores) != required:
        raise ValueError(f"Inner branch scores must be exactly {sorted(required)}")
    baseline = scores["compact_tabular"]
    baseline_ap = macro_ap(frame, baseline)
    _, baseline_f1 = macro_f1_threshold(frame, baseline)
    diagnostics = []
    eligible: list[tuple[float, int, str]] = []
    complexity = {
        "compact_tabular": 0,
        "all77_tabular": 1,
        "guarded_cv_tabular": 2,
    }
    for branch in ("compact_tabular", "all77_tabular", "guarded_cv_tabular"):
        score = scores[branch]
        branch_ap = macro_ap(frame, score)
        _, branch_f1 = macro_f1_threshold(frame, score)
        deltas = country_ap_deltas(frame, baseline, score)
        passed = branch == "compact_tabular" or (
            branch_ap - baseline_ap
            >= INNER_SELECTION_GUARD["minimum_macro_ap_gain"]
            and sum(delta > 0 for delta in deltas.values())
            >= INNER_SELECTION_GUARD["minimum_improved_countries"]
            and min(deltas.values())
            >= INNER_SELECTION_GUARD["minimum_worst_country_delta"]
            and branch_f1 - baseline_f1
            >= INNER_SELECTION_GUARD["minimum_macro_f1_delta"]
        )
        diagnostics.append({
            "branch": branch,
            "macro_ap": branch_ap,
            "macro_ap_gain": branch_ap - baseline_ap,
            "macro_f1": branch_f1,
            "macro_f1_delta": branch_f1 - baseline_f1,
            "improved_countries": sum(delta > 0 for delta in deltas.values()),
            "worst_country_delta": min(deltas.values()),
            "eligible": bool(passed),
        })
        if passed:
            eligible.append((branch_ap, -complexity[branch], branch))
    return max(eligible)[2], diagnostics


def paired_country_bootstrap(
    predictions: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    repeats: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype="float64")
    groups = [part.reset_index(drop=True) for _, part in predictions.groupby("country")]
    for repeat in range(repeats):
        country_differences = []
        for part in groups:
            sampled = []
            for label in (0, 1):
                index = np.flatnonzero(part.is_eog_flare.to_numpy() == label)
                sampled.append(rng.choice(index, size=len(index), replace=True))
            index = np.concatenate(sampled)
            y = part.is_eog_flare.to_numpy(dtype="int8")[index]
            country_differences.append(
                average_precision_score(y, part[candidate_column].to_numpy()[index])
                - average_precision_score(y, part[baseline_column].to_numpy()[index])
            )
        differences[repeat] = np.mean(country_differences)
    return {
        "mean": float(differences.mean()),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "repeats": repeats,
    }


def selection_guard(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate: str,
    bootstrap_repeats: int,
    seed: int,
) -> dict:
    baseline_rows = metrics.loc[metrics.branch.eq("compact_tabular")].set_index("country")
    candidate_rows = metrics.loc[metrics.branch.eq(candidate)].set_index("country")
    delta = candidate_rows.pr_auc - baseline_rows.pr_auc
    bootstrap = paired_country_bootstrap(
        predictions,
        "score_compact_tabular",
        f"score_{candidate}",
        repeats=bootstrap_repeats,
        seed=seed,
    )
    macro_gain = float(delta.mean())
    macro_f1_delta = float(
        candidate_rows.f1.mean() - baseline_rows.f1.mean()
    )
    indonesia_delta = float(delta.loc["Indonesia"])
    passed = (
        macro_gain >= SELECTION_GUARD["minimum_macro_ap_gain"]
        and int((delta > 0).sum()) >= SELECTION_GUARD["minimum_improved_countries"]
        and float(delta.min()) >= SELECTION_GUARD["minimum_worst_country_delta"]
        and indonesia_delta >= SELECTION_GUARD["minimum_indonesia_delta"]
        and bootstrap["lower_95"] >= SELECTION_GUARD["minimum_bootstrap_lower_95"]
        and macro_f1_delta >= SELECTION_GUARD["minimum_macro_f1_delta"]
    )
    return {
        "candidate": candidate,
        "macro_ap_gain": macro_gain,
        "improved_countries": int((delta > 0).sum()),
        "worst_country_delta": float(delta.min()),
        "indonesia_delta": indonesia_delta,
        "macro_f1_delta": macro_f1_delta,
        "country_deltas": delta.to_dict(),
        "bootstrap": bootstrap,
        "passed": bool(passed),
    }


def budget_metrics(predictions: pd.DataFrame, branches: list[str]) -> pd.DataFrame:
    rows = []
    for country, part in predictions.groupby("country"):
        for branch in branches:
            ordered = part.sort_values(f"score_{branch}", ascending=False)
            positives = max(int(ordered.is_eog_flare.sum()), 1)
            for fraction in (0.10, 0.20, 0.30):
                count = max(1, int(math.ceil(fraction * len(ordered))))
                selected = ordered.head(count)
                found = int(selected.is_eog_flare.sum())
                rows.append({
                    "country": country,
                    "branch": branch,
                    "review_fraction": fraction,
                    "review_count": count,
                    "positive_found": found,
                    "precision_at_budget": found / count,
                    "recall_at_budget": found / positives,
                })
    return pd.DataFrame(rows)


def _aligned_matrix(
    cohort: pd.DataFrame,
    frame: pd.DataFrame,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    columns = sorted(column for column in frame if column.startswith(prefix))
    if not columns or FORBIDDEN_FEATURES & set(columns):
        raise ValueError(f"Invalid feature columns for prefix {prefix}")
    aligned = cohort[["source_id"]].merge(
        frame[["source_id"] + columns], on="source_id", how="left", validate="one_to_one"
    )
    matrix = aligned[columns].to_numpy(dtype="float32")
    if np.isinf(matrix).any() or np.isnan(matrix).all(axis=1).any():
        raise ValueError(f"Invalid aligned matrix for prefix {prefix}")
    return matrix, columns


def run(
    input_root: str | Path = "/kaggle/input",
    output_root: str | Path = "/kaggle/working/nb12_cv_tabular",
    rounds: int = 300,
    pca_components: int = 12,
    bootstrap_repeats: int = 1000,
    seed: int = 271,
    checkpoint_path: str | Path | None = None,
):
    started = time.time()
    input_root = Path(input_root)
    output_root = Path(output_root)
    outputs = output_root / "outputs"
    models = output_root / "models"
    cache = output_root / "cache"
    outputs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs / "12_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh NB12 output directory")
    shutil.copy2(Path(__file__), output_root / "kg_12_cv_tabular.py")
    baseline_source = Path(__file__).with_name("kg_08_fusion.py")
    if not baseline_source.is_file():
        raise FileNotFoundError(f"Missing baseline implementation: {baseline_source}")
    shutil.copy2(baseline_source, output_root / "kg_08_fusion.py")

    cohort, input_metadata = load_cohort(input_root)
    if HOLDOUT in set(cohort.country) or set(cohort.country) != set(COUNTRIES):
        raise ValueError("NB12 must contain exactly the six foreign countries")
    chip_inventory = _cohort_chip_inventory(input_root, cohort)
    embeddings, embedding_metadata = extract_cv_embeddings(
        input_root,
        cohort,
        cache / "12_cv_embeddings.parquet",
        checkpoint_path=checkpoint_path,
        chip_inventory=chip_inventory,
    )
    morphology, morphology_metadata = extract_morphology(
        input_root,
        cohort,
        cache / "12_morphology.parquet",
        chip_inventory=chip_inventory,
    )
    embedding_matrix, embedding_columns = _aligned_matrix(cohort, embeddings, "cv_")
    morphology_matrix, morphology_columns = _aligned_matrix(cohort, morphology, "morph_")
    feature_manifest_path = find_unique(input_root, "feature_manifest.json")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if (
        feature_manifest.get("protocol") != "nb6-context-v1"
        or feature_manifest.get("holdout_loaded") is not False
    ):
        raise ValueError("NB6 feature manifest is incompatible or loaded India")
    image_columns = list(feature_manifest.get("features", []))
    if (
        len(image_columns) != 77
        or len(set(image_columns)) != 77
        or not all(column.startswith("img_") for column in image_columns)
        or not set(image_columns).issubset(cohort.columns)
    ):
        raise ValueError("NB6 feature manifest must declare exactly 77 image features")
    all77_columns = list(THERMAL_COLS) + image_columns
    if FORBIDDEN_FEATURES & set(all77_columns):
        raise ValueError("Forbidden field entered the all-image model")
    image_matrix = cohort[image_columns].to_numpy(dtype="float32")
    visual_aux = np.concatenate([image_matrix, morphology_matrix], axis=1)
    empty_aux = np.empty((len(cohort), 0), dtype="float32")

    metric_rows = []
    selection_rows = []
    prediction_parts = []
    for outer_index, held_out in enumerate(COUNTRIES):
        train_mask = cohort.country.ne(held_out).to_numpy()
        test_mask = ~train_mask
        train = cohort.loc[train_mask].reset_index(drop=True)
        test = cohort.loc[test_mask].reset_index(drop=True)
        train_embeddings = embedding_matrix[train_mask]
        test_embeddings = embedding_matrix[test_mask]
        train_visual_aux = visual_aux[train_mask]
        test_visual_aux = visual_aux[test_mask]
        train_empty = empty_aux[train_mask]
        test_empty = empty_aux[test_mask]
        fold_seed = seed + outer_index * 10_000

        inner_scores = {
            "compact_tabular": inner_country_oof(
                train, TABULAR_COLUMNS, fold_seed, rounds
            ),
            "all77_tabular": inner_country_oof(
                train, all77_columns, fold_seed + 1000, rounds
            ),
            "handcrafted_image": inner_country_oof(
                train, image_columns, fold_seed + 2000, rounds
            ),
            "cnn_only": visual_country_oof(
                train, train_embeddings, train_empty, fold_seed + 3000, pca_components
            ),
            "cv_multispectral": visual_country_oof(
                train, train_embeddings, train_visual_aux,
                fold_seed + 4000, pca_components,
            ),
        }
        seeds = [fold_seed + 20_000 + offset for offset in (0, 101, 202)]
        outer_scores = {
            "compact_tabular": fit_predict(
                train, test, TABULAR_COLUMNS, seeds, rounds
            ),
            "all77_tabular": fit_predict(
                train, test, all77_columns, [value + 1000 for value in seeds], rounds
            ),
            "handcrafted_image": fit_predict(
                train, test, image_columns, [value + 2000 for value in seeds], rounds
            ),
        }
        outer_scores["cnn_only"], _ = visual_outer_predict(
            train, train_embeddings, test_embeddings, train_empty, test_empty,
            fold_seed + 3000, pca_components,
        )
        outer_scores["cv_multispectral"], _ = visual_outer_predict(
            train, train_embeddings, test_embeddings,
            train_visual_aux, test_visual_aux,
            fold_seed + 4000, pca_components,
        )
        alpha, inner_fusion, alpha_diagnostics = choose_alpha(
            train,
            inner_scores["compact_tabular"],
            inner_scores["cv_multispectral"],
        )
        inner_scores["guarded_cv_tabular"] = inner_fusion
        outer_scores["guarded_cv_tabular"] = calibrated_residual_fusion(
            inner_scores["compact_tabular"],
            inner_scores["cv_multispectral"],
            outer_scores["compact_tabular"],
            outer_scores["cv_multispectral"],
            alpha,
        )
        nested_selected, nested_diagnostics = choose_inner_branch(
            train,
            {
                branch: inner_scores[branch]
                for branch in (
                    "compact_tabular", "all77_tabular", "guarded_cv_tabular"
                )
            },
        )
        inner_scores["nested_champion"] = inner_scores[nested_selected]
        outer_scores["nested_champion"] = outer_scores[nested_selected]

        for branch, score in inner_scores.items():
            threshold, inner_f1 = macro_f1_threshold(train, score)
            metric_rows.append(
                metric_row(test, outer_scores[branch], threshold, branch, held_out)
            )
            selection_rows.append({
                "held_out_country": held_out,
                "branch": branch,
                "inner_macro_ap": macro_ap(train, score),
                "inner_macro_f1": inner_f1,
                "threshold": threshold,
                "fusion_alpha": alpha if branch == "guarded_cv_tabular" else np.nan,
                "inner_selected_branch": (
                    nested_selected if branch == "nested_champion" else None
                ),
                "branch_selection_diagnostics": (
                    json.dumps(nested_diagnostics)
                    if branch == "nested_champion" else None
                ),
                "alpha_diagnostics": (
                    json.dumps(alpha_diagnostics)
                    if branch == "guarded_cv_tabular" else None
                ),
            })
        part = test[[
            "source_id", "country", "is_eog_flare", "eog_flare_id", "block_id"
        ]].copy()
        for branch, score in outer_scores.items():
            part[f"score_{branch}"] = score
        prediction_parts.append(part)
        print(
            f"Completed NB12 holdout {held_out}, alpha={alpha:.2f}, "
            f"inner winner={nested_selected}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    branches = sorted(metrics.branch.unique())
    summary = metrics.groupby("branch").agg(
        macro_f1=("f1", "mean"),
        macro_pr_auc=("pr_auc", "mean"),
        macro_roc_auc=("roc_auc", "mean"),
        worst_country_pr_auc=("pr_auc", "min"),
    ).reset_index().sort_values(["macro_pr_auc", "macro_f1"], ascending=False)
    guards = {
        candidate: selection_guard(
            metrics, predictions, candidate, bootstrap_repeats, seed + index * 1000
        )
        for index, candidate in enumerate(
            ["all77_tabular", "guarded_cv_tabular", "nested_champion"]
        )
    }
    selected = (
        "nested_champion"
        if guards["nested_champion"]["passed"]
        else "compact_tabular"
    )

    ordered = predictions.set_index("source_id").loc[cohort.source_id]
    selected_oof = ordered[f"score_{selected}"].to_numpy(dtype="float64")
    pilot_threshold, _ = macro_f1_threshold(cohort, selected_oof)
    diagnostic_alpha, _, global_alpha_diagnostics = choose_alpha(
        cohort,
        ordered.score_compact_tabular.to_numpy(dtype="float64"),
        ordered.score_cv_multispectral.to_numpy(dtype="float64"),
    )
    if selected == "nested_champion":
        deployment_branch, deployment_diagnostics = choose_inner_branch(
            cohort,
            {
                branch: ordered[f"score_{branch}"].to_numpy(dtype="float64")
                for branch in (
                    "compact_tabular", "all77_tabular", "guarded_cv_tabular"
                )
            },
        )
    else:
        deployment_branch = "compact_tabular"
        deployment_diagnostics = []
    global_alpha = (
        diagnostic_alpha if deployment_branch == "guarded_cv_tabular" else 0.0
    )

    model_artifacts = []
    importance_parts = []
    model_schemas = {"compact_fallback": TABULAR_COLUMNS}
    if deployment_branch == "all77_tabular":
        model_schemas["deployment_all77"] = all77_columns
    final_seeds = [seed + 50_000, seed + 50_101, seed + 50_202]
    for model_name, columns in model_schemas.items():
        importance = np.zeros(len(columns), dtype="float64")
        for index, final_seed in enumerate(final_seeds):
            model = train_model(cohort, columns, final_seed, rounds)
            path = models / f"12_{model_name}_{index}.txt"
            model.save_model(str(path))
            model_artifacts.append(path.name)
            importance += model.feature_importance("gain") / len(final_seeds)
        importance_parts.append(pd.DataFrame({
            "model": model_name,
            "feature": columns,
            "mean_gain": importance,
        }))
    pd.concat(importance_parts, ignore_index=True).sort_values(
        ["model", "mean_gain"], ascending=[True, False]
    ).to_csv(outputs / "12_final_feature_importance.csv", index=False)
    final_visual = VisualPUModel(
        pca_components=pca_components, seed=seed + 60_000
    ).fit(cohort, embedding_matrix, visual_aux)
    joblib.dump(final_visual, models / "12_visual_pu.joblib", compress=3)
    model_artifacts.append("12_visual_pu.joblib")
    np.savez_compressed(
        models / "12_fusion_calibration.npz",
        compact_oof=ordered.score_compact_tabular.to_numpy(dtype="float64"),
        visual_oof=ordered.score_cv_multispectral.to_numpy(dtype="float64"),
    )
    model_artifacts.append("12_fusion_calibration.npz")
    clean_environment = os.environ.copy()
    clean_environment["PYTHONPATH"] = str(output_root)
    portability = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import joblib; "
                "joblib.load(r'models/12_visual_pu.joblib'); "
                "print('portable model load passed')"
            ),
        ],
        cwd=output_root,
        env=clean_environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if portability.returncode != 0:
        raise RuntimeError(
            "Saved visual model failed a clean portability check:\n"
            f"{portability.stdout}\n{portability.stderr}"
        )

    budget = budget_metrics(predictions, branches)
    metrics.to_csv(outputs / "12_country_metrics.csv", index=False)
    summary.to_csv(outputs / "12_branch_summary.csv", index=False)
    selections.to_csv(outputs / "12_inner_selection.csv", index=False)
    predictions.to_parquet(outputs / "12_loco_predictions.parquet", index=False)
    budget.to_csv(outputs / "12_budget_metrics.csv", index=False)
    schema = {
        "selected_branch": selected,
        "selected_branch_is_nested_policy": selected == "nested_champion",
        "deployment_branch": deployment_branch,
        "deployment_branch_diagnostics": deployment_diagnostics,
        "tabular_columns": (
            all77_columns
            if deployment_branch == "all77_tabular" else TABULAR_COLUMNS
        ),
        "compact_fallback_columns": TABULAR_COLUMNS,
        "image_columns": image_columns,
        "morphology_columns": morphology_columns,
        "embedding_columns": embedding_columns,
        "pca_components": pca_components,
        "fusion_alpha": global_alpha,
        "fusion_alpha_source": "foreign-country OOF after nested policy evaluation",
        "fusion_alpha_policy": (
            "Each outer country used an alpha selected only by its inner countries. "
            "The final alpha is reselected by full foreign-country LOCO."
        ),
        "diagnostic_oof_best_alpha": diagnostic_alpha,
        "fusion_formula": (
            "calibrated_tabular + alpha * 4 * calibrated_tabular * "
            "(1-calibrated_tabular) * "
            "(calibrated_visual-calibrated_tabular)"
        ),
        "score_calibration": "logit z-score fitted on foreign OOF reference scores",
        "missing_image_fallback": "compact tabular score",
        "pilot_threshold": pilot_threshold,
        "threshold_is_deployment_calibrated": False,
    }
    (outputs / "12_selected_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "training_countries": COUNTRIES,
        "holdout_country": HOLDOUT,
        "holdout_loaded": False,
        "interpretation": (
            "Enriched foreign imagery reranker comparison, not population precision"
        ),
        "selected_branch": selected,
        "deployment_branch": deployment_branch,
        "selected_branch_estimate": (
            "fully nested outer-country policy estimate"
            if selected == "nested_champion"
            else "preregistered compact fallback estimate"
        ),
        "selection_guard": SELECTION_GUARD,
        "selection_results": guards,
        "global_alpha_diagnostics": global_alpha_diagnostics,
        "rounds": rounds,
        "pca_components": pca_components,
        "seed": seed,
        "checkpoint": embedding_metadata,
        "morphology": morphology_metadata,
        "input": input_metadata,
        "source_sha256": {
            "kg_08_fusion.py": file_hash(output_root / "kg_08_fusion.py"),
            "kg_12_cv_tabular.py": file_hash(output_root / "kg_12_cv_tabular.py"),
        },
        "model_artifacts": model_artifacts,
        "pilot_threshold": pilot_threshold,
        "threshold_is_deployment_calibrated": False,
        "limitations": [
            "The imagery cohort is enriched and has fewer than 300 QA-approved sources.",
            "EOG supplies gas-flare positives, so unlabelled does not mean non-industrial.",
            "The frozen encoder uses RGB only; NIR and SWIR enter through morphology features.",
            "No India imagery or labels were loaded.",
            "Sources without QA-approved imagery fall back to the compact tabular model.",
        ],
        "elapsed_minutes": (time.time() - started) / 60,
        "python": platform.python_version(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                "torch", "pyarrow", "joblib",
            ]
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NB12 selected branch: {selected}", flush=True)
    return summary, metrics, predictions


if __name__ == "__main__":
    run()
