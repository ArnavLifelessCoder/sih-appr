"""Foreign-only temporal TCN features for the final SIH fusion model.

The module builds compact 36-month FIRMS sequences, trains a small residual
TCN, and produces country-excluded scores for nested fusion evaluation. India
is rejected at every public boundary. All normalization is fitted on the
training countries for the current exclusion, never on a scored country.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


PROTOCOL_VERSION = "10-temporal-tcn-v1"
COUNTRIES = ("Algeria", "Angola", "Indonesia", "Iraq", "Libya", "Nigeria")
HOLDOUT = "India"
WINDOW_YEARS = (2022, 2023, 2024)
FEATURE_TAG = "2022_2024"
MODEL_SENSORS = ("MODIS", "VIIRS_SNPP")
MONTHS = 36
CHANNELS = (
    "log_detection_count",
    "log_active_days",
    "log_frp_sum",
    "log_frp_mean",
    "mir_mean",
    "mir_lwir_mean",
    "night_fraction",
    "saturation_fraction",
    "modis_fraction",
    "presence",
    "month_sin",
    "month_cos",
)
CONTINUOUS_CHANNELS = tuple(range(9))
PRESENCE_CHANNEL = CHANNELS.index("presence")
DEFAULT_ARCHITECTURE = {
    "width": 32,
    "dilations": [1, 2, 4, 8],
    "kernel_size": 3,
    "dropout": 0.15,
}
HARDNESS_COLUMNS = (
    "active_days_per_year",
    "active_months_per_year",
    "det_per_year",
    "night_frac",
    "sat_frac",
    "dt_mir_lwir_mean",
    "frp_mean",
)
FEATURE_COLUMNS = (
    "source_id",
    "country",
    "block_id",
    "is_eog_flare",
    "eog_flare_id",
    *HARDNESS_COLUMNS,
)
DETECTION_COLUMNS = (
    "source_id",
    "acq_dt",
    "frp",
    "t_mir",
    "t_lwir",
    "daynight",
    "sensor",
    "year",
)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _find_equivalent(input_root: Path, name: str) -> tuple[Path, list[str]]:
    """Choose one file, accepting only byte-identical duplicate attachments."""
    matches = sorted(path for path in input_root.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Missing required input {name}")
    if len(matches) == 1:
        return matches[0], []
    sizes = {path.stat().st_size for path in matches}
    if len(sizes) != 1:
        raise FileNotFoundError(
            f"Conflicting copies of {name}; attach one source: {matches}"
        )
    digests = {_file_hash(path) for path in matches}
    if len(digests) != 1:
        raise FileNotFoundError(
            f"Non-identical copies of {name}; attach one source: {matches}"
        )
    return matches[0], [str(path) for path in matches[1:]]


def _validate_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    required = {"source_id", "country"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Cohort is missing columns: {sorted(missing)}")
    out = cohort.copy()
    if out.source_id.isna().any() or out.country.isna().any():
        raise ValueError("Cohort source IDs and countries must be non-null")
    out["source_id"] = out.source_id.astype(str)
    out["country"] = out.country.astype(str)
    if not out.source_id.is_unique:
        raise ValueError("Cohort source IDs must be non-null and unique")
    if HOLDOUT in set(out.country) or not set(out.country).issubset(COUNTRIES):
        raise ValueError("Cohort must contain foreign countries only; India is forbidden")
    return out


def _hardness(frame: pd.DataFrame) -> pd.Series:
    ranks = []
    for column in HARDNESS_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        ranks.append(values.rank(method="average", pct=True, na_option="bottom"))
    return pd.concat(ranks, axis=1).mean(axis=1)


def _sample_country(
    frame: pd.DataFrame,
    negative_ratio: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Select one source per known site and a hard/random unlabeled mixture."""
    country = str(frame.country.iloc[0])
    if not frame.country.eq(country).all() or country not in COUNTRIES:
        raise ValueError(f"Invalid country frame: {country}")
    if (
        not frame.source_id.is_unique
        or frame.block_id.isna().any()
        or not frame.is_eog_flare.isin([0, 1]).all()
    ):
        raise ValueError(f"Invalid identifiers or labels for {country}")

    known_positive = frame.loc[frame.is_eog_flare.eq(1)].copy()
    if known_positive.empty or known_positive.eog_flare_id.isna().any():
        raise ValueError(f"Known positives and site identifiers are required for {country}")
    positive_blocks = set(known_positive.block_id)
    positive = known_positive.sort_values(
        ["active_days_per_year", "det_per_year", "source_id"],
        ascending=[False, False, True],
        kind="stable",
    ).drop_duplicates("eog_flare_id", keep="first")
    positive["sample_role"] = "positive_site"
    positive["hardness"] = np.nan

    unlabeled_all = frame.loc[frame.is_eog_flare.eq(0)].copy()
    positive_block_overlap = unlabeled_all.block_id.isin(positive_blocks)
    excluded_positive_block_rows = int(positive_block_overlap.sum())
    unlabeled = unlabeled_all.loc[~positive_block_overlap].copy()
    unlabeled["hardness"] = _hardness(unlabeled)
    target = min(len(unlabeled), negative_ratio * len(positive))
    if target < 1:
        raise ValueError(f"No unlabeled training examples available for {country}")
    hard_count = (target + 1) // 2
    hard = unlabeled.sort_values(
        ["hardness", "source_id"], ascending=[False, True], kind="stable"
    ).head(hard_count).copy()
    hard["sample_role"] = "hard_unlabeled"

    remainder = unlabeled.loc[~unlabeled.source_id.isin(hard.source_id)].copy()
    random_count = target - len(hard)
    if random_count:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(remainder), size=random_count, replace=False)
        sampled_random = remainder.iloc[np.sort(positions)].copy()
    else:
        sampled_random = remainder.head(0).copy()
    sampled_random["sample_role"] = "random_unlabeled"

    selected = pd.concat([positive, hard, sampled_random], ignore_index=True)
    if not selected.source_id.is_unique:
        raise ValueError(f"Sampling produced duplicate sources for {country}")
    diagnostic = {
        "country": country,
        "known_positive_sources": len(known_positive),
        "unique_positive_sites": len(positive),
        "available_unlabeled_before_block_guard": len(unlabeled_all),
        "positive_block_unlabeled_excluded": excluded_positive_block_rows,
        "available_unlabeled_after_block_guard": len(unlabeled),
        "selected_hard_unlabeled": len(hard),
        "selected_random_unlabeled": len(sampled_random),
    }
    return selected, diagnostic


def _monthly_sequence(detections: pd.DataFrame, source_ids: Sequence[str]) -> np.ndarray:
    source_ids = [str(value) for value in source_ids]
    positions = {source_id: index for index, source_id in enumerate(source_ids)}
    sequence = np.zeros((len(source_ids), len(CHANNELS), MONTHS), dtype="float32")

    month_number = np.arange(MONTHS) % 12
    angle = 2.0 * np.pi * month_number / 12.0
    sequence[:, CHANNELS.index("month_sin"), :] = np.sin(angle).astype("float32")
    sequence[:, CHANNELS.index("month_cos"), :] = np.cos(angle).astype("float32")
    if detections.empty:
        return sequence

    frame = detections.copy()
    frame["source_id"] = frame.source_id.astype(str)
    frame = frame.loc[frame.source_id.isin(positions)].copy()
    if frame.empty:
        return sequence
    frame["acq_dt"] = pd.to_datetime(frame.acq_dt, errors="coerce")
    frame = frame.loc[
        frame.acq_dt.dt.year.isin(WINDOW_YEARS)
        & frame.sensor.astype(str).isin(MODEL_SENSORS)
    ].copy()
    if frame.empty:
        return sequence
    frame["month_index"] = (
        (frame.acq_dt.dt.year - WINDOW_YEARS[0]) * 12
        + frame.acq_dt.dt.month - 1
    ).astype("int16")
    frame["day"] = frame.acq_dt.dt.normalize()
    frame["mir_lwir"] = frame.t_mir - frame.t_lwir
    frame["is_night"] = frame.daynight.astype(str).str.upper().eq("N").astype("float32")
    frame["is_saturated"] = frame.t_mir.ge(367.0).astype("float32")
    frame["is_modis"] = frame.sensor.astype(str).eq("MODIS").astype("float32")

    grouped = frame.groupby(["source_id", "month_index"], observed=True)
    monthly = grouped.agg(
        detection_count=("frp", "size"),
        active_days=("day", "nunique"),
        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        mir_mean=("t_mir", "mean"),
        mir_lwir_mean=("mir_lwir", "mean"),
        night_fraction=("is_night", "mean"),
        saturation_fraction=("is_saturated", "mean"),
        modis_fraction=("is_modis", "mean"),
    ).reset_index()
    monthly = monthly.replace([np.inf, -np.inf], np.nan)

    row = monthly.source_id.map(positions).to_numpy(dtype="int64")
    month = monthly.month_index.to_numpy(dtype="int64")
    if not ((month >= 0) & (month < MONTHS)).all():
        raise ValueError("Detection outside the fixed 2022 to 2024 window")
    log_columns = (
        "detection_count", "active_days", "frp_sum", "frp_mean"
    )
    for channel, column in enumerate(log_columns):
        values = pd.to_numeric(monthly[column], errors="coerce").fillna(0)
        values = np.log1p(np.maximum(values.to_numpy(dtype="float32"), 0))
        sequence[row, channel, month] = values
    for channel, column in enumerate((
        "mir_mean", "mir_lwir_mean", "night_fraction",
        "saturation_fraction", "modis_fraction",
    ), start=4):
        values = pd.to_numeric(monthly[column], errors="coerce").fillna(0)
        sequence[row, channel, month] = values.to_numpy(dtype="float32")
    sequence[row, PRESENCE_CHANNEL, month] = 1.0
    return sequence


def _descriptors(meta: pd.DataFrame, sequences: np.ndarray) -> pd.DataFrame:
    present = sequences[:, PRESENCE_CHANNEL, :] > 0.5
    counts = np.expm1(sequences[:, CHANNELS.index("log_detection_count"), :])
    active_days = np.expm1(sequences[:, CHANNELS.index("log_active_days"), :])
    frp_sum = np.expm1(sequences[:, CHANNELS.index("log_frp_sum"), :])
    night = sequences[:, CHANNELS.index("night_fraction"), :]
    saturation = sequences[:, CHANNELS.index("saturation_fraction"), :]
    modis = sequences[:, CHANNELS.index("modis_fraction"), :]
    denominator = np.maximum(counts.sum(axis=1), 1.0)
    adjacent = (present[:, 1:] & present[:, :-1]).sum(axis=1)
    possible = np.maximum(present.sum(axis=1) - 1, 1)
    descriptor = meta[[
        "source_id", "country", "is_eog_flare", "sample_role", "train_selected",
        "is_cohort",
    ]].copy()
    descriptor["ts_detection_count"] = counts.sum(axis=1).astype("float32")
    descriptor["ts_active_days"] = active_days.sum(axis=1).astype("float32")
    descriptor["ts_active_months"] = present.sum(axis=1).astype("int16")
    descriptor["ts_frp_sum"] = frp_sum.sum(axis=1).astype("float32")
    descriptor["ts_mir_mean"] = (
        (sequences[:, CHANNELS.index("mir_mean"), :] * counts).sum(axis=1)
        / denominator
    ).astype("float32")
    descriptor["ts_night_fraction"] = (
        (night * counts).sum(axis=1) / denominator
    ).astype("float32")
    descriptor["ts_saturation_fraction"] = (
        (saturation * counts).sum(axis=1) / denominator
    ).astype("float32")
    descriptor["ts_modis_fraction"] = (
        (modis * counts).sum(axis=1) / denominator
    ).astype("float32")
    descriptor["ts_month_continuity"] = (adjacent / possible).astype("float32")
    return descriptor


def prepare_temporal_data(
    input_root: str | Path,
    cohort: pd.DataFrame,
    negative_ratio: int = 10,
    seed: int = 4103,
    include_population: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    """Build sampled foreign training data and aligned 36-month sequences.

    The returned ``meta`` and ``sequences`` share row order. ``meta`` contains
    every sampled training source plus every requested cohort source. By
    default it also contains the complete foreign common-window population so
    single-country exclusions can be measured at real prevalence.
    """
    input_root = Path(input_root)
    cohort = _validate_cohort(cohort)
    if not isinstance(negative_ratio, int) or negative_ratio < 1:
        raise ValueError("negative_ratio must be a positive integer")

    selected_parts = []
    sampling_diagnostics = []
    cohort_feature_parts = []
    population_feature_parts = []
    selected_paths: dict[str, str] = {}
    duplicate_paths: dict[str, list[str]] = {}
    input_hashes: dict[str, str] = {}
    for country_index, country in enumerate(COUNTRIES):
        name = f"features_{country}_{FEATURE_TAG}.parquet"
        path, duplicates = _find_equivalent(input_root, name)
        frame = pd.read_parquet(path, columns=list(FEATURE_COLUMNS))
        if frame.source_id.isna().any() or frame.country.isna().any():
            raise ValueError(f"Null source ID or country in {path}")
        frame["source_id"] = frame.source_id.astype(str)
        frame["country"] = frame.country.astype(str)
        if not frame.country.eq(country).all():
            raise ValueError(f"Wrong country content in {path}")
        selected_country, sampling_diagnostic = _sample_country(
            frame, negative_ratio, seed + country_index * 1009
        )
        selected_parts.append(selected_country)
        sampling_diagnostics.append(sampling_diagnostic)
        wanted_ids = set(cohort.loc[cohort.country.eq(country), "source_id"])
        cohort_feature_parts.append(frame.loc[
            frame.source_id.isin(wanted_ids),
            ["source_id", "country", "block_id", "is_eog_flare", "eog_flare_id"],
        ].copy())
        if include_population:
            population_feature_parts.append(frame[[
                "source_id", "country", "block_id", "is_eog_flare", "eog_flare_id"
            ]].copy())
        selected_paths[name] = str(path)
        duplicate_paths[name] = duplicates
        input_hashes[name] = _file_hash(path)
        del frame
        gc.collect()

    selected = pd.concat(selected_parts, ignore_index=True)
    if not selected.source_id.is_unique:
        raise ValueError("Sampled source IDs must be globally unique")
    selected["train_selected"] = True

    cohort_features = pd.concat(cohort_feature_parts, ignore_index=True)
    cohort_lookup = cohort[["source_id", "country"]].merge(
        cohort_features,
        on=["source_id", "country"], how="left", validate="one_to_one",
    )
    if cohort_lookup.is_eog_flare.isna().any():
        missing = cohort_lookup.loc[cohort_lookup.is_eog_flare.isna(), "source_id"]
        raise ValueError(f"Cohort source missing from common-window features: {missing.tolist()}")
    if "is_eog_flare" in cohort:
        supplied = cohort.set_index("source_id").is_eog_flare
        expected = cohort_lookup.set_index("source_id").is_eog_flare
        if not supplied.astype("int8").equals(expected.astype("int8")):
            raise ValueError("Cohort labels disagree with common-window features")

    meta_columns = [
        "source_id", "country", "block_id", "is_eog_flare", "eog_flare_id",
        "sample_role", "train_selected",
    ]
    selected_meta = selected[meta_columns].copy()
    if include_population:
        base = pd.concat(population_feature_parts, ignore_index=True)
        if not base.source_id.is_unique:
            raise ValueError("Population source IDs must be globally unique")
        extra_role = "population_only"
    else:
        base = cohort_lookup.copy()
        extra_role = "cohort_only"
    extra = base.loc[~base.source_id.isin(selected_meta.source_id)].copy()
    extra["sample_role"] = extra_role
    extra["train_selected"] = False
    meta = pd.concat([selected_meta, extra[meta_columns]], ignore_index=True)
    meta = meta.sort_values(["country", "source_id"], kind="stable").reset_index(drop=True)
    meta["is_eog_flare"] = meta.is_eog_flare.astype("int8")
    meta["is_cohort"] = meta.source_id.isin(cohort.source_id)
    meta["population_complete"] = bool(include_population)
    if not meta.source_id.is_unique or HOLDOUT in set(meta.country):
        raise ValueError("Aligned temporal metadata is invalid")
    del (
        selected_parts, cohort_feature_parts, population_feature_parts,
        cohort_features, selected, cohort_lookup, base, extra,
    )
    gc.collect()

    sequences = np.zeros((len(meta), len(CHANNELS), MONTHS), dtype="float32")
    month_number = np.arange(MONTHS) % 12
    sequences[:, CHANNELS.index("month_sin"), :] = np.sin(
        2.0 * np.pi * month_number / 12.0
    ).astype("float32")
    sequences[:, CHANNELS.index("month_cos"), :] = np.cos(
        2.0 * np.pi * month_number / 12.0
    ).astype("float32")

    for country in COUNTRIES:
        name = f"detections_{country}.parquet"
        path, duplicates = _find_equivalent(input_root, name)
        country_ids = meta.loc[meta.country.eq(country), "source_id"].tolist()
        detections = pd.read_parquet(
            path,
            columns=list(DETECTION_COLUMNS),
            filters=[("year", ">=", WINDOW_YEARS[0]), ("year", "<=", WINDOW_YEARS[-1])],
        )
        country_sequence = _monthly_sequence(detections, country_ids)
        indices = meta.index[meta.country.eq(country)].to_numpy(dtype="int64")
        sequences[indices] = country_sequence
        del detections, country_sequence
        gc.collect()
        selected_paths[name] = str(path)
        duplicate_paths[name] = duplicates
        input_hashes[name] = _file_hash(path)

    if not np.isfinite(sequences).all():
        raise ValueError("Temporal sequences contain non-finite values")
    if (sequences[:, PRESENCE_CHANNEL, :].sum(axis=1) == 0).any():
        missing = meta.loc[
            sequences[:, PRESENCE_CHANNEL, :].sum(axis=1) == 0, "source_id"
        ]
        raise ValueError(f"Sources without common-window detections: {missing.tolist()}")
    descriptors = _descriptors(meta, sequences)

    sample_counts = (
        meta.loc[meta.train_selected]
        .groupby(["country", "sample_role"], observed=True)
        .size().rename("sources").reset_index().to_dict("records")
    )
    cohort_signature = hashlib.sha256(
        cohort[["source_id", "country"]]
        .sort_values(["country", "source_id"])
        .to_csv(index=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        "protocol": PROTOCOL_VERSION,
        "countries": list(COUNTRIES),
        "holdout": HOLDOUT,
        "holdout_loaded": False,
        "window_years": list(WINDOW_YEARS),
        "channels": list(CHANNELS),
        "sequence_shape": [len(CHANNELS), MONTHS],
        "negative_ratio": negative_ratio,
        "hard_fraction": 0.5,
        "seed": seed,
        "sample_counts": sample_counts,
        "sampling_diagnostics": sampling_diagnostics,
        "training_sources": int(meta.train_selected.sum()),
        "cohort_sources": len(cohort),
        "population_included": bool(include_population),
        "aligned_sources": len(meta),
        "cohort_sha256": cohort_signature,
        "input_paths": selected_paths,
        "ignored_identical_duplicates": duplicate_paths,
        "input_sha256": input_hashes,
    }
    return meta, sequences, descriptors, metadata


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _bag_training_mask(
    meta: pd.DataFrame,
    excluded_countries: Sequence[str],
    unlabeled_keep_fraction: float,
    seed: int,
) -> np.ndarray:
    """Keep all positive sites and a deterministic fraction of each PU pool."""
    if not 0 < unlabeled_keep_fraction <= 1:
        raise ValueError("unlabeled_keep_fraction must be in (0, 1]")
    eligible = (
        meta.train_selected.to_numpy(dtype=bool)
        & ~meta.country.isin(excluded_countries).to_numpy()
    )
    labels = meta.is_eog_flare.to_numpy(dtype="int8")
    keep = eligible & labels.astype(bool)
    rng = np.random.default_rng(seed)
    unlabeled_positions = np.flatnonzero(eligible & (labels == 0))
    unlabeled = meta.iloc[unlabeled_positions][["country", "sample_role"]].copy()
    unlabeled["row_position"] = unlabeled_positions
    expected_roles = {"hard_unlabeled", "random_unlabeled"}
    if not set(unlabeled.sample_role).issubset(expected_roles):
        raise ValueError("Unexpected unlabeled sample role in TCN training pool")
    for _, part in unlabeled.groupby(["country", "sample_role"], observed=True):
        positions = part.row_position.to_numpy(dtype="int64")
        count = max(1, int(round(unlabeled_keep_fraction * len(positions))))
        chosen = rng.choice(positions, size=min(count, len(positions)), replace=False)
        keep[chosen] = True
    return keep


@dataclass(frozen=True)
class Normalization:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype="float32").copy()
        if values.ndim != 3 or values.shape[1:] != (len(CHANNELS), MONTHS):
            raise ValueError(f"Expected [N,{len(CHANNELS)},{MONTHS}] sequences")
        observed = values[:, PRESENCE_CHANNEL, :] > 0.5
        for channel in CONTINUOUS_CHANNELS:
            normalized = (values[:, channel, :] - self.center[channel]) / self.scale[channel]
            values[:, channel, :] = np.where(observed, normalized, 0.0)
            np.clip(
                values[:, channel, :], -8.0, 8.0,
                out=values[:, channel, :],
            )
        return values

    def to_dict(self) -> dict:
        return {
            "center": self.center.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }


def _fit_normalization(sequences: np.ndarray) -> Normalization:
    values = np.asarray(sequences, dtype="float32")
    observed = values[:, PRESENCE_CHANNEL, :] > 0.5
    if not observed.any():
        raise ValueError("Training sequences have no observed months")
    center = np.zeros(len(CHANNELS), dtype="float32")
    scale = np.ones(len(CHANNELS), dtype="float32")
    for channel in CONTINUOUS_CHANNELS:
        channel_values = values[:, channel, :][observed]
        channel_values = channel_values[np.isfinite(channel_values)]
        if not len(channel_values):
            raise ValueError(f"No finite values for channel {CHANNELS[channel]}")
        center[channel] = np.median(channel_values)
        q25, q75 = np.quantile(channel_values, [0.25, 0.75])
        width = float(q75 - q25)
        if width < 1e-4:
            width = float(np.std(channel_values))
        scale[channel] = width if width >= 1e-4 else 1.0
    return Normalization(center=center, scale=scale)


class ResidualTCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.convolution1 = nn.Conv1d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.normalization1 = nn.GroupNorm(4, width)
        self.convolution2 = nn.Conv1d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.normalization2 = nn.GroupNorm(4, width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.convolution1(values)
        values = self.normalization1(values)
        values = self.dropout(self.activation(values))
        values = self.convolution2(values)
        values = self.normalization2(values)
        return self.activation(residual + self.dropout(values))


class TinyTemporalTCN(nn.Module):
    """Small sequence classifier with a 61-month nominal receptive field."""

    def __init__(
        self,
        input_channels: int = len(CHANNELS),
        width: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if width % 4:
            raise ValueError("TCN width must be divisible by four")
        self.projection = nn.Conv1d(input_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualTCNBlock(width, int(dilation), dropout)
            for dilation in dilations
        ])
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.projection(values))
        return torch.cat([values.mean(dim=-1), values.amax(dim=-1)], dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(values)).squeeze(1)


def _fit_model(
    meta: pd.DataFrame,
    sequences: np.ndarray,
    excluded_countries: Iterable[str],
    epochs: int,
    seed: int,
    device: str | torch.device | None,
    batch_size: int = 512,
    unlabeled_keep_fraction: float = 1.0,
) -> tuple[TinyTemporalTCN, Normalization, dict]:
    excluded = tuple(sorted(set(str(value) for value in excluded_countries)))
    if HOLDOUT in excluded or not set(excluded).issubset(COUNTRIES):
        raise ValueError(f"Invalid country exclusion: {excluded}")
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs and batch_size must be positive")
    train_mask = _bag_training_mask(
        meta, excluded, unlabeled_keep_fraction, seed
    )
    labels = meta.loc[train_mask, "is_eog_flare"].to_numpy(dtype="float32")
    if len(labels) < 2 or len(np.unique(labels)) != 2:
        raise ValueError(f"Both labels are required after excluding {excluded}")
    training_sequences = sequences[train_mask]
    normalization = _fit_normalization(training_sequences)
    training_sequences = normalization.transform(training_sequences)

    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    model = TinyTemporalTCN().to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=1e-3
    )
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=resolved_device)
    )
    dataset = TensorDataset(
        torch.from_numpy(training_sequences), torch.from_numpy(labels)
    )
    generator = torch.Generator().manual_seed(seed)
    training_countries = meta.loc[train_mask, "country"].astype(str).to_numpy()
    country_counts = pd.Series(training_countries).value_counts().sort_index()
    sample_weights = np.asarray(
        [1.0 / country_counts[country] for country in training_countries],
        dtype="float64",
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
    history = []
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(resolved_device)
            batch_labels = batch_labels.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_values), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_values)
            examples += len(batch_values)
        history.append(loss_sum / max(examples, 1))
    diagnostic = {
        "excluded_countries": list(excluded),
        "train_countries": sorted(set(meta.loc[train_mask, "country"])),
        "training_sources": int(train_mask.sum()),
        "training_positives": int(labels.sum()),
        "training_unlabeled": int(len(labels) - labels.sum()),
        "seed": seed,
        "epochs": epochs,
        "batch_size": min(batch_size, len(dataset)),
        "unlabeled_keep_fraction": unlabeled_keep_fraction,
        "device": str(resolved_device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sampler": {
            "policy": "inverse-country-count",
            "replacement": True,
            "draws_per_epoch": len(dataset),
            "country_source_counts": {
                str(key): int(value) for key, value in country_counts.items()
            },
        },
        "loss": history,
        "normalization": normalization.to_dict(),
    }
    return model, normalization, diagnostic


def _predict(
    model: TinyTemporalTCN,
    normalization: Normalization,
    sequences: np.ndarray,
    device: str | torch.device | None,
    batch_size: int = 512,
    return_embeddings: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    values = normalization.transform(sequences)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=min(batch_size, max(len(values), 1)),
        shuffle=False,
        num_workers=0,
    )
    scores = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for (batch_values,) in loader:
            batch_values = batch_values.to(resolved_device)
            embedding = model.encode(batch_values)
            logit = model.head(embedding).squeeze(1)
            scores.append(torch.sigmoid(logit).cpu().numpy())
            if return_embeddings:
                embeddings.append(embedding.cpu().numpy())
    if not scores:
        return np.empty(0, dtype="float32"), np.empty((0, 64), dtype="float32")
    embedding_array = (
        np.concatenate(embeddings).astype("float32")
        if return_embeddings else np.empty((0, 64), dtype="float32")
    )
    return (
        np.concatenate(scores).astype("float32"),
        embedding_array,
    )


def _fit_bagged_scores(
    meta: pd.DataFrame,
    sequences: np.ndarray,
    query_indices: np.ndarray,
    excluded_countries: Sequence[str],
    epochs: int,
    seed: int,
    device: str | torch.device | None,
    batch_size: int,
    pu_bags: int,
    unlabeled_keep_fraction: float,
) -> tuple[np.ndarray, list[dict]]:
    if pu_bags < 1:
        raise ValueError("pu_bags must be at least one")
    score = np.zeros(len(query_indices), dtype="float64")
    diagnostics = []
    for bag_index in range(pu_bags):
        bag_seed = seed + bag_index * 101
        model, normalization, diagnostic = _fit_model(
            meta, sequences, excluded_countries, epochs, bag_seed, device,
            batch_size, unlabeled_keep_fraction,
        )
        bag_score, _ = _predict(
            model, normalization, sequences[query_indices], device,
            return_embeddings=False,
        )
        score += bag_score / pu_bags
        diagnostic["bag_index"] = bag_index
        diagnostic["pu_bags"] = pu_bags
        diagnostics.append(diagnostic)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return score.astype("float32"), diagnostics


def precompute_nested_scores(
    meta: pd.DataFrame,
    sequences: np.ndarray,
    cohort: pd.DataFrame,
    epochs: int = 24,
    seed: int = 5107,
    device: str | torch.device | None = None,
    batch_size: int = 512,
    pu_bags: int = 2,
    unlabeled_keep_fraction: float = 0.8,
) -> dict:
    """Produce strict one-country population and two-country cohort scores."""
    cohort = _validate_cohort(cohort)
    if len(meta) != len(sequences) or not meta.source_id.is_unique:
        raise ValueError("meta and sequences must be aligned and source IDs unique")
    if HOLDOUT in set(meta.country) or not set(meta.country).issubset(COUNTRIES):
        raise ValueError("Temporal data contains a forbidden country")
    position = pd.Series(np.arange(len(meta), dtype="int64"), index=meta.source_id)
    if not set(cohort.source_id).issubset(position.index):
        raise ValueError("All cohort sources must be present in temporal data")

    single_maps = {}
    pair_maps = {}
    single_frames = []
    pair_frames = []
    diagnostics = []
    population_complete = bool(meta.population_complete.all()) if (
        "population_complete" in meta
    ) else False
    for country_index, country in enumerate(COUNTRIES):
        fold_seed = seed + 1000 * (country_index + 1)
        if population_complete:
            query = meta.loc[
                meta.country.eq(country),
                ["source_id", "country", "block_id", "is_eog_flare", "is_cohort"],
            ].copy()
        else:
            query = cohort.loc[
                cohort.country.eq(country), ["source_id", "country"]
            ].copy()
            query["is_cohort"] = True
        indices = position.loc[query.source_id].to_numpy(dtype="int64")
        score, fold_diagnostics = _fit_bagged_scores(
            meta, sequences, indices, [country], epochs, fold_seed, device,
            batch_size, pu_bags, unlabeled_keep_fraction,
        )
        query["temporal_score"] = score
        query["excluded_country"] = country
        single_frames.append(query)
        cohort_query = query.loc[query.is_cohort]
        single_maps[country] = dict(zip(
            cohort_query.source_id, cohort_query.temporal_score.astype(float)
        ))
        for diagnostic in fold_diagnostics:
            diagnostic["kind"] = "single"
            diagnostic["scored_sources"] = len(query)
            diagnostic["scored_cohort_sources"] = len(cohort_query)
            diagnostic["score_population"] = population_complete
            diagnostic["embedding_dimension"] = 0
            diagnostics.append(diagnostic)

    for pair_index, pair in enumerate(combinations(COUNTRIES, 2)):
        fold_seed = seed + 100000 + 1000 * (pair_index + 1)
        query = cohort.loc[
            cohort.country.isin(pair), ["source_id", "country"]
        ].copy()
        indices = position.loc[query.source_id].to_numpy(dtype="int64")
        score, fold_diagnostics = _fit_bagged_scores(
            meta, sequences, indices, pair, epochs, fold_seed, device,
            batch_size, pu_bags, unlabeled_keep_fraction,
        )
        query["temporal_score"] = score
        query["excluded_country_a"] = pair[0]
        query["excluded_country_b"] = pair[1]
        pair_frames.append(query)
        pair_maps[pair] = dict(zip(query.source_id, score.astype(float)))
        for diagnostic in fold_diagnostics:
            diagnostic["kind"] = "pair"
            diagnostic["scored_sources"] = len(query)
            diagnostic["embedding_dimension"] = 0
            diagnostics.append(diagnostic)

    single_frame = pd.concat(single_frames, ignore_index=True)
    pair_frame = pd.concat(pair_frames, ignore_index=True)
    if not single_frame.source_id.is_unique:
        raise ValueError("Single-country scores must cover each aligned source once")
    expected_single = len(meta) if population_complete else len(cohort)
    if len(single_frame) != expected_single:
        raise ValueError(
            f"Expected {expected_single} strict single-country scores; got {len(single_frame)}"
        )
    return {
        "single_scores": single_maps,
        "pair_scores": pair_maps,
        "single_score_frame": single_frame,
        "pair_score_frame": pair_frame,
        "diagnostics": diagnostics,
        "single_score_population": population_complete,
        "pu_bags": pu_bags,
        "unlabeled_keep_fraction": unlabeled_keep_fraction,
        "protocol": PROTOCOL_VERSION,
    }


def fit_final_tcn(
    meta: pd.DataFrame,
    sequences: np.ndarray,
    output_dir: str | Path,
    epochs: int = 24,
    seed: int = 6101,
    device: str | torch.device | None = None,
    batch_size: int = 512,
    pu_bags: int = 2,
    unlabeled_keep_fraction: float = 0.8,
) -> dict:
    """Fit PU bags on all sampled foreign sources and save artifacts."""
    if pu_bags < 1:
        raise ValueError("pu_bags must be at least one")
    required = {"source_id", "country", "is_eog_flare", "sample_role", "train_selected"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Temporal metadata is missing columns: {sorted(missing)}")
    if len(meta) != len(sequences) or not meta.source_id.is_unique:
        raise ValueError("meta and sequences must be aligned with unique source IDs")
    countries = set(meta.country.astype(str))
    if HOLDOUT in countries or not countries.issubset(COUNTRIES):
        raise ValueError("Final TCN training data contains a forbidden country")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_signature = hashlib.sha256(
        meta.loc[meta.train_selected, [
            "source_id", "country", "is_eog_flare", "sample_role"
        ]].sort_values(["country", "source_id"]).to_csv(index=False).encode("utf-8")
    ).hexdigest()
    artifact_records = []
    artifact_paths = []
    for bag_index in range(pu_bags):
        bag_seed = seed + bag_index * 101
        model, normalization, diagnostic = _fit_model(
            meta, sequences, [], epochs, bag_seed, device, batch_size,
            unlabeled_keep_fraction,
        )
        artifact_path = output_dir / f"10_final_temporal_tcn_bag{bag_index}.pt"
        artifact = {
            "protocol": PROTOCOL_VERSION,
            "holdout": HOLDOUT,
            "holdout_loaded": False,
            "countries": list(COUNTRIES),
            "window_years": list(WINDOW_YEARS),
            "channels": list(CHANNELS),
            "architecture": DEFAULT_ARCHITECTURE,
            "normalization": normalization.to_dict(),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "seed": bag_seed,
            "epochs": epochs,
            "bag_index": bag_index,
            "pu_bags": pu_bags,
            "unlabeled_keep_fraction": unlabeled_keep_fraction,
            "training_source_sha256": training_signature,
        }
        torch.save(artifact, artifact_path)
        artifact_hash = _file_hash(artifact_path)
        artifact_paths.append(str(artifact_path))
        artifact_records.append({
            "artifact": artifact_path.name,
            "artifact_sha256": artifact_hash,
            "bag_index": bag_index,
            "seed": bag_seed,
            "training": diagnostic,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = {
        "protocol": PROTOCOL_VERSION,
        "holdout": HOLDOUT,
        "holdout_loaded": False,
        "countries": list(COUNTRIES),
        "window_years": list(WINDOW_YEARS),
        "channels": list(CHANNELS),
        "architecture": DEFAULT_ARCHITECTURE,
        "ensemble": "mean_probability",
        "pu_bags": pu_bags,
        "unlabeled_keep_fraction": unlabeled_keep_fraction,
        "base_seed": seed,
        "epochs": epochs,
        "training_source_sha256": training_signature,
        "artifacts": artifact_records,
    }
    config_path = output_dir / "10_final_temporal_tcn.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {
        "artifact_path": artifact_paths[0],
        "artifact_paths": artifact_paths,
        "config_path": str(config_path),
        "artifact_sha256": artifact_records[0]["artifact_sha256"],
        "artifact_sha256s": [record["artifact_sha256"] for record in artifact_records],
        "config": config,
    }


def load_tcn_artifact(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[TinyTemporalTCN, Normalization, dict]:
    """Load a saved final model with its exact fold-safe normalization."""
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        artifact.get("protocol") != PROTOCOL_VERSION
        or artifact.get("holdout_loaded") is not False
        or artifact.get("holdout") != HOLDOUT
    ):
        raise ValueError("Incompatible or holdout-contaminated temporal artifact")
    if artifact.get("channels") != list(CHANNELS):
        raise ValueError("Temporal artifact channel order does not match")
    model = TinyTemporalTCN().to(_resolve_device(device))
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    normalizer = Normalization(
        center=np.asarray(artifact["normalization"]["center"], dtype="float32"),
        scale=np.asarray(artifact["normalization"]["scale"], dtype="float32"),
    )
    return model, normalizer, artifact


def score_tcn_artifact(
    path: str | Path,
    sequences: np.ndarray,
    device: str | torch.device | None = None,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Score sequences, optionally returning the 64D pooled embedding."""
    model, normalization, _ = load_tcn_artifact(path, device)
    return _predict(
        model, normalization, sequences, device,
        return_embeddings=return_embeddings,
    )


def score_tcn_ensemble(
    paths: Sequence[str | Path],
    sequences: np.ndarray,
    device: str | torch.device | None = None,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average saved PU-bag probabilities, optionally averaging embeddings."""
    if not paths:
        raise ValueError("At least one temporal artifact is required")
    score = np.zeros(len(sequences), dtype="float64")
    embedding_sum = None
    for path in paths:
        model, normalization, _ = load_tcn_artifact(path, device)
        bag_score, bag_embedding = _predict(
            model, normalization, sequences, device,
            return_embeddings=return_embeddings,
        )
        score += bag_score / len(paths)
        if return_embeddings:
            if embedding_sum is None:
                embedding_sum = np.zeros_like(bag_embedding, dtype="float64")
            embedding_sum += bag_embedding / len(paths)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    embeddings = (
        embedding_sum.astype("float32")
        if embedding_sum is not None else np.empty((0, 64), dtype="float32")
    )
    return score.astype("float32"), embeddings
