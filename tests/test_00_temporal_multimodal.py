import torch
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_10_temporal_tcn as temporal
import kg_11_multimodal as multimodal


def test_temporal_model_is_small_and_shape_safe():
    model = temporal.TinyTemporalTCN()
    count = sum(parameter.numel() for parameter in model.parameters())
    values = torch.zeros(3, len(temporal.CHANNELS), temporal.MONTHS)
    assert count < 100_000
    assert model(values).shape == (3,)
    assert model.encode(values).shape == (3, 64)


def test_positive_block_fragments_are_not_sampled_as_unlabeled():
    frame = pd.DataFrame({
        "source_id": ["p0", "p1", "u_overlap", "u_hard", "u_random"],
        "country": ["Algeria"] * 5,
        "block_id": ["positive", "other_positive", "positive", "u1", "u2"],
        "is_eog_flare": [1, 1, 0, 0, 0],
        "eog_flare_id": ["site0", "site1", None, None, None],
        "active_days_per_year": [5, 4, 9, 3, 1],
        "active_months_per_year": [3, 3, 9, 2, 1],
        "det_per_year": [8, 7, 20, 4, 1],
        "night_frac": [1.0, 1.0, 1.0, 0.5, 0.0],
        "sat_frac": [0.0, 0.0, 0.0, 0.0, 0.0],
        "dt_mir_lwir_mean": [20, 18, 30, 10, 5],
        "frp_mean": [10, 9, 25, 5, 1],
    })
    selected, diagnostic = temporal._sample_country(frame, negative_ratio=1, seed=5)
    assert "u_overlap" not in set(selected.source_id)
    assert diagnostic["positive_block_unlabeled_excluded"] == 1
    assert selected.is_eog_flare.sum() == 2


def test_monthly_sequence_preserves_mir_and_saturation_physics():
    detections = pd.DataFrame({
        "source_id": ["s0", "s0"],
        "acq_dt": pd.to_datetime(["2022-01-03", "2022-01-04"]),
        "frp": [4.0, 6.0],
        "t_mir": [368.0, 360.0],
        "t_lwir": [300.0, 302.0],
        "daynight": ["N", "D"],
        "sensor": ["MODIS", "VIIRS_SNPP"],
        "year": [2022, 2022],
    })
    sequence = temporal._monthly_sequence(detections, ["s0"])
    assert sequence.shape == (1, len(temporal.CHANNELS), temporal.MONTHS)
    assert sequence[0, temporal.CHANNELS.index("mir_mean"), 0] == 364.0
    assert sequence[0, temporal.CHANNELS.index("saturation_fraction"), 0] == 0.5
    assert sequence[0, temporal.CHANNELS.index("modis_fraction"), 0] == 0.5
    assert sequence[0, temporal.PRESENCE_CHANNEL, 0] == 1.0


def test_rgb_conversion_uses_b4_b3_b2_and_fills_nan():
    image = np.zeros((6, 2, 2), dtype="float32")
    image[0] = 0.1
    image[1] = 0.2
    image[2] = 0.3
    image[2, 0, 0] = np.nan
    rgb = multimodal._fill_rgb(image)
    assert rgb.shape == (3, 2, 2)
    assert np.isfinite(rgb).all()
    assert rgb[0, 1, 1] == np.float32(0.3)
    assert rgb[1, 1, 1] == np.float32(0.2)
    assert rgb[2, 1, 1] == np.float32(0.1)


def test_descriptor_alignment_keeps_only_explicit_ts_features():
    cohort = pd.DataFrame({"source_id": ["a", "b"]})
    descriptors = pd.DataFrame({
        "source_id": ["a", "b"],
        "country": ["A", "B"],
        "is_eog_flare": [1, 0],
        "sample_role": ["positive_site", "cohort_only"],
        "train_selected": [True, False],
        "is_cohort": [True, True],
        "ts_active_months": [2, 1],
        "ts_frp_sum": [5.0, 1.0],
    })
    embeddings = pd.DataFrame({
        "source_id": ["a", "b"], "ssl_0000": [0.1, 0.2]
    })
    aligned, matrix, columns = multimodal._aligned_extra(
        cohort, descriptors, embeddings
    )
    assert columns == ["ts_active_months", "ts_frp_sum"]
    assert list(aligned.columns) == ["source_id", *columns]
    assert matrix.shape == (2, 1)


def _selection_frame() -> pd.DataFrame:
    rows = []
    for country in ["A", "B", "C", "D", "E"]:
        for label in [1, 0, 1, 0]:
            rows.append({"country": country, "is_eog_flare": label})
    return pd.DataFrame(rows)


def test_inner_guard_selects_consistent_gain_and_rejects_bad_tail():
    frame = _selection_frame()
    baseline = np.tile([0.9, 0.8, 0.7, 0.6], 5)
    improved = np.tile([0.9, 0.1, 0.8, 0.2], 5)
    candidates = {"nb9_baseline": baseline, "candidate": improved}
    selected, _, _ = multimodal.choose_inner_candidate(
        frame, candidates, minimum_improved_countries=3,
        maximum_worst_drop=0.04,
    )
    assert selected == "candidate"

    unstable = improved.copy()
    unstable[-4:] = [0.1, 0.9, 0.2, 0.8]
    candidates["candidate"] = unstable
    selected, _, diagnostics = multimodal.choose_inner_candidate(
        frame, candidates, minimum_improved_countries=3,
        maximum_worst_drop=0.04,
    )
    assert selected == "nb9_baseline"
    candidate = next(row for row in diagnostics if row["branch"] == "candidate")
    assert candidate["eligible"] is False


def test_flat_pair_score_lookup_is_source_aligned():
    score = multimodal._lookup_flat_score(
        {"a": 0.2, "b": 0.8}, pd.Series(["b", "a"])
    )
    assert score.tolist() == [0.8, 0.2]


def test_final_tcn_writer_rejects_india_before_fitting(tmp_path):
    meta = pd.DataFrame({
        "source_id": ["india0"],
        "country": ["India"],
        "is_eog_flare": [1],
        "sample_role": ["positive_site"],
        "train_selected": [True],
    })
    sequences = np.zeros(
        (1, len(temporal.CHANNELS), temporal.MONTHS), dtype="float32"
    )
    with pytest.raises(ValueError, match="forbidden country"):
        temporal.fit_final_tcn(meta, sequences, tmp_path, epochs=1, pu_bags=1)


def test_stage1_feature_lineage_rejects_mismatch():
    current = {
        f"features_{country}_2022_2024.parquet": f"hash-{country}"
        for country in multimodal.COUNTRIES
    }
    multimodal.validate_stage1_feature_hashes(
        {"input_sha256": current}, {"input_sha256": current}
    )
    changed = dict(current)
    changed["features_Angola_2022_2024.parquet"] = "different"
    with pytest.raises(ValueError, match="Angola"):
        multimodal.validate_stage1_feature_hashes(
            {"input_sha256": current}, {"input_sha256": changed}
        )
