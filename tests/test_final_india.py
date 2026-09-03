from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_09_final_india as final


def manifest():
    return {
        "protocol": "05d-nested-v1",
        "status": "complete",
        "holdout_country": "India",
        "holdout_loaded": False,
        "selected_variant": {"name": "unweighted"},
        "features": ["f1", "f2"],
        "final_threshold": 0.4,
    }


def india_frame():
    return pd.DataFrame({
        "source_id": ["i0", "i1", "i2", "i3"],
        "block_id": ["b0", "b1", "b2", "b3"],
        "country": ["India"] * 4,
        "lat": [1.0, 2.0, 3.0, 4.0],
        "lon": [5.0, 6.0, 7.0, 8.0],
        "is_eog_flare": [0, 1, 0, 1],
        "eog_flare_id": [np.nan, "e1", np.nan, "e2"],
        "f1": [0.0, 1.0, 0.2, 0.8],
        "f2": [1.0, 0.0, 0.8, 0.2],
    })


def tiny_model(frame):
    params = {
        "objective": "binary", "verbose": -1, "num_threads": 1,
        "min_data_in_leaf": 1, "num_leaves": 3,
    }
    return lgb.train(
        params,
        lgb.Dataset(frame[["f1", "f2"]], label=frame.is_eog_flare),
        num_boost_round=3,
    )


def test_manifest_rejects_holdout_leak_and_forbidden_feature():
    bad = manifest()
    bad["holdout_loaded"] = True
    with pytest.raises(ValueError, match="untouched holdout"):
        final.validate_nb5_manifest(bad)
    bad = manifest()
    bad["features"] = ["f1", "country"]
    with pytest.raises(ValueError, match="Forbidden"):
        final.validate_nb5_manifest(bad)


def test_frozen_scoring_and_metrics_are_complete():
    frame = india_frame()
    predictions = final.score_features(frame, tiny_model(frame), manifest(), batch_size=2)
    assert len(predictions) == len(frame)
    assert predictions.source_id.is_unique
    assert predictions.eog_like_score.between(0, 1).all()
    assert predictions.score_rank.sort_values().tolist() == [1, 2, 3, 4]
    assert (
        predictions.predicted_eog_like
        == predictions.eog_like_score.ge(manifest()["final_threshold"])
    ).all()
    summary, sites = final.evaluate_frozen_predictions(predictions, 0.4, 3)
    assert summary.loc[0, "n"] == 4
    assert sites.loc[0, "eog_sites_recoverable"] == 2
    assert sites.loc[0, "eog_sites_total"] == 3


def test_feature_validation_rejects_non_india_and_infinity():
    frame = india_frame()
    frame.loc[0, "country"] = "Iraq"
    with pytest.raises(ValueError, match="another country"):
        final.validate_india_features(frame, ["f1", "f2"])
    frame = india_frame()
    frame.loc[0, "f1"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        final.validate_india_features(frame, ["f1", "f2"])
