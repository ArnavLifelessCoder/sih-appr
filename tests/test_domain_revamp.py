from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_05e_domain_revamp as revamp


def test_country_ranks_remove_between_country_scale_without_using_labels():
    frame = pd.DataFrame({
        "country": ["A", "A", "B", "B"],
        "frp_mean": [1.0, 2.0, 100.0, 200.0],
        "is_eog_flare": [0, 1, 1, 0],
    })
    ranked, columns = revamp.add_country_ranks(frame, ["frp_mean"])
    assert columns == ["frp_mean"]
    assert ranked.frp_mean_country_pct.tolist() == [0.5, 1.0, 0.5, 1.0]
    changed = frame.copy()
    changed["is_eog_flare"] = 1 - changed.is_eog_flare
    changed_ranked, _ = revamp.add_country_ranks(changed, ["frp_mean"])
    pd.testing.assert_series_equal(
        ranked.frp_mean_country_pct, changed_ranked.frp_mean_country_pct
    )


def test_country_ranks_forbid_india():
    frame = pd.DataFrame({
        "country": ["India"], "frp_mean": [1.0], "is_eog_flare": [0]
    })
    with pytest.raises(ValueError, match="India is forbidden"):
        revamp.add_country_ranks(frame, ["frp_mean"])


def test_raw_schema_does_not_absorb_generated_rank_columns():
    frame = pd.DataFrame({
        "frp_mean": [1.0], "frp_mean_country_pct": [1.0],
        "active_days_per_year": [1.0],
    })
    schemas = revamp.feature_schemas(
        frame, ["frp_mean", "active_days_per_year"]
    )
    assert schemas["raw"] == ["frp_mean", "active_days_per_year"]
    assert "frp_mean" not in schemas["ranked"]
    assert "frp_mean_country_pct" in schemas["ranked"]


def test_persistent_training_filter_is_train_only(monkeypatch):
    frame = pd.DataFrame({
        "f": np.arange(6, dtype=float), "n_days": [1, 2, 3, 1, 4, 5],
        "is_eog_flare": [0, 1, 0, 0, 1, 0],
    })
    observed = {}

    class FakeModel:
        pass

    def fake_train(params, dataset, num_boost_round):
        observed["rows"] = dataset.data.shape[0]
        observed["labels"] = dataset.label.tolist()
        return FakeModel()

    monkeypatch.setattr(revamp.lgb, "train", fake_train)
    model, used = revamp._model(
        frame, ["f"], np.arange(6), rounds=2, seed=1, persistent_train=True
    )
    assert isinstance(model, FakeModel)
    assert used == observed["rows"] == 4
    assert observed["labels"] == [1, 0, 1, 0]
