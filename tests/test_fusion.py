from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_08_fusion as fusion


def test_review_mask_handles_csv_strings():
    values = pd.Series([True, False, "True", "False", "1", "0", None])
    assert fusion.review_mask(values).tolist() == [
        True, False, True, False, True, False, False
    ]


def test_feature_schemas_exclude_location_and_label_proxies():
    forbidden = {"lat", "lon", "country", "type", "eog_dist_m", "is_eog_flare"}
    for columns in fusion.BRANCHES.values():
        assert forbidden.isdisjoint(columns)
    assert set(fusion.THERMAL_COLS).isdisjoint(fusion.IMAGE_COLS)


def test_inner_oof_holds_out_each_country(monkeypatch):
    frame = pd.DataFrame({
        "country": ["A", "A", "B", "B", "C", "C"],
        "feature": np.arange(6, dtype=float),
        "is_eog_flare": [0, 1, 0, 1, 0, 1],
    })
    fitted_countries = []

    class FakeModel:
        def predict(self, values):
            return np.full(len(values), len(fitted_countries) / 10)

    def fake_train(data, columns, seed, rounds):
        fitted_countries.append(set(data.country))
        return FakeModel()

    monkeypatch.setattr(fusion, "train_model", fake_train)
    scores = fusion.inner_country_oof(frame, ["feature"], seed=7, rounds=2)
    assert fitted_countries == [{"B", "C"}, {"A", "C"}, {"A", "B"}]
    assert scores.tolist() == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]


def test_macro_threshold_is_chosen_from_predictions():
    frame = pd.DataFrame({
        "country": ["A", "A", "B", "B"],
        "is_eog_flare": [0, 1, 0, 1],
    })
    threshold, score = fusion.macro_f1_threshold(
        frame, np.array([0.1, 0.9, 0.2, 0.8])
    )
    assert threshold == 0.8
    assert score == 1.0
