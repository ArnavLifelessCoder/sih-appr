from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))

from kg_eval import best_f1_threshold
from kg_05c_balanced_tabular import macro_f1_threshold, sample_weights
import kg_05c_balanced_tabular as balanced


def small_sources():
    return pd.DataFrame({
        "source_id": [f"s{i}" for i in range(8)],
        "country": ["A"] * 4 + ["B"] * 4,
        "block_id": [f"b{i}" for i in range(8)],
        "lat": np.zeros(8), "lon": np.zeros(8),
        "is_eog_flare": [1, 1, 0, 0, 1, 1, 0, 0],
        "eog_flare_id": ["a", "a", None, None, "b", "c", None, None],
        "feature": np.arange(8, dtype=float),
    })


def test_cv_weights_use_training_rows_only(monkeypatch):
    df = small_sources()
    splits = [(np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
              (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))]
    calls = []
    original = balanced.sample_weights

    def weights(part, **kwargs):
        calls.append(part.source_id.tolist())
        return original(part, **kwargs)

    class Model:
        def predict(self, x):
            return np.full(len(x), .5)

    monkeypatch.setattr(balanced, "grouped_folds", lambda *args: splits)
    monkeypatch.setattr(balanced, "sample_weights", weights)
    monkeypatch.setattr(balanced, "_train", lambda *args, **kwargs: Model())
    balanced.cv_predict(df, ["feature"], balanced.VARIANTS[1], n_splits=2)
    assert calls == [df.iloc[train].source_id.tolist() for train, _ in splits]


def test_weighting_variants_use_matched_seeds(monkeypatch):
    df = small_sources()
    seeds = []

    def predict(part, cols, variant, **kwargs):
        seeds.append(kwargs["seed"])
        return np.where(part.is_eog_flare, .8, .2), 0., 0.

    monkeypatch.setattr(balanced, "cv_predict", predict)
    monkeypatch.setattr(balanced, "active_eog_counts", lambda _: {"A": 1, "B": 2})
    balanced.evaluate_variants(df, ["feature"], n_splits=2, rounds=1)
    assert len(seeds) == 4 and len(set(seeds)) == 1


def test_exact_f1_threshold_matches_brute_force():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0], dtype="int8")
    score = np.array([0.01, 0.02, 0.21, 0.22, 0.23, 0.80, 0.81, 0.90])
    threshold, f1 = best_f1_threshold(y, score)
    brute_force = max(
        f1_score(y, score >= candidate, zero_division=0)
        for candidate in np.unique(score)
    )
    assert threshold in score
    assert f1 == brute_force


def test_macro_threshold_is_valid_and_finite():
    y = np.array([1, 0, 0, 1, 0, 0], dtype="int8")
    score = np.array([0.8, 0.7, 0.1, 0.3, 0.2, 0.1])
    country = np.array(["A", "A", "A", "B", "B", "B"])
    threshold, macro_f1 = macro_f1_threshold(y, score, country)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= macro_f1 <= 1.0


def test_fragment_balancing_preserves_positive_mass_by_country():
    df = pd.DataFrame({
        "country": ["A"] * 6 + ["B"] * 4,
        "is_eog_flare": [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        "eog_flare_id": ["x", "x", "x", "y", None, None,
                         "m", "n", None, None],
    })
    unbalanced = sample_weights(df, fragment_balance=False)
    balanced = sample_weights(df, fragment_balance=True)

    assert np.isclose(unbalanced.mean(), 1.0)
    assert np.isclose(balanced.mean(), 1.0)

    positive_a = df.country.eq("A") & df.is_eog_flare.eq(1)
    x = positive_a & df.eog_flare_id.eq("x")
    y = positive_a & df.eog_flare_id.eq("y")
    assert np.isclose(balanced[x].sum(), balanced[y].sum())
