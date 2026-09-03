from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_05c_balanced_tabular as balanced
import kg_05d_nested_tabular as nested
from test_robust_evaluation import small_sources


def test_known_site_blocks_are_merged_without_merging_countries():
    df = small_sources()
    df.loc[4, "eog_flare_id"] = "a"  # Same text in a different country.
    revised, mapping = nested.site_safe_blocks(df)
    assert revised.block_id.iloc[0] == revised.block_id.iloc[1]
    assert revised.block_id.iloc[0] != revised.block_id.iloc[4]
    assert len(mapping) == 8
    assert mapping.evaluation_block.nunique() == 7
    pd.testing.assert_series_equal(df.block_id, small_sources().block_id)


def test_outer_labels_do_not_affect_that_countrys_selection(monkeypatch):
    df = small_sources()
    selections, fits = [], []

    def select(train, cols, **kwargs):
        selections.append(set(train.country))
        name = "unweighted" if train.is_eog_flare.sum() % 2 == 0 else "site_balanced"
        variant = next(v for v in balanced.VARIANTS if v["name"] == name)
        summary = pd.DataFrame([dict(experiment=name, threshold_exact=.4,
                                     macro_country_f1=.5, train_s=0.)])
        return summary, None, None, pd.DataFrame(), variant

    def predict(train, test, cols, variant, **kwargs):
        assert set(train.country).isdisjoint(set(test.country))
        fits.append((set(train.country), set(test.country), kwargs["seed"]))
        return np.full(len(test), .5), 0., 0.

    monkeypatch.setattr(balanced, "evaluate_variants", select)
    monkeypatch.setattr(balanced, "fit_predict", predict)
    monkeypatch.setattr(balanced, "active_eog_counts", lambda _: {"A": 1, "B": 2})
    first = nested.nested_loco(df, ["feature"], inner_splits=2, rounds=1)
    changed = df.copy()
    changed.loc[0, "is_eog_flare"] = 0
    second = nested.nested_loco(changed, ["feature"], inner_splits=2, rounds=1)
    for column in ["model_variant", "threshold_exact", "inner_seed", "fit_seed"]:
        assert first[0].set_index("country").loc["A", column] == second[0].set_index("country").loc["A", column]
    assert selections == [{"B"}, {"A"}, {"B"}, {"A"}]
    assert len(fits) == 4


def test_india_is_rejected():
    df = small_sources()
    df.loc[0, "country"] = "India"
    with pytest.raises(ValueError, match="India is forbidden"):
        nested.validate_features(df, ["feature"])


def test_real_lightgbm_grouped_smoke(monkeypatch):
    monkeypatch.setitem(balanced.PARAMS, "num_threads", 2)
    df, _ = nested.site_safe_blocks(small_sources())
    score, _, _ = balanced.cv_predict(
        df, ["feature"], balanced.VARIANTS[1], n_splits=2, rounds=2, seed=31,
    )
    assert len(score) == len(df)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()


def test_small_end_to_end_run_saves_complete_artifacts(monkeypatch, tmp_path):
    parts = []
    for country in nested.TRAIN_COUNTRIES:
        part = small_sources()
        part["country"] = country
        part["source_id"] = country + "_" + part.source_id
        part["block_id"] = country + "_" + part.block_id
        parts.append(part)
        part.to_parquet(tmp_path / f"features_{country}_{nested.FEATURE_TAG}.parquet")
    df = pd.concat(parts, ignore_index=True)
    monkeypatch.setattr(nested, "CACHE", tmp_path)
    monkeypatch.setattr(nested, "OUT", tmp_path)
    monkeypatch.setattr(nested, "load_features", lambda *args, **kwargs: df)
    monkeypatch.setattr(nested, "robust_feature_cols", lambda _: ["feature"])
    monkeypatch.setattr(balanced, "active_eog_counts", lambda countries: {c: 3 for c in countries})
    monkeypatch.setitem(balanced.PARAMS, "num_threads", 2)
    summary, loco = nested.run(n_splits=2, inner_splits=2, rounds=2)
    manifest = json.loads((tmp_path / "05d_manifest.json").read_text())
    saved = pd.read_parquet(tmp_path / "05d_loco_predictions.parquet")
    assert manifest["status"] == "complete" and not manifest["holdout_loaded"]
    assert len(summary) == 4 and len(loco) == 6
    assert saved.source_id.is_unique and len(saved) == len(df)
    assert (tmp_path / "05d_final_foreign_model.txt").is_file()
    assert manifest["final_lightgbm_params"]["seed"] == 30031
    assert pd.read_csv(tmp_path / "05d_inner_selection.csv").groupby("held_out_country").selected.sum().eq(1).all()
    with pytest.raises(FileExistsError, match="already exists"):
        nested.run(n_splits=2, inner_splits=2, rounds=2)
