from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "kaggle"))
import kg_13_india_guarded as india_guarded


def india_frame(rows: int = 90) -> pd.DataFrame:
    positive_rows = 30
    frame = pd.DataFrame({
        "source_id": [f"india_{index:03d}" for index in range(rows)],
        "country": ["India"] * rows,
        "block_id": [f"block_{index:03d}" for index in range(rows)],
        "lat": np.linspace(8.0, 28.0, rows),
        "lon": np.linspace(72.0, 88.0, rows),
        "is_eog_flare": [1] * positive_rows + [0] * (rows - positive_rows),
        "eog_flare_id": [
            *[f"eog_{index:03d}" for index in range(positive_rows)],
            *([np.nan] * (rows - positive_rows)),
        ],
    })
    for index, column in enumerate(
        india_guarded.THERMAL_RAW + india_guarded.THERMAL_RANK
    ):
        frame[column] = np.linspace(0.01 + index, 1.0 + index, rows)
    return frame


def test_panel_selection_is_deterministic_and_spatially_separated():
    frame = india_frame()
    first = india_guarded.select_panel(
        frame, n_sources=60, positive_site_quota=20, seed=123
    )
    second = india_guarded.select_panel(
        frame.sample(frac=1, random_state=99),
        n_sources=60,
        positive_site_quota=20,
        seed=123,
    )
    assert first.source_id.tolist() == second.source_id.tolist()
    assert first.is_eog_flare.sum() == 20
    assert first.block_id.is_unique
    assert first.source_id.is_unique
    assert first.chip_id.is_unique
    assert first.batch_order.tolist() == list(range(60))
    assert first.iloc[:9].is_eog_flare.tolist() == [1, 0, 0] * 3


def test_panel_rejects_non_india_and_missing_positive_site_ids():
    frame = india_frame()
    frame.loc[0, "country"] = "Iraq"
    with pytest.raises(ValueError, match="India only"):
        india_guarded.select_panel(frame, n_sources=60, positive_site_quota=20)


def test_prepare_and_offline_progress_round_trip(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    india_frame().to_parquet(
        input_root / india_guarded.FEATURE_FILE, index=False
    )
    panel = india_guarded.prepare(
        input_root,
        output_root,
        n_sources=60,
        positive_site_quota=20,
        seed=123,
    )
    repeated = india_guarded.prepare(
        input_root,
        output_root,
        n_sources=60,
        positive_site_quota=20,
        seed=123,
    )
    assert panel.source_id.tolist() == repeated.source_id.tolist()
    manifest = india_guarded.run_batch(
        output_root, input_root, max_new=0, max_minutes=1, offline=True
    )
    assert manifest.status.eq("pending").all()
    image, quality = india_guarded.export_image_features(output_root)
    assert len(image) == len(panel)
    assert len(quality) == len(panel)
    assert quality.review_reflectance_tail.eq(False).all()
    with pytest.raises(RuntimeError, match="pending chips"):
        india_guarded.score_frozen_model(output_root, Path(__file__).parents[1])
    frame = india_frame()
    frame.loc[0, "eog_flare_id"] = np.nan
    with pytest.raises(ValueError, match="positive source"):
        india_guarded.select_panel(frame, n_sources=60, positive_site_quota=20)


def test_review_budgets_use_fixed_fractions_and_source_ties():
    frame = india_frame(rows=60)[
        ["source_id", "is_eog_flare", "eog_flare_id"]
    ].copy()
    frame["score_compact_tabular"] = np.linspace(0.0, 1.0, len(frame))
    frame["score_guarded_cv_tabular"] = np.linspace(1.0, 0.0, len(frame))
    output = india_guarded._review_budget_metrics(
        frame, ("score_compact_tabular", "score_guarded_cv_tabular")
    )
    assert len(output) == 6
    assert set(output.review_fraction) == {0.1, 0.2, 0.3}
    assert output.positive_row_recall.between(0, 1).all()
    assert output.positive_site_recall.between(0, 1).all()


def test_review_mask_does_not_treat_false_string_as_true():
    observed = india_guarded._review_mask(
        pd.Series(["True", "False", "1", "0", None])
    )
    assert observed.tolist() == [True, False, True, False, False]


def test_spatial_bootstrap_is_seeded_and_finite():
    frame = india_frame(rows=60)[
        ["source_id", "block_id", "is_eog_flare", "eog_flare_id"]
    ].copy()
    rng = np.random.default_rng(4)
    frame["score_compact_tabular"] = rng.uniform(size=len(frame))
    frame["score_guarded_cv_tabular"] = rng.uniform(size=len(frame))
    first = india_guarded._block_bootstrap(frame, repeats=50, seed=22)
    second = india_guarded._block_bootstrap(frame, repeats=50, seed=22)
    assert first == second
    assert np.isfinite([first["mean"], first["lower_95"], first["upper_95"]]).all()
    assert first["lower_95"] <= first["upper_95"]


def test_india_decision_enforces_both_precommitted_guards():
    metrics = pd.DataFrame({
        "branch": ["compact_tabular", "guarded_cv_tabular"],
        "eog_proxy_pr_auc": [0.50, 0.495],
    })
    review = pd.DataFrame({
        "branch": ["compact_tabular", "guarded_cv_tabular"],
        "review_fraction": [0.20, 0.20],
        "positive_sites_found": [15, 14],
    })
    passed = india_guarded._india_decision(metrics, review)
    assert passed["passed"]
    assert passed["india_ranking_branch"] == "guarded_cv_tabular"
    review.loc[1, "positive_sites_found"] = 13
    failed = india_guarded._india_decision(metrics, review)
    assert not failed["passed"]
    assert failed["india_ranking_branch"] == "compact_tabular"


def test_frozen_nb12b_artifacts_match_recorded_hashes():
    repo_root = Path(__file__).parents[1]
    models, manifest, schema = india_guarded._verified_model_inputs(repo_root)
    assert models.is_dir()
    assert manifest["selected_branch"] == "guarded_cv_tabular"
    assert schema["threshold_is_deployment_calibrated"] is False
    assert len(schema["model_ensemble"]["seed_pipelines"]) == 3


def test_frozen_ensemble_inference_contract_is_executable():
    repo_root = Path(__file__).parents[1]
    models, _, schema = india_guarded._verified_model_inputs(repo_root)
    rng = np.random.default_rng(13)
    rows = 8
    tabular = rng.normal(size=(rows, len(schema["tabular_columns"]))).astype("float32")
    embeddings = rng.normal(
        size=(rows, len(schema["embedding_columns"]))
    ).astype("float32")
    auxiliary = rng.normal(
        size=(
            rows,
            len(schema["image_columns"]) + len(schema["morphology_columns"]),
        )
    ).astype("float32")
    compact = np.empty((rows, 3, 3), dtype="float64")
    visual = np.empty((rows, 3), dtype="float64")
    for seed_index, pipeline in enumerate(
        schema["model_ensemble"]["seed_pipelines"]
    ):
        for model_index, name in enumerate(pipeline["compact_models"]):
            compact[:, seed_index, model_index] = lgb.Booster(
                model_file=str(models / name)
            ).predict(tabular)
        visual[:, seed_index] = joblib.load(
            models / pipeline["visual_model"]
        ).predict(embeddings, auxiliary)
    with np.load(
        models / schema["model_ensemble"]["calibration_file"],
        allow_pickle=False,
    ) as calibration:
        score = india_guarded.apply_confirmatory_policy(
            pd.DataFrame({"source_id": range(rows), "country": ["India"] * rows}),
            compact,
            "guarded_cv_tabular",
            visual_model_scores=visual,
            seed_fusion_alphas=calibration["fusion_alpha"],
            compact_oof_reference=calibration["compact_oof"],
            visual_oof_reference=calibration["visual_oof"],
            image_available=np.ones(rows, dtype=bool),
        )
    assert score.shape == (rows,)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
