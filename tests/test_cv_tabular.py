import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import kg_12_cv_tabular as cv_tabular


def test_finite_rgb_uses_sentinel_band_order_and_fills_nan():
    image = np.zeros((6, 4, 4), dtype="float32")
    image[0] = 0.1
    image[1] = 0.2
    image[2] = 0.3
    image[2, 0, 0] = np.nan
    rgb = cv_tabular._finite_rgb(image)
    assert rgb.shape == (3, 4, 4)
    assert np.isfinite(rgb).all()
    assert np.isclose(rgb[0, 1, 1], 0.3)
    assert np.isclose(rgb[1, 1, 1], 0.2)
    assert np.isclose(rgb[2, 1, 1], 0.1)


def test_morphology_features_are_finite_on_valid_chip():
    size = 40
    image = np.full((6, size, size), 0.2, dtype="float32")
    image[3, 10:30, 10:30] = 0.5
    image[4, 15:25, 15:25] = 0.6
    valid = np.ones((size, size), dtype=bool)
    worldcover = np.full((size, size), 30, dtype="uint8")
    worldcover[16:24, 16:24] = 50
    features = cv_tabular.morphology_features_from_chip(
        image, valid, worldcover, valid
    )
    assert len(features) >= 30
    assert all(name.startswith("morph_") for name in features)
    assert np.isfinite(np.array(list(features.values()), dtype="float64")).all()
    assert features["morph_built_fraction_r25"] > 0


def test_country_percentile_is_computed_within_country():
    frame = pd.DataFrame({"country": ["A", "A", "B", "B"]})
    ranked = cv_tabular.country_percentile(
        frame, np.array([10.0, 20.0, 1000.0, 2000.0])
    )
    assert np.allclose(ranked, [0.5, 1.0, 0.5, 1.0])


def test_zero_alpha_preserves_country_ranking():
    frame = pd.DataFrame({
        "country": ["A"] * 6 + ["B"] * 6,
        "is_eog_flare": [0, 1, 0, 1, 0, 1] * 2,
    })
    tabular = np.array([0.1, 0.9, 0.2, 0.7, 0.3, 0.8] * 2)
    visual = tabular[::-1]
    fused = cv_tabular.guarded_rank_fusion(frame, tabular, visual, alpha=0.0)
    for country, part in frame.groupby("country"):
        index = part.index.to_numpy()
        assert np.isclose(
            average_precision_score(part.is_eog_flare, tabular[index]),
            average_precision_score(part.is_eog_flare, fused[index]),
        )


def test_visual_pu_model_predicts_finite_probabilities():
    rng = np.random.default_rng(12)
    rows = 60
    frame = pd.DataFrame({
        "country": np.repeat(["A", "B", "C"], rows // 3),
        "is_eog_flare": np.tile([0, 0, 1, 0, 1], rows // 5),
    })
    embeddings = rng.normal(size=(rows, 16)).astype("float32")
    aux = rng.normal(size=(rows, 5)).astype("float32")
    aux[0, 0] = np.nan
    model = cv_tabular.VisualPUModel(
        pca_components=6, pu_bags=2, seed=7
    ).fit(frame, embeddings, aux)
    score = model.predict(embeddings[:8], aux[:8])
    assert score.shape == (8,)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()


def test_embedding_views_are_l2_normalised_independently():
    embeddings = np.zeros((2, 1024), dtype="float32")
    embeddings[:, :512] = 2.0
    embeddings[:, 512:] = 7.0
    normalised = cv_tabular.VisualPUModel._normalise_embedding_views(embeddings)
    norms = np.linalg.norm(normalised.reshape(2, 2, 512), axis=2)
    assert np.allclose(norms, 1.0)


def test_checkpoint_hash_is_full_sha256():
    assert len(cv_tabular.CHECKPOINT_SHA256) == 64
    int(cv_tabular.CHECKPOINT_SHA256, 16)


def test_choose_alpha_can_shrink_harmful_visual_branch_to_zero():
    frame = pd.DataFrame({
        "country": np.repeat(["A", "B", "C"], 8),
        "is_eog_flare": np.tile([0, 0, 0, 0, 1, 1, 1, 1], 3),
    })
    tabular = np.tile(np.arange(8, dtype="float64"), 3)
    visual = -tabular
    alpha, score, diagnostics = cv_tabular.choose_alpha(frame, tabular, visual)
    assert alpha == 0.0
    assert len(score) == len(frame)
    assert diagnostics[0]["eligible"]


def test_calibrated_zero_alpha_preserves_each_country_order():
    frame = pd.DataFrame({
        "country": np.repeat(["A", "B", "C"], 8),
        "is_eog_flare": np.tile([0, 0, 0, 0, 1, 1, 1, 1], 3),
    })
    tabular = np.tile(np.linspace(0.05, 0.95, 8), 3)
    visual = np.tile(np.linspace(0.95, 0.05, 8), 3)
    fused = cv_tabular.guarded_rank_fusion(frame, tabular, visual, alpha=0.0)
    for _, part in frame.groupby("country"):
        index = part.index.to_numpy()
        assert np.array_equal(np.argsort(tabular[index]), np.argsort(fused[index]))


def test_inner_branch_guard_keeps_compact_when_challengers_are_harmful():
    frame = pd.DataFrame({
        "country": np.repeat(["A", "B", "C"], 8),
        "is_eog_flare": np.tile([0, 0, 0, 0, 1, 1, 1, 1], 3),
    })
    compact = np.tile(np.arange(8, dtype="float64"), 3)
    harmful = -compact
    selected, diagnostics = cv_tabular.choose_inner_branch(frame, {
        "compact_tabular": compact,
        "all77_tabular": harmful,
        "guarded_cv_tabular": harmful,
    })
    assert selected == "compact_tabular"
    assert len(diagnostics) == 3


def test_deployment_guarded_policy_uses_compact_fallback_on_missing_image():
    frame = pd.DataFrame({"country": ["A", "A", "B", "B"]})
    compact = np.array([0.1, 0.8, 0.2, 0.9])
    visual = np.array([0.8, 0.2, 0.9, 0.1])
    available = np.array([True, False, True, False])
    reference_tabular = np.linspace(0.02, 0.98, 20)
    reference_visual = np.linspace(0.03, 0.97, 20)
    output = cv_tabular.apply_deployment_policy(
        frame,
        compact,
        "guarded_cv_tabular",
        fusion_alpha=0.3,
        visual_score=visual,
        image_available=available,
        tabular_reference=reference_tabular,
        visual_reference=reference_visual,
    )
    fallback = cv_tabular._calibrate_score(reference_tabular, compact)
    assert np.isfinite(output).all()
    assert np.allclose(output[~available], fallback[~available])


def test_all77_policy_uses_positional_country_groups_with_non_range_index():
    frame = pd.DataFrame(
        {"country": ["A", "A", "B", "B"]}, index=[10, 20, 30, 40]
    )
    compact = np.array([0.1, 0.2, 0.3, 0.4])
    all77 = np.array([0.9, 0.8, 0.7, 0.6])
    output = cv_tabular.apply_deployment_policy(
        frame,
        compact,
        "all77_tabular",
        all77_score=all77,
        image_available=np.array([True, True, True, False]),
    )
    assert np.allclose(output[:2], all77[:2])
    assert np.allclose(output[2:], compact[2:])
