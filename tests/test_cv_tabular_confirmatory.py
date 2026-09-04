import numpy as np
import pandas as pd

import kg_12b_confirmatory as confirmatory


def test_review_budget_breaks_score_ties_by_source_id():
    predictions = pd.DataFrame({
        "source_id": ["z", "a", "m", "b", "c"],
        "country": ["A"] * 5,
        "is_eog_flare": [0, 1, 0, 0, 0],
        "score_compact_tabular": [0.5] * 5,
    })
    metrics, membership = confirmatory.review_budget_tables(
        predictions, (confirmatory.BASELINE,)
    )
    selected = membership.loc[membership.review_fraction.eq(0.20)]
    assert selected.source_id.tolist() == ["a"]
    row = metrics.loc[metrics.review_fraction.eq(0.20)].iloc[0]
    assert row.positive_found == 1


def test_ensemble_predictions_average_every_frozen_seed():
    rows = []
    for seed, offset in zip(confirmatory.CONFIRMATORY_SEEDS, (0.0, 0.1, 0.2)):
        for source_id, base in (("a", 0.2), ("b", 0.6)):
            rows.append({
                "seed": seed,
                "source_id": source_id,
                "country": "A",
                "is_eog_flare": int(source_id == "b"),
                "eog_flare_id": None,
                "block_id": source_id,
                "score_compact_tabular": base + offset,
                "score_guarded_cv_tabular": base + offset / 2,
                "score_cv_multispectral": base,
                "score_all77_tabular": base,
            })
    ensemble = confirmatory._ensemble_predictions(pd.DataFrame(rows))
    score = ensemble.set_index("source_id").score_compact_tabular
    assert np.isclose(score.loc["a"], 0.3)
    assert np.isclose(score.loc["b"], 0.7)


def test_confirmatory_policy_averages_models_and_keeps_missing_image_fallback():
    frame = pd.DataFrame({"country": ["A", "A"]})
    seed_compact = np.array([[0.1, 0.3], [0.7, 0.9]])
    compact_models = np.stack(
        [seed_compact, seed_compact + 0.02, seed_compact - 0.02], axis=1
    )
    visual_models = np.array([[0.8, 0.7, 0.6], [0.2, 0.3, 0.4]])
    compact_reference = np.column_stack([
        np.linspace(0.05, 0.95, 20),
        np.linspace(0.06, 0.96, 20),
        np.linspace(0.04, 0.94, 20),
    ])
    visual_reference = np.column_stack([
        np.linspace(0.10, 0.90, 20),
        np.linspace(0.11, 0.91, 20),
        np.linspace(0.09, 0.89, 20),
    ])
    alphas = np.array([0.15, 0.30, 0.50])
    available = np.array([True, False])
    visual_models[~available] = np.nan
    output = confirmatory.apply_confirmatory_policy(
        frame,
        compact_models,
        confirmatory.ELIGIBLE_CHALLENGER,
        visual_model_scores=visual_models,
        seed_fusion_alphas=alphas,
        compact_oof_reference=compact_reference,
        visual_oof_reference=visual_reference,
        image_available=available,
    )
    expected_parts = []
    for index in range(3):
        expected_parts.append(confirmatory.apply_deployment_policy(
            frame,
            compact_models[:, index, :].mean(axis=1),
            confirmatory.ELIGIBLE_CHALLENGER,
            fusion_alpha=alphas[index],
            visual_score=visual_models[:, index],
            image_available=available,
            tabular_reference=compact_reference[:, index],
            visual_reference=visual_reference[:, index],
        ))
    expected = np.column_stack(expected_parts).mean(axis=1)
    assert np.allclose(output, expected)


def test_acceptance_rejects_one_nonpositive_fresh_seed():
    prediction_rows = []
    metric_rows = []
    for country_index, country in enumerate(confirmatory.COUNTRIES):
        for row_index, (label, compact, guarded) in enumerate((
            (0, 0.1, 0.1),
            (1, 0.4, 0.8),
            (0, 0.8, 0.2),
            (1, 0.7, 0.9),
        )):
            prediction_rows.append({
                "source_id": f"{country_index}-{row_index}",
                "block_id": f"{country_index}-{row_index}",
                "country": country,
                "is_eog_flare": label,
                "score_compact_tabular": compact,
                "score_guarded_cv_tabular": guarded,
            })
        metric_rows.extend([
            {"country": country, "branch": confirmatory.BASELINE,
             "pr_auc": 0.70, "f1": 0.70},
            {"country": country, "branch": confirmatory.ELIGIBLE_CHALLENGER,
             "pr_auc": 0.74, "f1": 0.72},
        ])
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    budget, _ = confirmatory.review_budget_tables(
        predictions,
        (confirmatory.BASELINE, confirmatory.ELIGIBLE_CHALLENGER),
    )
    seed_summary = pd.DataFrame({
        "seed": confirmatory.CONFIRMATORY_SEEDS,
        "macro_pr_auc_gain": [0.02, -0.001, 0.02],
    })
    decision = confirmatory._acceptance_decision(
        predictions,
        metrics,
        budget,
        seed_summary,
        {seed: 0.3 for seed in confirmatory.CONFIRMATORY_SEEDS},
        {"passed": True},
        bootstrap_repeats=20,
    )
    conditions = {row["name"]: row for row in decision["conditions"]}
    assert not conditions["all_seed_macro_pr_auc_gains_positive"]["passed"]
    assert not decision["passed_all_conditions"]
