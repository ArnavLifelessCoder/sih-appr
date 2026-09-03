"""Stage 05c: country-balanced training and transferable thresholds.

Stage 05b showed that pooled metrics hide severe country imbalance and that a
coarse threshold grid misses useful operating points in the extreme score tail.
This stage keeps the fixed 2022-2024 features and evaluates four predefined
weighting schemes with exact and country-macro threshold selection.

India is never loaded. The selected foreign-only model and threshold are saved
for a later, separately authorized India run.
"""
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from kg_common import *
from kg_eval import *
from kg_05b_robust_tabular import (
    FEATURE_TAG,
    N_ROUNDS,
    PARAMS,
    WINDOW_YEARS,
    robust_feature_cols,
)


VARIANTS = [
    dict(name="unweighted", country_alpha=0.0, fragment_balance=False),
    dict(name="site_balanced", country_alpha=0.0, fragment_balance=True),
    dict(name="sqrt_country_site", country_alpha=0.5, fragment_balance=True),
    dict(name="equal_country_site", country_alpha=1.0, fragment_balance=True),
]


def sample_weights(df, country_alpha=0.0, fragment_balance=False):
    """Create mean-one training weights without using validation labels.

    ``country_alpha=1`` gives every country equal total source weight. A value
    of 0.5 is the square-root compromise between row weighting and full country
    balancing. Fragment balancing gives each EOG site equal positive mass
    within its country while preserving that country's total positive mass.
    """
    counts = df.country.value_counts()
    n_countries = len(counts)
    base = len(df) / (n_countries * counts)
    country_weight = base.pow(float(country_alpha))
    weight = df.country.map(country_weight).to_numpy(dtype="float64")

    if fragment_balance:
        positive = df.is_eog_flare.eq(1) & df.eog_flare_id.notna()
        pos = df.loc[positive, ["country", "eog_flare_id"]].copy()
        fragment_count = pos.groupby(
            ["country", "eog_flare_id"], observed=True
        ).eog_flare_id.transform("size").to_numpy(dtype="float64")
        country_pos = pos.groupby("country", observed=True).eog_flare_id.transform(
            "size"
        ).to_numpy(dtype="float64")
        country_sites = pos.groupby("country", observed=True).eog_flare_id.transform(
            "nunique"
        ).to_numpy(dtype="float64")
        mean_fragments = country_pos / np.maximum(country_sites, 1.0)
        weight[positive.to_numpy()] *= mean_fragments / fragment_count

    weight /= max(float(weight.mean()), 1e-15)
    return weight.astype("float32")


def _train(X, y, train_idx, weights, rounds=N_ROUNDS, seed=31):
    params = dict(PARAMS)
    params.update(
        seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        data_random_seed=seed,
    )
    dataset = lgb.Dataset(
        X[train_idx],
        label=y[train_idx],
        weight=weights[train_idx],
        free_raw_data=True,
    )
    return lgb.train(params, dataset, num_boost_round=rounds)


def cv_predict(df, cols, variant, n_splits=5, rounds=N_ROUNDS, seed=31):
    X = df[cols].to_numpy(dtype="float32")
    y = df.is_eog_flare.to_numpy(dtype="int8")
    weights = sample_weights(
        df,
        country_alpha=variant["country_alpha"],
        fragment_balance=variant["fragment_balance"],
    )
    oof = np.zeros(len(df), dtype="float64")
    train_s = infer_s = 0.0
    for fold, (train_idx, test_idx) in enumerate(grouped_folds(df, n_splits)):
        t0 = time.time()
        model = _train(
            X, y, train_idx, weights, rounds=rounds,
            seed=seed + fold * 100,
        )
        train_s += time.time() - t0
        t0 = time.time()
        oof[test_idx] = model.predict(X[test_idx])
        infer_s += time.time() - t0
    return oof, train_s, infer_s


def fit_predict(train_df, test_df, cols, variant, rounds=N_ROUNDS, seed=31):
    X_train = train_df[cols].to_numpy(dtype="float32")
    X_test = test_df[cols].to_numpy(dtype="float32")
    y = train_df.is_eog_flare.to_numpy(dtype="int8")
    weights = sample_weights(
        train_df,
        country_alpha=variant["country_alpha"],
        fragment_balance=variant["fragment_balance"],
    )
    train_idx = np.arange(len(train_df), dtype="int64")
    t0 = time.time()
    model = _train(
        X_train, y, train_idx, weights, rounds=rounds, seed=seed
    )
    train_s = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    infer_s = time.time() - t0
    return pred, train_s, infer_s


def _f1_arrays_at_thresholds(y, score, thresholds):
    order = np.argsort(score)
    sorted_score = score[order]
    sorted_y = y[order].astype("int64")
    cumulative_pos = np.concatenate([[0], np.cumsum(sorted_y)])
    index = np.searchsorted(sorted_score, thresholds, side="left")
    predicted = len(y) - index
    true_positive = int(sorted_y.sum()) - cumulative_pos[index]
    denominator = predicted + int(sorted_y.sum())
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype="float64"),
        where=denominator > 0,
    )


def macro_f1_threshold(y, score, countries):
    """Select one threshold that maximizes mean country F1."""
    y = np.asarray(y, dtype="int8")
    score = np.asarray(score, dtype="float64")
    countries = np.asarray(countries)

    candidates = np.unique(score)
    unique_countries = np.unique(countries)
    macro_f1 = np.zeros(len(candidates), dtype="float64")
    for country in unique_countries:
        mask = countries == country
        macro_f1 += (
            _f1_arrays_at_thresholds(y[mask], score[mask], candidates)
            / len(unique_countries)
        )
    index = int(np.nanargmax(macro_f1))
    return float(candidates[index]), float(macro_f1[index])


def country_metrics(df, score, threshold, experiment):
    rows = []
    score = np.asarray(score)
    for country in sorted(df.country.unique()):
        mask = df.country.eq(country).to_numpy()
        y = df.loc[mask, "is_eog_flare"].to_numpy(dtype="int8")
        p = score[mask]
        prevalence = float(y.mean())
        row = metrics(
            y, p, threshold,
            name=experiment,
            extra=dict(
                country=country,
                prevalence=prevalence,
                predicted_positive=int((p >= threshold).sum()),
                prediction_rate=float((p >= threshold).mean()),
            ),
        )
        row["pr_auc_lift"] = (
            row["pr_auc"] / prevalence if prevalence > 0 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def active_eog_counts(countries):
    sites = eog_sites(active_years=WINDOW_YEARS)
    return {
        country: int((sites.country == country).sum())
        for country in countries
    }


def site_metrics(df, score, threshold, experiment, totals):
    rows = []
    scored = df[[
        "country", "is_eog_flare", "eog_flare_id"
    ]].copy()
    scored["predicted_positive"] = np.asarray(score) >= threshold
    for country in sorted(scored.country.unique()):
        part = scored[scored.country == country]
        labelled = part[
            part.is_eog_flare.eq(1) & part.eog_flare_id.notna()
        ]
        recoverable = int(labelled.eog_flare_id.nunique())
        detected = int(labelled.loc[
            labelled.predicted_positive, "eog_flare_id"
        ].nunique())
        total = int(totals.get(country, recoverable))
        rows.append(dict(
            experiment=experiment,
            country=country,
            eog_sites_total=total,
            eog_sites_recoverable=recoverable,
            eog_sites_detected=detected,
            recall_of_recoverable=detected / max(recoverable, 1),
            recall_of_all_active=detected / max(total, 1),
        ))
    return pd.DataFrame(rows)


def evaluate_variants(df, cols, n_splits=5, rounds=N_ROUNDS):
    y = df.is_eog_flare.to_numpy(dtype="int8")
    countries = df.country.to_numpy()
    totals = active_eog_counts(sorted(df.country.unique()))
    summary_rows = []
    country_tables = []
    site_tables = []
    oof_frame = df[[
        "source_id", "country", "block_id", "lat", "lon",
        "is_eog_flare", "eog_flare_id",
    ]].copy()

    for index, variant in enumerate(VARIANTS):
        score, train_s, infer_s = cv_predict(
            df, cols, variant, n_splits=n_splits, rounds=rounds,
            seed=31 + index * 1000,
        )
        pooled_threshold, pooled_f1 = best_f1_threshold(y, score)
        threshold, macro_f1 = macro_f1_threshold(y, score, countries)
        ctable = country_metrics(df, score, threshold, variant["name"])
        stable = site_metrics(
            df, score, threshold, variant["name"], totals
        )
        row = metrics(
            y, score, threshold, train_s, infer_s,
            name=variant["name"],
            extra=dict(
                country_alpha=variant["country_alpha"],
                fragment_balance=variant["fragment_balance"],
                n_features=len(cols),
                threshold_exact=threshold,
                threshold_policy="maximize macro country F1 on grouped OOF",
                macro_country_f1=macro_f1,
                worst_country_f1=float(ctable.f1.min()),
                macro_country_pr_auc=float(ctable.pr_auc.mean()),
                pooled_best_threshold=pooled_threshold,
                pooled_best_f1=pooled_f1,
                macro_site_recall_all=float(
                    stable.recall_of_all_active.mean()
                ),
            ),
        )
        summary_rows.append(row)
        country_tables.append(ctable)
        site_tables.append(stable)
        oof_frame[f"score_{variant['name']}"] = score
        print(
            f"{variant['name']}: macro F1={macro_f1:.4f}, "
            f"pooled F1={row['f1']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
            f"threshold={threshold:.4f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_country_f1", "macro_country_pr_auc"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(summary.iloc[0].experiment)
    selected = next(v for v in VARIANTS if v["name"] == selected_name)
    return (
        summary,
        pd.concat(country_tables, ignore_index=True),
        pd.concat(site_tables, ignore_index=True),
        oof_frame,
        selected,
    )


def corrected_loco(df, cols, variant, inner_splits=3, rounds=N_ROUNDS):
    """Nested LOCO with country-macro threshold selection on inner OOF."""
    rows = []
    country_rows = []
    site_rows = []
    predictions = []
    totals = active_eog_counts(sorted(df.country.unique()))

    for index, country in enumerate(sorted(df.country.unique())):
        train_df = df[df.country != country].reset_index(drop=True)
        test_df = df[df.country == country].reset_index(drop=True)
        inner_score, inner_train_s, _ = cv_predict(
            train_df, cols, variant, n_splits=inner_splits,
            rounds=rounds, seed=5000 + index * 1000,
        )
        inner_y = train_df.is_eog_flare.to_numpy(dtype="int8")
        threshold, inner_macro_f1 = macro_f1_threshold(
            inner_y, inner_score, train_df.country.to_numpy()
        )

        score, final_train_s, infer_s = fit_predict(
            train_df, test_df, cols, variant, rounds=rounds,
            seed=9000 + index * 1000,
        )
        y = test_df.is_eog_flare.to_numpy(dtype="int8")
        row = metrics(
            y, score, threshold,
            train_s=inner_train_s + final_train_s,
            infer_s=infer_s,
            name=f"05c LOCO holdout={country}",
            extra=dict(
                country=country,
                model_variant=variant["name"],
                threshold_policy=(
                    "maximize macro country F1 on grouped inner OOF"
                ),
                inner_macro_country_f1=inner_macro_f1,
                inner_n=len(train_df),
                inner_n_pos=int(inner_y.sum()),
            ),
        )
        rows.append(row)
        country_rows.append(country_metrics(
            test_df, score, threshold, variant["name"]
        ))
        site_rows.append(site_metrics(
            test_df, score, threshold, variant["name"], totals
        ))
        predictions.append(pd.DataFrame({
            "source_id": test_df.source_id,
            "country": country,
            "is_eog_flare": y,
            "eog_flare_id": test_df.eog_flare_id,
            "score": score,
            "threshold": threshold,
            "predicted_positive": score >= threshold,
        }))
        print(
            f"LOCO {country}: threshold={threshold:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, F1={row['f1']:.4f}, "
            f"recall={row['recall']:.4f}",
            flush=True,
        )
        del train_df, test_df, inner_score, score
        gc.collect()

    return (
        pd.DataFrame(rows),
        pd.concat(country_rows, ignore_index=True),
        pd.concat(site_rows, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
    )


def run(n_splits=5, inner_splits=3, rounds=N_ROUNDS):
    df = load_features(TRAIN_COUNTRIES, tag=FEATURE_TAG)
    if HOLDOUT in set(df.country):
        raise AssertionError("India entered Stage 05c training data")
    cols = robust_feature_cols(df)

    summary, countries, sites, oof, selected = evaluate_variants(
        df, cols, n_splits=n_splits, rounds=rounds
    )
    summary.to_csv(OUT / "05c_weighting_variants.csv", index=False)
    countries.to_csv(OUT / "05c_country_oof.csv", index=False)
    sites.to_csv(OUT / "05c_site_oof.csv", index=False)
    oof.to_parquet(CACHE / "05c_oof_predictions.parquet", index=False)

    loco, loco_countries, loco_sites, loco_predictions = corrected_loco(
        df, cols, selected, inner_splits=inner_splits, rounds=rounds
    )
    loco.to_csv(OUT / "05c_corrected_loco.csv", index=False)
    loco_countries.to_csv(OUT / "05c_loco_country_metrics.csv", index=False)
    loco_sites.to_csv(OUT / "05c_loco_site_metrics.csv", index=False)
    loco_predictions.to_parquet(
        CACHE / "05c_corrected_loco_predictions.parquet", index=False
    )

    selected_row = summary[summary.experiment == selected["name"]].iloc[0]
    final_threshold = float(selected_row.threshold_exact)
    X = df[cols].to_numpy(dtype="float32")
    y = df.is_eog_flare.to_numpy(dtype="int8")
    weights = sample_weights(
        df,
        country_alpha=selected["country_alpha"],
        fragment_balance=selected["fragment_balance"],
    )
    model = _train(
        X, y, np.arange(len(df), dtype="int64"), weights,
        rounds=rounds, seed=13031,
    )
    model.save_model(str(CACHE / "05c_final_foreign_model.txt"))
    importance = pd.DataFrame({
        "feature": cols,
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(OUT / "05c_feature_importance.csv", index=False)

    manifest = dict(
        stage="05c_balanced_tabular",
        observation_years=list(WINDOW_YEARS),
        training_countries=TRAIN_COUNTRIES,
        holdout_country=HOLDOUT,
        holdout_loaded=False,
        selected_variant=selected,
        selection_rule=(
            "highest macro country F1 on foreign grouped OOF; "
            "macro country PR-AUC breaks ties"
        ),
        threshold=final_threshold,
        threshold_policy="maximize macro country F1 on foreign grouped OOF",
        features=cols,
        lightgbm_params=PARAMS,
        num_boost_round=rounds,
        grouped_cv_folds=n_splits,
        corrected_loco_inner_folds=inner_splits,
        india_policy=(
            "Do not build or score India until Stage 05c results are reviewed "
            "and this model specification is accepted without retuning."
        ),
    )
    with open(OUT / "05c_model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nWEIGHTING VARIANTS")
    print(summary[[
        "experiment", "thr", "precision", "recall", "f1", "pr_auc",
        "macro_country_f1", "worst_country_f1", "macro_site_recall_all",
    ]].to_string(index=False))
    print(f"\nSelected variant: {selected['name']}")
    print("\nCORRECTED LOCO")
    print(show(loco))
    return summary, loco


def main():
    if len(sys.argv) > 1:
        raise SystemExit("usage: kg_05c_balanced_tabular.py")
    run()


if __name__ == "__main__":
    main()
