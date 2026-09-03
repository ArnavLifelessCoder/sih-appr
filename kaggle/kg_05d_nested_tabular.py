"""NB5: fold-local weights and matched-seed, nested country evaluation.

Writes only 05d artifacts. Historical 05b/05c outputs are not overwritten.
India is never loaded. These are development estimates against EOG labels.
"""
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import kg_05c_balanced_tabular as balanced
from kg_05b_robust_tabular import FEATURE_TAG, PARAMS, WINDOW_YEARS, robust_feature_cols
from kg_common import CACHE, OUT, TRAIN_COUNTRIES, HOLDOUT
from kg_eval import load_features, metrics

PROTOCOL_VERSION = "05d-nested-v1"


def site_safe_blocks(df):
    """Merge spatial blocks connected by the same known site within a country.

    Unknown sites cannot be protected by this label-based integrity check.
    These group identifiers are never model features.
    """
    keys = pd.MultiIndex.from_frame(df[["country", "block_id"]])
    codes, unique = pd.factorize(keys, sort=True)
    parent = np.arange(len(unique))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    positive = df.is_eog_flare.eq(1)
    sites = df.loc[positive, ["country", "eog_flare_id"]].copy()
    sites["block_code"] = codes[positive.to_numpy()]
    for _, part in sites.groupby(["country", "eog_flare_id"], observed=True):
        blocks = part.block_code.unique()
        first = root(blocks[0])
        for block in blocks[1:]:
            parent[root(block)] = first
    roots = np.array([root(i) for i in range(len(unique))])
    mapping = unique.to_frame(index=False)
    mapping["evaluation_block"] = [f"siteblock_{r}" for r in roots]
    revised = df.copy()
    revised["block_id"] = mapping.evaluation_block.to_numpy()[codes]
    return revised, mapping


def validate_features(df, cols):
    required = ["source_id", "country", "block_id", "is_eog_flare"]
    if df[required].isna().any().any() or not df.source_id.is_unique:
        raise ValueError("Source IDs must be unique and required identifiers/labels non-null")
    if set(df.country) != set(TRAIN_COUNTRIES) or HOLDOUT in set(df.country):
        raise ValueError("Expected exactly the six foreign countries; India is forbidden")
    if not df.is_eog_flare.isin([0, 1]).all():
        raise ValueError("Expected binary EOG-match labels")
    if df.loc[df.is_eog_flare.eq(1), "eog_flare_id"].isna().any():
        raise ValueError("Positive source missing its physical EOG site identifier")
    for country, part in df.groupby("country", observed=True):
        if part.is_eog_flare.nunique() != 2:
            raise ValueError(f"Both labels are required for country {country}")
    if not cols or np.isinf(df[cols].to_numpy(dtype="float32")).any():
        raise ValueError("Feature set is empty or contains infinite values")


def nested_loco(df, cols, inner_splits=3, rounds=500, seed=31, checkpoint=None):
    rows, selections, sites, predictions = [], [], [], []
    totals = balanced.active_eog_counts(sorted(df.country.unique()))
    for index, country in enumerate(sorted(df.country.unique())):
        train = df.loc[df.country.ne(country)].reset_index(drop=True)
        test = df.loc[df.country.eq(country)].reset_index(drop=True)
        inner_seed, fit_seed = seed + 10000 + index * 1000, seed + 20000 + index * 1000
        print(f"\nHOLDOUT {country}: selecting from training countries only", flush=True)
        summary, _, _, inner_oof, selected = balanced.evaluate_variants(
            train, cols, n_splits=inner_splits, rounds=rounds, seed=inner_seed,
        )
        winner = summary.loc[summary.experiment.eq(selected["name"])].iloc[0]
        threshold = float(winner.threshold_exact)
        selection = summary.copy()
        selection["held_out_country"] = country
        selection["selected"] = selection.experiment.eq(selected["name"])
        selection["inner_seed"] = inner_seed
        selections.append(selection)
        # Only now predict and evaluate the outer country.
        del inner_oof
        score, train_s, infer_s = balanced.fit_predict(
            train, test, cols, selected, rounds=rounds, seed=fit_seed,
        )
        rows.append(metrics(
            test.is_eog_flare.to_numpy(), score, threshold,
            train_s=float(summary.train_s.sum()) + train_s, infer_s=infer_s,
            name=f"05d holdout={country}", extra={
                "country": country, "model_variant": selected["name"],
                "threshold_exact": threshold, "inner_seed": inner_seed,
                "fit_seed": fit_seed, "inner_macro_f1": float(winner.macro_country_f1),
                "threshold_policy": "training-country grouped OOF macro F1",
            },
        ))
        sites.append(balanced.site_metrics(test, score, threshold, selected["name"], totals))
        prediction = test[["source_id", "country", "block_id", "is_eog_flare", "eog_flare_id"]].copy()
        prediction["score"] = score
        prediction["threshold"] = threshold
        prediction["predicted_positive"] = score >= threshold
        prediction["model_variant"] = selected["name"]
        predictions.append(prediction)
        print(f"{country}: {selected['name']}, F1={rows[-1]['f1']:.4f}, AP={rows[-1]['pr_auc']:.4f}", flush=True)
        if checkpoint is not None:
            checkpoint(pd.DataFrame(rows), pd.concat(selections, ignore_index=True),
                       pd.concat(sites, ignore_index=True), pd.concat(predictions, ignore_index=True))
        del train, test, score
        gc.collect()
    return (pd.DataFrame(rows), pd.concat(selections, ignore_index=True),
            pd.concat(sites, ignore_index=True), pd.concat(predictions, ignore_index=True))


def _save_loco(loco, selections, sites, predictions):
    loco.to_csv(OUT / "05d_nested_loco.csv", index=False)
    selections.to_csv(OUT / "05d_inner_selection.csv", index=False)
    sites.to_csv(OUT / "05d_loco_site_metrics.csv", index=False)
    predictions.to_parquet(CACHE / "05d_loco_predictions.parquet", index=False)


def run(n_splits=5, inner_splits=3, rounds=500, seed=31):
    if n_splits < 2 or inner_splits < 2 or rounds < 1:
        raise ValueError("Need at least two folds and one boosting round")
    manifest_path = OUT / "05d_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("05d run already exists; use a fresh Kaggle session/output directory")
    df = load_features(TRAIN_COUNTRIES, tag=FEATURE_TAG)
    cols = robust_feature_cols(df)
    validate_features(df, cols)
    df, mapping = site_safe_blocks(df)
    mapping.to_csv(OUT / "05d_spatial_group_map.csv", index=False)
    inputs = {}
    for country in TRAIN_COUNTRIES:
        path = CACHE / f"features_{country}_{FEATURE_TAG}.parquet"
        with path.open("rb") as stream:
            inputs[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "protocol": PROTOCOL_VERSION, "status": "running",
        "holdout_country": HOLDOUT, "holdout_loaded": False,
        "training_countries": TRAIN_COUNTRIES, "years": list(WINDOW_YEARS),
        "features": cols, "n_sources": len(df), "n_positive": int(df.is_eog_flare.sum()),
        "n_splits": n_splits, "inner_splits": inner_splits, "rounds": rounds,
        "seed": seed, "seed_repetitions": 1,
        "seed_policy": "matched across variants; inner=seed+10000+country_index*1000; outer=seed+20000+country_index*1000; CV fold adds fold_index*100",
        "weights": "computed from training fold only",
        "selection": "macro-country F1, then macro-country AP, using only outer training countries",
        "spatial_groups_before": len(mapping),
        "spatial_groups_after": int(mapping.evaluation_block.nunique()),
        "input_sha256": inputs, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow"]},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip(),
        "limitations": "EOG proxy labels; unknown site overlap remains possible; foreign data used in historical research decisions; no India accuracy or repeated-seed uncertainty",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Loaded {len(df):,} sources, {len(cols)} features; India excluded", flush=True)
    loco, selections, sites, predictions = nested_loco(
        df, cols, inner_splits=inner_splits, rounds=rounds, seed=seed, checkpoint=_save_loco,
    )
    # All-country development selection is separate from the outer evaluation.
    summary, countries, site_oof, oof, selected = balanced.evaluate_variants(
        df, cols, n_splits=n_splits, rounds=rounds, seed=seed,
    )
    summary.to_csv(OUT / "05d_development_variants.csv", index=False)
    countries.to_csv(OUT / "05d_development_countries.csv", index=False)
    site_oof.to_csv(OUT / "05d_development_sites.csv", index=False)
    oof.to_parquet(CACHE / "05d_oof_predictions.parquet", index=False)
    final_seed = seed + 30000
    weights = balanced.sample_weights(df, selected["country_alpha"], selected["fragment_balance"])
    model = balanced._train(df[cols].to_numpy(dtype="float32"), df.is_eog_flare.to_numpy(dtype="int8"),
                            np.arange(len(df)), weights, rounds=rounds, seed=final_seed)
    model.save_model(str(CACHE / "05d_final_foreign_model.txt"))
    pd.DataFrame({"feature": cols, "gain": model.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(OUT / "05d_feature_importance.csv", index=False)
    final_params = dict(PARAMS)
    final_params.update({key: final_seed for key in ["seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"]})
    manifest.update(status="complete", selected_variant=selected,
                    final_threshold=float(summary.loc[summary.experiment.eq(selected["name"]), "threshold_exact"].iloc[0]),
                    final_lightgbm_params=final_params,
                    macro_loco_f1=float(loco.f1.mean()), macro_loco_pr_auc=float(loco.pr_auc.mean()))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(loco[["country", "model_variant", "f1", "pr_auc", "recall"]].to_string(index=False))
    return summary, loco


if __name__ == "__main__":
    run()
