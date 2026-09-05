"""Verify retained India evidence without downloading imagery or loading models."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_hash(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == expected, f"SHA-256 mismatch: {path}")


def check_number(actual: float, expected: float, context: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), context)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    evidence = repo / "results/nb13_india_transfer"
    manifest = json.loads((evidence / "13_manifest.json").read_text())
    config = json.loads((evidence / "run_config.json").read_text())
    decision = json.loads((evidence / "13_india_decision.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        require(Path(name).name == name, f"Unexpected artifact path: {name}")
        check_hash(evidence / name, expected)
    foreign = repo / "results/nb12b_confirmatory"
    check_hash(foreign / "12b_manifest.json", manifest["model_manifest_sha256"])
    check_hash(foreign / "12b_selected_schema.json", manifest["deployment_schema_sha256"])

    panel = pd.read_parquet(evidence / "india_panel.parquet")
    predictions = pd.read_parquet(evidence / "13_india_predictions.parquet")
    require(len(panel) == manifest["panel_sources"] == config["n_sources"], "Panel size mismatch")
    require(len(predictions) == manifest["qa_scored_sources"], "Scored size mismatch")
    for name, frame in (("panel", panel), ("predictions", predictions)):
        require(frame.source_id.is_unique, f"Duplicate source IDs in {name}")
        require(frame.block_id.is_unique, f"Duplicate spatial blocks in {name}")
        require(frame.country.eq("India").all(), f"Unexpected country in {name}")
        require(frame.is_eog_flare.isin([0, 1]).all(), f"Invalid labels in {name}")
    require(set(predictions.source_id) <= set(panel.source_id), "Predictions outside panel")
    columns = ["source_id", "country", "block_id", "lat", "lon", "is_eog_flare",
               "eog_flare_id", "chip_id", "batch_order"]
    canonical = panel[columns].copy()
    for column in columns:
        canonical[column] = canonical[column].fillna("<NA>").astype(str)
    payload = canonical.sort_values("batch_order").to_csv(index=False, lineterminator="\n")
    panel_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    require(panel_hash == config["panel_sha256"] == manifest["panel_sha256"], "Panel hash mismatch")
    shared = columns[:-1]
    expected_rows = panel.set_index("source_id").loc[predictions.source_id].reset_index()
    pd.testing.assert_frame_equal(predictions[shared].reset_index(drop=True),
                                  expected_rows[shared], check_dtype=False)
    labels = predictions.is_eog_flare
    require(int(labels.sum()) == manifest["scored_positive_sites"], "Positive count mismatch")
    positive_sites = predictions.loc[labels.eq(1), "eog_flare_id"].nunique()
    require(positive_sites == int(labels.sum()), "Repeated positive sites")

    metrics = pd.read_csv(evidence / "13_india_ranking_metrics.csv").set_index("branch")
    budgets = pd.read_csv(evidence / "13_india_review_budgets.csv")
    branches = {"compact_tabular", "guarded_cv_tabular"}
    require(set(metrics.index) == branches and metrics.index.is_unique, "Unexpected metric branches")
    require(len(budgets) == 6 and not budgets.duplicated(["branch", "review_fraction"]).any(),
            "Review budget rows missing or duplicated")
    ap = {}
    recovered = {}
    for branch in sorted(branches):
        scores = predictions[f"score_{branch}"]
        require(np.isfinite(scores).all() and scores.between(0, 1).all(), f"Invalid scores: {branch}")
        ap[branch] = average_precision_score(labels, scores)
        check_number(ap[branch], metrics.loc[branch, "eog_proxy_pr_auc"], f"PR-AUC mismatch: {branch}")
        check_number(roc_auc_score(labels, scores), metrics.loc[branch, "eog_proxy_roc_auc"],
                     f"ROC-AUC mismatch: {branch}")
        require(pd.isna(metrics.loc[branch, "f1"]) and pd.isna(metrics.loc[branch, "threshold"]),
                "Unexpected threshold or F1")
        ordered = predictions.sort_values([f"score_{branch}", "source_id"], ascending=[False, True])
        for fraction in (0.1, 0.2, 0.3):
            row = budgets.loc[budgets.branch.eq(branch) & budgets.review_fraction.eq(fraction)]
            require(len(row) == 1, f"Missing review budget: {branch}, {fraction}")
            row = row.iloc[0]
            count = math.ceil(fraction * len(predictions))
            reviewed = ordered.head(count)
            found = int(reviewed.is_eog_flare.sum())
            sites = reviewed.loc[reviewed.is_eog_flare.eq(1), "eog_flare_id"].nunique()
            for field, value in {"reviewed_sources": count, "positive_rows_found": found,
                                 "positive_sites_found": sites, "positive_row_recall": found / labels.sum(),
                                 "positive_site_recall": sites / positive_sites}.items():
                check_number(value, row[field], f"Review mismatch: {branch}, {fraction}, {field}")
            if fraction == 0.2:
                recovered[branch] = sites
    delta = ap["guarded_cv_tabular"] - ap["compact_tabular"]
    site_delta = recovered["guarded_cv_tabular"] - recovered["compact_tabular"]
    check_number(delta, decision["eog_proxy_pr_auc_delta_guarded_minus_compact"], "Decision AP mismatch")
    check_number(site_delta, decision["positive_sites_at_20_delta_guarded_minus_compact"], "Decision site mismatch")
    passed = bool(delta >= -0.01 and site_delta >= -1)
    require(passed == decision["passed"], "Decision guard mismatch")
    selected = "guarded_cv_tabular" if passed else "compact_tabular"
    require(selected == decision["india_ranking_branch"] == manifest["selected_branch"], "Branch mismatch")
    print(f"Verified {len(manifest['output_sha256'])} artifact hashes and frozen model/schema anchors.")
    print(f"Verified {len(panel)} panel rows, {len(predictions)} predictions, AP/ROC and all six review budgets.")
    print(f"Decision: {selected}; PR-AUC delta {delta:.6f}.")
    print("No model inference, bootstrap rerun, or external imagery/input verification performed.")


if __name__ == "__main__":
    main()
