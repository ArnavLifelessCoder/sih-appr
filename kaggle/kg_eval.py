"""Shared evaluation: one metric block for every experiment, plus split builders."""
import time, json
import numpy as np, pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, average_precision_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix,
                             brier_score_loss, precision_recall_curve)
from sklearn.model_selection import GroupKFold
from kg_common import *

def metrics(y, p, thr=0.5, train_s=None, infer_s=None, name="", extra=None):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    d = dict(
        experiment=name, n=len(y), n_pos=int(y.sum()), thr=round(float(thr), 4),
        precision=precision_score(y, yh, zero_division=0),
        recall=recall_score(y, yh, zero_division=0),
        f1=f1_score(y, yh, zero_division=0),
        pr_auc=average_precision_score(y, p) if y.sum() else np.nan,
        roc_auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        mcc=matthews_corrcoef(y, yh) if len(set(yh)) > 1 else 0.0,
        brier=brier_score_loss(y, np.clip(p, 0, 1)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        train_s=round(train_s, 2) if train_s is not None else np.nan,
        infer_s=round(infer_s, 3) if infer_s is not None else np.nan)
    if extra: d.update(extra)
    return d

def best_f1_threshold(y, p, grid=None):
    """Return the exact score threshold that maximizes binary F1.

    The previous default searched 120 score quantiles. With fewer than 0.3%
    labelled positives, that grid skipped most of the operational score tail
    and could miss materially better thresholds. An explicit ``grid`` is still
    supported for sensitivity analysis; the default now evaluates every
    precision-recall operating point.
    """
    y = np.asarray(y, dtype="int8")
    p = np.asarray(p, dtype="float64")
    if len(y) != len(p) or not len(y):
        raise ValueError("y and p must be non-empty arrays of equal length")

    if grid is not None:
        grid = np.asarray(grid, dtype="float64")
        if not len(grid):
            raise ValueError("grid cannot be empty")
        fs = np.asarray([
            f1_score(y, p >= t, zero_division=0) for t in grid
        ])
        j = int(np.nanargmax(fs))
        return float(grid[j]), float(fs[j])

    if y.sum() == 0:
        return float(np.nextafter(p.max(), np.inf)), 0.0

    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        return 0.5, 0.0
    f1 = (2 * precision[:-1] * recall[:-1] /
          np.maximum(precision[:-1] + recall[:-1], 1e-15))
    j = int(np.nanargmax(f1))
    return float(thresholds[j]), float(f1[j])

def calibration_table(y, p, bins=10):
    q = pd.qcut(pd.Series(p), bins, duplicates="drop", labels=False)
    t = pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(
        n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
    t["gap"] = (t.pred - t.obs).abs()
    return t

def load_features(countries, tag=None):
    suffix = f"_{tag}" if tag else ""
    fs = [pd.read_parquet(CACHE / f"features_{c}{suffix}.parquet") for c in countries]
    return pd.concat(fs, ignore_index=True)

def grouped_folds(df, n_splits=5, seed=0):
    """GroupKFold on block_id -- fragments of one physical flare stay on one side."""
    gk = GroupKFold(n_splits=n_splits)
    return list(gk.split(df, df.is_eog_flare.values, groups=df.block_id.values))

def leave_country_out(df, countries=None):
    cs = countries or sorted(df.country.unique())
    for c in cs:
        te = (df.country == c).values
        if df.loc[te, "is_eog_flare"].sum() == 0:
            continue                     # no positives -> nothing to score
        yield c, ~te, te

def log_experiment(rows, path):
    p = OUT / path
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return df

def show(df, cols=None):
    cols = cols or ["experiment", "n", "n_pos", "precision", "recall", "f1",
                    "pr_auc", "roc_auc", "mcc", "brier", "tp", "fp", "fn", "train_s"]
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].round(4)
    return d.to_string(index=False)
