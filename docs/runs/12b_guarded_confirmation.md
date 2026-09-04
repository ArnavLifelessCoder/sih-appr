# NB12b: guarded CV stability confirmation

NB12 found a real but not yet promotable computer-vision gain. The guarded
Sentinel branch improved macro country PR-AUC from 0.9082 to 0.9283 and improved
all six countries. The broad nested selector still fell back because it chose
the all-77 branch for Angola, where that choice lost 0.0456 PR-AUC.

NB12b tests the useful finding without repeating that failed selector. There is
one eligible challenger, `guarded_cv_tabular`, against the protected
`compact_tabular` baseline. The all-77 and nested branches are diagnostics only
and cannot affect the decision.

## Frozen confirmation design

- NB12 seed 271 is discovery evidence and does not count toward confirmation.
- Three fresh seeds are fixed as 272, 273, and 274.
- Each seed reruns all six outer leave-one-country-out folds.
- Every scaler, PCA, PU probe, LightGBM model, alpha, and F1 threshold is fitted
  without the current outer country.
- The final comparison averages the three outer prediction scores per source.
- Ensemble F1 uses the median of the three seed-specific fold-local thresholds.
- The paired spatial-block bootstrap uses 5,000 repeats and fixed seed 12012.
- Tied review scores are broken by source ID, so Recall@20 is deterministic.

The guarded ensemble is accepted only if all of these checks pass:

1. Macro country PR-AUC gain is at least 0.010.
2. At least four of six countries improve.
3. Worst country PR-AUC change is at least -0.020.
4. Indonesia PR-AUC change is at least -0.020.
5. Macro F1 change is at least -0.010.
6. Paired bootstrap lower 95% bound is at least -0.010.
7. Macro Recall@20 change is at least -0.020.
8. No country loses more than one recovered positive at the 20% review budget.
9. Every fresh seed has a positive macro PR-AUC gain.
10. Median seed gain is at least 0.010.
11. At least two of three full-foreign seed pipelines select a nonzero alpha
    from the frozen grid.

Any failed condition selects compact. Seeds, margins, and branches must not be
changed after seeing the run.

## Kaggle setup

Use `notebooks/kaggle/12b_guarded_confirmation.ipynb`.

Set:

- Accelerator: None. The audited embedding cache makes this run CPU-only.
- Internet: Off.

Attach exactly these saved notebook outputs:

1. NB2, containing one common-window feature parquet for each of Algeria,
   Angola, Indonesia, Iraq, Libya, and Nigeria.
2. NB6 v2, containing the 294 successful image chips plus its pilot source,
   image feature, image quality, run-state, and feature-manifest files.
3. NB12, containing the unpacked `cache/12_cv_embeddings.parquet`,
   `cache/12_cv_embeddings.json`, `cache/12_morphology.parquet`, and
   `cache/12_morphology.json` files.

Do not attach India, NB4, NB6 v1, or multiple copies of NB12. The notebook
verifies the audited cache hashes and fails closed if the expected cache is not
present. It does not download or refit the frozen image encoder.

Use Save Version and choose Save and Run All. Download
`nb12b_confirmatory_results.zip` from Output. With the cached embeddings, the
expected free-plan runtime is several minutes, not a long CNN training run.

## Output and interpretation

The ZIP contains per-seed predictions and metrics, the three-seed ensemble,
fold-local threshold records, review queue membership, acceptance results,
country stability, nine compact LightGBM models, three visual PU models, the
ensemble calibration reference, verified caches, exact code, and hashes.

If guarded passes, deployment averages the nine compact model probabilities and
the three visual pipelines. Within each seed, it first averages that seed's three
compact models, applies that seed's OOF calibration and selected alpha with its
visual model, then averages the three fused pipeline scores. This matches the
procedure evaluated by the three-seed outer predictions. Missing imagery uses
the mean calibrated compact pipeline score. If guarded fails, deployment uses
the raw nine-model compact ensemble only.

This run measures algorithmic seed stability on the same enriched 294-source
foreign cohort. It is not an independent country test, an India result, or a
population precision estimate. A historical India result exists for the superseded
NB8 model, but the final NB12b guarded model has not been evaluated on India.
