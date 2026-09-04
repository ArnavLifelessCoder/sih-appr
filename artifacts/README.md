# Frozen model artifacts

`nb12b/` contains the exact files required to reconstruct the selected foreign-country
ensemble without retaining the large imagery-chip directory.

## Inventory

- Nine LightGBM text models: three folds for each confirmation seed 272, 273, and 274.
- Three visual positive-unlabelled models, one per confirmation seed.
- One fusion calibration archive containing the selected per-seed blending state.
- Cached CV embeddings and morphology tables plus their metadata.

The deployment branch is `guarded_cv_tabular`. Each seed selected fusion alpha 0.5.
The protocol is `12b-guarded-confirmatory-ensemble-v1`.

Load the joblib models with scikit-learn 1.6.1, the version recorded by the run. Newer or
older scikit-learn versions can deserialize them with warnings and are not the frozen
execution environment.

## Integrity anchors

| Item | SHA256 |
|---|---|
| Stable 294-source panel | `7c44d70f3505d65d92bba4abbd2c2e9bebf3eef6a1eb9a432880612ffef3a2f0` |
| CV embedding table | `44b78d9137b0cebc3f07d6aba2322a0139d402f04c1329fbac84ecc43a74f00b` |
| Morphology table | `b873bf53eaf9161f9d4f775bc84af9670a7d7d52124988a921347894e0127719` |
| Image checkpoint used to build embeddings | `e3a335e38d1d189ad3b0eba4be4004a9c52c5e846317b6737ac9f0fac57e1ac8` |
| Acceptance decision file | `972f3ac8afeb8d1d056e07ca7db39d28432ce108192dff5e644184f9f3693ba1` |
| Final ensemble predictions | `6386efa6739e0710182e311cbb36314891afe89eeaad79723741a35132e982b8` |

The authoritative per-file hashes, package versions, input hashes, and acceptance policy
are recorded in `../results/nb12b_confirmatory/12b_manifest.json`.
