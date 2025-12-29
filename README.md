# MVSC-CD (Multi-view Subspace Clustering: Consensus + Difference)

This repository contains a MATLAB implementation of MVSC-CD with:

- Robust multi-view dataset loader (`load_multiview_dataset.m`)
- ADMM solver (`mvsc_cd_admm.m`) with consensus/diversity modeling
- Spectral clustering pipeline (`spectral_clustering.m`)
- Evaluation metrics (`eval_clustering_metrics.m`)
- Utility scripts for dataset inspection and a runnable demo

## Requirements
- MATLAB R2021a+ (or Octave with compatible toolboxes for `svd`, `kmeans`, and `functiontests`)
- Datasets placed under `datasets/` as `.mat` files using any of the supported formats.

## Quick start
```matlab
addpath(genpath('.'));
run_demo; % defaults to datasets/Caltech101-7.mat or synthetic fallback
```

To run on a specific dataset:
```matlab
run_demo('datasets/BBCSport.mat');
```

If the dataset is missing, the demo generates a synthetic two-view dataset so the full
pipeline can still be exercised.

## Dataset loader
`load_multiview_dataset.m` supports the following field layouts inside `.mat` files:
- `data` + `truelabel`
- `X` + `Y`
- `fea` + `gt`
- `data` + `label`

Behavior:
- Returns a cell array of views with samples on rows and column-wise L2 normalization.
- Automatically orients views when samples are stored as columns.
- Remaps labels to `1..K` (with the mapping recorded in `meta.label_map`).
- Provides rich metadata (shapes, detected format, etc.).

## ADMM solver
`mvsc_cd_admm.m` implements the consensus + diversity ADMM:
- `Z` update via a Sylvester solve (with `diag(Z)=0` enforced)
- `C` closed-form linear solve
- `S` Schatten-`p` proximal (soft-thresholding when `p=1`, GST when `p<1`)
- `E` weighted L1 via ISTA-like update
- Dual update on `Y = Y + C - S`
- Complementary diversity term counts pairs once (`v < w`)

The returned affinity for clustering is `|S|`.

## Spectral clustering
`spectral_clustering.m` builds a normalized Laplacian from an affinity matrix and applies
`k`-means on the leading eigenvectors.

## Evaluation metrics
`eval_clustering_metrics.m` computes:
- ACC (Hungarian best mapping)
- NMI
- Purity
- ARI
- Pairwise Precision / Recall / F-score

No external toolboxes are required.

## Scripts
- `scripts/inspect_dataset.m` — iterate through known datasets and print shapes.
- `scripts/run_demo.m` — end-to-end pipeline: load data → ADMM → spectral clustering →
  metrics (if ground truth is present).

## Tests
`tests/test_loader_all_datasets.m` attempts to load every dataset listed in the spec and
asserts consistent dimensions. Missing datasets are skipped with a warning.

Run all tests:
```matlab
addpath(genpath('.'));
results = runtests('tests');
table(results)
```

## Recent enhancements (interfaces unchanged)
- Data preprocessing (loader): optional row-wise z-score per view (`opts.preprocess.rowZScore`, default **false**) and view energy alignment (`opts.preprocess.viewEnergyAlign`, default **true**) applied before column L2 normalization.
- Graph polishing (`src/build_affinity_from_S.m`): removes diag, abs-symmetrizes, adaptive Top-K (`choose_topk`), auto symmetrize (`max/avg/auto`), optional column normalization and diffusion. Controlled via `opts.graph` (default: `topK='auto'`, `colNormalize=true`, `symmetrizeMode='auto'`).
- Spectral clustering stability: normalized Laplacian, eigenvector row-normalization, k-means with `Start='plus'`, `Replicates=30`, `MaxIter=1000`, fixed seed (`opts.seed`, default 0).
- ADMM engineering: multiple inner ISTA steps on `E` (`opts.innerEIter`, default 5) and verbose residual print of `r(C-S)` when `verbose=true` (core updates unchanged).
- Lambda light autotune (`choose_lambda_light`): short warmup over small lambda grid using graph density/connectivity (and ACC/NMI if labels exist) to pick `lambda_e`. Enabled in `run_demo` by default (`opts.autotune.lambda='light'`), disable via `'off'`.
- Quick tuning script: `scripts/tune_params_grid.m` runs a small grid over `{alpha, lambda_e, gamma, p}`, saves to `results/tuning_*.mat`, and rechecks top configs with more iterations.

## How to toggle enhancements in run_demo
```matlab
% disable lambda autotune and graph Top-K
run_demo('datasets/BBCSport.mat', struct( ...
    'autotune', struct('lambda','off'), ...
    'graph', struct('topKEnabled', false)));

% enable diffusion or change Top-K override
opts = struct('graph', struct('diffusion', true, 'topKOverride', 15));
run_demo('datasets/Caltech101-7.mat', opts);
```

## Notes and memory hints
- Normalizing columns helps numerical stability across heterogeneous views.
- Large datasets may require increasing memory; consider running with reduced `max_iter`
  or `verbose=false` to reduce overhead.
- The ADMM history returned by `mvsc_cd_admm` includes primal/dual residuals and objective
  values for monitoring convergence.
