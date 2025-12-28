# MVSC-CD (Multi-view Subspace Clustering with Consensus & Diversity) — Implementation Spec for Codex

## 0) Goal
Implement a MATLAB project that runs MVSC-CD on multiple multi-view .mat datasets with inconsistent variable names.
The project must:
1) Robustly load any dataset in the provided formats into a single internal format (Xcell, gt, k).
2) Run the ADMM optimization to learn {Z^v}, consensus S*, and {E^v}.
3) Build an affinity matrix from S* and perform spectral clustering to output labels.
4) Provide reproducible demo scripts + minimal tests.

This spec includes HARD constraints confirmed by user:
- Column-wise l2 normalization for each view
- Enforce diag(Z^v)=0 after each Z update
- Diversity term sums only over v<w (no double counting)
- Spectral clustering must use S* (not C or avg(Z))
- ADMM multiplier update uses (C - S*)

Also: remap ground-truth labels to 1..k using unique() stable mapping (keep original in meta).

---

## 1) Internal Data Format (the ONLY format the core algorithm consumes)

### 1.1 Internal variables
- Xcell: 1xV cell, Xcell{v} is dv x N (columns are samples)
- gt:    N x 1 double (may be empty if not available)
- k:     number of clusters (if gt exists and user does not provide k, infer from unique(gt))
- meta:  struct containing dataset parsing decisions (chosen variable names, dims, notes)

### 1.2 Column-wise normalization (HARD)
For each view Xv, normalize each column to unit l2 norm:
- Dense:  Xv = Xv ./ (sqrt(sum(Xv.^2,1)) + eps)
- Sparse: Xv = Xv * spdiags(1./(sqrt(sum(Xv.^2,1))'+eps),0,N,N)

---

## 2) Dataset Adapter Layer (MUST be implemented first)

### 2.1 Supported dataset formats (from user's whos listing)
The loader must auto-detect these patterns:

Format A: data + truelabel (both are cell)
- data:      1xV cell
- truelabel: 1xV cell (labels per view; should be identical)
Examples: 3sources, 100leaves, BBC, BBCSport, Hdigit, HW, HW2sources, Mfeat, NGs, WebKB

Format B: X + Y
- X: cell (Vx1 or 1xV)
- Y: Nx1 double
Examples: Caltech101-7, Caltech101-20, coil-20, NUS, ORL

Format C: fea + gt
- fea: 1xV cell
- gt:  Nx1 double
Examples: MSRC_v1, UCI_Digits

Format D: data + label (label is row vector)
- data:  1xV cell
- label: 1xN double
Example: YaleB10_3_650

### 2.2 Variable name priorities (hard-coded for this collection)
Data variable priority:  data > X > fea
Label variable priority: truelabel > Y > gt > label

### 2.3 Loader function signature
Create:
  [Xcell, gt, k, meta] = load_multiview_dataset(matPath, opts)

opts fields (all optional):
- opts.dataVar, opts.labelVar: force variable names if needed
- opts.k: override k (otherwise infer from gt)
- opts.normalizeColumns: default true
- opts.forceTranspose: default false (only for emergency)
- opts.maxN: optional subsampling for huge datasets (e.g., coil-20)

### 2.4 Loader algorithm (required behavior)
1) Use whos('-file', matPath) to list variable names.
2) Choose dataVar/labelVar based on priorities unless user forces them.
3) Load ONLY the chosen variables (load(matPath, dataVar, labelVar)) to save memory.
4) Convert data container to 1xV cell:
   - If Vx1, transpose to 1xV; if 1xV keep; if numeric matrix -> wrap as {X}.
5) Standardize labels:
   - If label is cell (truelabel):
       * convert each to column, check all views identical; if identical use first; else error.
   - If label is 1xN -> reshape to Nx1
   - Remap to 1..k using [~,~,gt] = unique(gt,'stable')
   - Keep original labels in meta.gt_original
6) Decide orientation for each view:
   - If gt available: N = length(gt)
       * If size(Xv,2)==N keep dv x N
       * Else if size(Xv,1)==N transpose
       * Else error with detailed meta.notes
7) Convert Xv to double (keep sparse if already sparse).
8) Column normalize each view (HARD, default on).
9) Compute k:
   - if opts.k exists use it; else infer from gt if available; else require user later.
10) meta must include:
   - meta.chosenDataVar, meta.chosenLabelVar, meta.V, meta.N, meta.viewDims
   - meta.orientationFixApplied (per view)
   - meta.normalized
   - meta.notes

---

## 3) Model / Objective (MVSC-CD)

Given multi-view data X^v (dv x N), learn:
- Z^v: N x N self-representation per view
- S:   N x N consensus representation (S*)
- E^v: N x N view-specific error
Introduce auxiliary variable C with constraint C = S to enable ADMM.

Objective (high level, per provided docs):
Sum over views:
- reconstruction: ||X^v - X^v Z^v||_F^2
- consensus/diversity modeling: alpha * ||Z^v - Z^v (S + E^v)||_F^2
Regularizers:
- lambda * Schatten-p norm on S (p=1 nuclear norm; 0<p<1 via GST on singular values)
- gamma * sum_{v<w} ||E^v ⊙ E^w||_1  (HARD: only v<w)

---

## 4) Optimization: ADMM with auxiliary C (C = S)

Maintain variables:
- Z{v}, E{v}, C, S, Y (multiplier), mu (penalty)

At each outer iteration:
(1) Update each Z^v (independent across views):
  min_Z ||X^v - X^v Z||_F^2 + alpha ||Z - Z(C+E^v)||_F^2
  Let A = Xv' * Xv
      M = I - C - Ev
      Q = M * M'
  Solve Sylvester:
      A*Z + alpha*Z*Q = A
  MATLAB: Z = sylvester(A, alpha*Q, A)
  HARD: set diag(Z)=0 after update

(2) Update C:
  Solve linear system:
    (2*alpha*sum_v(Zv'*Zv) + mu*I) * C
      = 2*alpha*sum_v( (Zv'*Zv) * (I - Ev) ) + mu*S - Y
  MATLAB: C = A \ B

(3) Update S (Schatten-p proximal):
  Sbar = C + Y/mu
  eta  = lambda/mu
  Solve: min_S eta*||S||_{Schatten-p}^p + 0.5||S - Sbar||_F^2
  Do SVD: Sbar = U*diag(delta)*V'
  For each singular value delta_i, compute theta_i:
    - if p==1: theta_i = max(delta_i - eta, 0)
    - if 0<p<1: theta_i = GST(delta_i, eta, p)  (implement stable scalar solver)
  S = U*diag(theta)*V'

(4) Update E^v (block coordinate over views, proximal gradient / ISTA)
  Fix all other E^w (w!=v). Then complement term becomes weighted L1:
    Wv = gamma * sum_{w!=v} abs(Ew)
  Smooth part:
    f(Ev) = alpha * || Zv * (I - C - Ev) ||_F^2
  Gradient:
    grad = 2*alpha * (Zv'*Zv) * (C + Ev - I)
  Step size:
    t = 1 / (2*alpha*norm(Zv'*Zv,2) + 1e-12)
  ISTA step (repeat innerEIter times, e.g. 1~5):
    Ev = soft_threshold(Ev - t*grad, t*Wv)

(5) Update multiplier and penalty (HARD):
  Y  = Y + mu * (C - S)
  mu = min(mu_max, rho*mu)   (rho ~ 1.5)

Stopping criteria (recommend):
- primal residual: r = ||C - S||_F / max(1,||C||_F,||S||_F)
- stop if r < tol or iter reaches maxIter

---

## 5) Spectral Clustering (MUST use S*)
After optimization:
- Build affinity:
    W = abs(S) + abs(S)'
    W(ii)=0
- Run normalized spectral clustering to get labels (N x 1 in 1..k):
    * compute Laplacian
    * take k smallest eigenvectors
    * row-normalize
    * kmeans

---

## 6) MATLAB Project Layout (required)

mvsc_cd/
  README.md
  src/
    mvsc_cd_admm.m
    load_multiview_dataset.m
    spectral_clustering.m
    soft_threshold.m
    gst_shrink_scalar.m
    objective_value.m              (optional but recommended)
    utils_row_cell.m
    utils_standardize_labels.m
  scripts/
    run_demo.m
    inspect_dataset.m
  tests/
    test_loader_all_datasets.m
    test_sanity_synthetic.m

README must include:
- how to run run_demo.m
- dataset formats supported
- explanation of key hyperparameters: alpha, lambda, gamma, p, mu, rho
- notes on large datasets (memory)

---

## 7) Acceptance / Definition of Done
1) Loader can successfully parse ALL datasets listed in this spec without manual variable naming.
2) run_demo.m can run at least one dataset end-to-end and output labels.
3) If gt exists, report at least ACC/NMI/ARI (implement or call common MATLAB evaluation utilities).
4) Tests pass:
   - loader shapes correct: each Xcell{v} is dv x N, consistent N across views
   - labels remapped to 1..k
   - diag(Z)=0 enforced
---

## 8) Clustering Evaluation Metrics (MUST output at the end)

If ground truth labels gt are available (gt is non-empty), the demo must compute and print:

- ACC (Clustering Accuracy with best one-to-one label mapping via Hungarian)
- NMI (Normalized Mutual Information)
- Purity
- ARI (Adjusted Rand Index)
- Precision / Recall / F-score (PAIRWISE definition for clustering)

All metrics must be returned as a struct `metrics` and also printed in run_demo.m.
Suggested numeric range: [0,1]. (Optionally also print percentages.)

### 8.1 Required function
Implement:
  metrics = eval_clustering_metrics(gt, pred)

Inputs:
- gt:   N x 1 ground-truth labels (already remapped to 1..k in loader; keep meta.gt_original separately)
- pred: N x 1 predicted cluster labels from spectral clustering (k clusters)

Outputs (fields in metrics):
- metrics.ACC
- metrics.NMI
- metrics.Purity
- metrics.ARI
- metrics.Precision
- metrics.Recall
- metrics.Fscore
- metrics.confusion (contingency table, after standardization)
- metrics.pred_mapped (pred after best mapping for ACC)
- metrics.mapping (predCluster -> gtCluster)

### 8.2 Standardization (mandatory inside eval)
Inside eval_clustering_metrics:
1) Convert gt and pred to consecutive integers starting at 1 using:
   [~,~,gt]   = unique(gt(:), 'stable');
   [~,~,pred] = unique(pred(:), 'stable');
2) Let N = length(gt). Build contingency table M (size Kgt x Kpred):
   M(i,j) = #{n | gt(n)=i and pred(n)=j}
   Use accumarray for efficiency.

### 8.3 ACC (Hungarian best mapping) — mandatory
ACC must be computed after best one-to-one mapping between predicted clusters and gt clusters:
- Find assignment that maximizes sum_i M(i, assign(i))
- Use Hungarian algorithm on cost matrix:
    cost = max(M(:)) - M   (pad to square if needed)
- Map each predicted cluster id -> gt cluster id
- pred_mapped = mapping(pred)
- ACC = mean(pred_mapped == gt)

Implementation note:
- Do NOT require external toolboxes.
- Implement Hungarian in pure MATLAB (e.g., hungarian_assignment.m) OR include a self-contained Munkres implementation.

### 8.4 Purity
Purity is defined as:
  Purity = (1/N) * sum_{j=1..Kpred} max_i M(i,j)

### 8.5 NMI (geometric-mean normalization)
Use:
  Pij = M / N
  Pi  = sum_j M(i,j) / N
  Pj  = sum_i M(i,j) / N
  MI  = sum_{i,j} Pij * log( Pij / (Pi*Pj) )   (ignore zero entries)
  Hgt = -sum_i Pi * log(Pi)
  Hpr = -sum_j Pj * log(Pj)
  NMI = MI / sqrt(Hgt * Hpr + eps)

Use natural log. Ensure numerical stability with eps.

### 8.6 Pairwise Precision / Recall / F-score (clustering)
Define comb2(x) = x*(x-1)/2.

Given contingency M, row sums a_i, column sums b_j:
- TP = sum_{i,j} comb2(M(i,j))
- PredPairs = sum_j comb2(b_j)
- TruePairs = sum_i comb2(a_i)
- FP = PredPairs - TP
- FN = TruePairs - TP

Then:
- Precision = TP / (TP + FP + eps)
- Recall    = TP / (TP + FN + eps)
- Fscore    = 2*Precision*Recall / (Precision + Recall + eps)

### 8.7 ARI (Adjusted Rand Index)
Let:
- sumComb = TP = sum_{i,j} comb2(M(i,j))
- sumRow  = sum_i comb2(a_i)
- sumCol  = sum_j comb2(b_j)
- totalPairs = comb2(N)

ARI:
  expected = (sumRow * sumCol) / (totalPairs + eps)
  maxIndex = 0.5*(sumRow + sumCol)
  ARI = (sumComb - expected) / (maxIndex - expected + eps)

### 8.8 Demo output requirements
In scripts/run_demo.m:
- If gt is non-empty:
    metrics = eval_clustering_metrics(gt, labels);
    print: ACC, NMI, Purity, ARI, Precision, Recall, Fscore
- Save to a results struct:
    results.labels = labels;
    results.metrics = metrics;
    results.meta = meta;
- If gt is empty: print a warning and only output labels.

### 8.9 Tests (required)
Add tests to validate:
- metrics are within [0,1] except ARI which is in [-1,1]
- mapping does not crash when Kpred != Kgt (pad contingency to square)
- trivial case: pred==gt => ACC=1, Purity=1, Precision=1, Recall=1, Fscore=1, ARI=1, NMI=1 (within tolerance)

