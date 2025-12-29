function results = tune_params_grid(dataset_path, grid_opts)
%TUNE_PARAMS_GRID Lightweight grid search over a few hyper-parameters.
%   RESULTS = TUNE_PARAMS_GRID(DATASET_PATH, GRID_OPTS)
%   Runs short MVSC-CD warmups over a small grid of alpha/lambda/gamma/p and
%   ranks by ACC/NMI if gt available (otherwise by NMI surrogate).

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
addpath(genpath(project_root));

if nargin < 1 || isempty(dataset_path)
    error('dataset_path is required');
end
if nargin < 2
    grid_opts = struct();
end

grid = default_grid();
fields = fieldnames(grid);
for i = 1:numel(fields)
    f = fields{i};
    if isfield(grid_opts, f) && ~isempty(grid_opts.(f))
        grid.(f) = grid_opts.(f);
    end
end

[views, gt, meta] = load_multiview_dataset(dataset_path);
if isempty(gt)
    warning('No ground truth available; metrics will be unsupervised.');
end

if ~isempty(gt)
    k = numel(unique(gt));
else
    k = grid.k_fallback;
end

combos = allcomb(grid.alpha, grid.lambda_e, grid.gamma, grid.p);
short_iters = grid.warmupIter;

records = [];
for i = 1:size(combos,1)
    a = combos(i,1); lam = combos(i,2); g = combos(i,3); pval = combos(i,4);
    opts = default_solver_opts();
    opts.alpha = a; opts.lambda_e = lam; opts.gamma = g; opts.p = pval;
    opts.max_iter = short_iters;
    opts.verbose = false;
    [S_star, C, Z_set, E_set] = mvsc_cd_admm(views, opts); %#ok<ASGLU>
    [W, ~] = build_affinity_from_S(S_star, k, struct());
    labels = spectral_clustering(W, k, struct('n_init', 10, 'max_iter', 300, 'seed', 0));
    if ~isempty(gt)
        m = eval_clustering_metrics(labels, gt);
        acc = m.ACC; nmi = m.NMI; ari = m.ARI;
    else
        acc = NaN; nmi = NaN; ari = NaN;
    end
    records = [records; struct('alpha', a, 'lambda_e', lam, 'gamma', g, 'p', pval, ...
        'ACC', acc, 'NMI', nmi, 'ARI', ari)];
end

% pick top 3 by ACC then NMI (if no gt, just return all)
if ~isempty(gt)
    acc_vals = arrayfun(@(r) r.ACC, records);
    nmi_vals = arrayfun(@(r) r.NMI, records);
    [~, idx] = sortrows([acc_vals(:), nmi_vals(:)], [-1 -1]);
    top_idx = idx(1:min(3,numel(idx)));
else
    top_idx = 1:numel(records);
end

results = records(top_idx);

% Re-run top entries with full iterations
full_iters = grid.fullIter;
final_records = [];
for i = 1:numel(results)
    r = results(i);
    opts = default_solver_opts();
    opts.alpha = r.alpha; opts.lambda_e = r.lambda_e; opts.gamma = r.gamma; opts.p = r.p;
    opts.max_iter = full_iters;
    [S_star, C, Z_set, E_set] = mvsc_cd_admm(views, opts); %#ok<ASGLU>
    [W, ~] = build_affinity_from_S(S_star, k, struct());
    labels = spectral_clustering(W, k, struct('seed', 0));
    if ~isempty(gt)
        m = eval_clustering_metrics(labels, gt);
        r.ACC = m.ACC; r.NMI = m.NMI; r.ARI = m.ARI;
    end
    final_records = [final_records; r]; %#ok<AGROW>
end

results = final_records;

% save
if ~exist(fullfile(project_root, 'results'), 'dir')
    mkdir(fullfile(project_root, 'results'));
end
[~, base, ~] = fileparts(dataset_path);
save(fullfile(project_root, 'results', ['tuning_' base '.mat']), 'results');

disp('Top configurations:');
disp(results);

end

function grid = default_grid()
grid.alpha = [0.1, 1, 10];
grid.lambda_e = [1e-3, 1e-2, 1e-1];
grid.gamma = [1e-4, 1e-3, 1e-2];
grid.p = [1, 0.8];
grid.warmupIter = 30;
grid.fullIter = 80;
grid.k_fallback = 3;
end

function opts = default_solver_opts()
opts = struct('alpha',1,'beta',1,'gamma',0.5,'lambda_e',0.1,'tau',1,'p',1,'rho',1, ...
    'max_iter',50,'tol',1e-4,'verbose',false,'innerEIter',5);
end

function combos = allcomb(varargin)
% simple cartesian product
args = varargin;
n = numel(args);
grid = cell(1,n);
[grid{:}] = ndgrid(args{:});
for i = 1:n
    grid{i} = grid{i}(:);
end
combos = [grid{:}];
end
