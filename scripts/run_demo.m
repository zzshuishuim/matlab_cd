function run_demo(dataset_path, opts)
%RUN_DEMO End-to-end demo for MVSC-CD with graph polishing and autotune.
%   RUN_DEMO(DATASET_PATH, OPTS) loads the dataset, runs ADMM, performs
%   spectral clustering, and prints metrics. If DATASET_PATH is omitted or
%   missing, a synthetic two-view dataset is generated.

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
addpath(genpath(project_root));

if nargin < 1 || isempty(dataset_path)
    dataset_path = fullfile(project_root, 'datasets', 'Caltech101-7.mat');
end
if nargin < 2 || isempty(opts)
    opts = struct();
end
opts = fill_demo_defaults(opts);

% Resolve relative path
if ~isfile(dataset_path)
    dataset_path = fullfile(project_root, dataset_path);
end

if exist(dataset_path, 'file')
    fprintf('Loading dataset: %s\n', dataset_path);
    [views, gt] = load_multiview_dataset(dataset_path, opts.loader);
else
    fprintf('Dataset not found (%s). Using synthetic demo data.\n', dataset_path);
    [views, gt] = synthetic_two_view();
end

% decide k
if ~isempty(opts.k)
    k = opts.k;
elseif ~isempty(gt)
    k = numel(unique(gt));
else
    k = 3;
end
opts.k = k;

% optionally autotune lambda (light)
lambda_used = opts.solver.lambda_e;
lambda_diag = [];
if isfield(opts, 'autotune') && isfield(opts.autotune, 'lambda') && ~strcmpi(opts.autotune.lambda, 'off')
    [lambda_used, lambda_diag] = choose_lambda_light(views, gt, k, opts_for_lambda(opts), []);
    fprintf('Auto-selected lambda_e = %.3g\n', lambda_used);
end

solver_opts = opts.solver;
solver_opts.lambda_e = lambda_used;

[S_star, C, Z_set, E_set, hist] = mvsc_cd_admm(views, solver_opts); %#ok<ASGLU>

% Build affinity and cluster
[W, graph_info] = build_affinity_from_S(S_star, k, opts);

labels = spectral_clustering(W, k, opts.cluster);

fprintf('\n=== Demo Results ===\n');
fprintf('Clusters: %d | Samples: %d | Views: %d\n', k, size(W,1), numel(views));

if ~isempty(gt)
    metrics = eval_clustering_metrics(labels, gt);
    disp(metrics);
else
    fprintf('No ground truth available; labels preview:\n');
    disp(labels(1:min(10,numel(labels))));
end

% Print graph/auto info
fprintf('TopK chosen: %d\n', graph_info.topK);
fprintf('Symmetrize mode: %s\n', graph_info.symmetrizeMode);
if ~isempty(lambda_diag)
    fprintf('Lambda warmup summary:\n');
    for i = 1:numel(lambda_diag)
        fprintf('  lambda=%.3g | dens=%.3f | lcr=%.3f', lambda_diag(i).lambda, lambda_diag(i).density_raw, lambda_diag(i).largeCompRatio);
        if ~isempty(lambda_diag(i).acc)
            fprintf(' | acc=%.3f | nmi=%.3f', lambda_diag(i).acc, lambda_diag(i).nmi);
        end
        fprintf('\n');
    end
end

end

function [views, gt] = synthetic_two_view()
% Generate a simple 3-cluster dataset in 2 views.
rng(42);
n_per = 40;
centers1 = [0 0; 3 0; 0 3];
centers2 = [0 0; -3 0; 0 -3];
gt = [];
X1 = [];
X2 = [];
for i = 1:3
    X1 = [X1; bsxfun(@plus, randn(n_per,2), centers1(i,:))];
    X2 = [X2; bsxfun(@plus, randn(n_per,2), centers2(i,:))];
    gt = [gt; i*ones(n_per,1)];
end
views = {normalize_view(X1), normalize_view(X2)};
end

function X = normalize_view(X)
X = column_normalize(double(X));
end

function X = column_normalize(X)
if isempty(X), return; end
norms = sqrt(sum(X.^2,1));
norms(norms==0) = 1;
X = X ./ norms;
end

function opts = fill_demo_defaults(opts)
% defaults for demo
defaults.loader = struct('preprocess', struct('rowZScore', false, 'viewEnergyAlign', true));
defaults.solver = struct('alpha', 1, 'beta', 1, 'gamma', 0.5, 'lambda_e', 0.1, ...
    'tau', 1, 'p', 1, 'rho', 1, 'max_iter', 50, 'tol', 1e-4, 'verbose', true, 'innerEIter', 5);
defaults.graph = struct('topKEnabled', true, 'topK', 'auto', 'topKOverride', [], ...
    'symmetrizeMode', 'auto', 'colNormalize', true, 'diffusion', false);
defaults.cluster = struct('n_init', 30, 'max_iter', 1000, 'type', 'normalized', 'seed', 0);
defaults.autotune = struct('lambda', 'light');
defaults.k = [];

fields = fieldnames(defaults);
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(opts, f) || isempty(opts.(f))
        opts.(f) = defaults.(f);
    else
        opts.(f) = merge_struct(opts.(f), defaults.(f));
    end
end

end

function out = merge_struct(s, defaults)
out = defaults;
if isempty(s), return; end
fn = fieldnames(s);
for i = 1:numel(fn)
    out.(fn{i}) = s.(fn{i});
end
end

function opts_out = opts_for_lambda(opts)
% minimal solver opts for lambda warmup
opts_out = opts.solver;
opts_out.graph = opts.graph;
end
