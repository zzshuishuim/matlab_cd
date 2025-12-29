function run_demo(dataset_path)
%RUN_DEMO End-to-end demo for MVSC-CD.
%   RUN_DEMO(DATASET_PATH) loads the dataset, runs ADMM, performs spectral
%   clustering, and prints metrics if ground truth exists. If DATASET_PATH
%   is omitted or missing, a synthetic two-view dataset is generated.

if nargin < 1
    dataset_path = 'datasets/Caltech101-7.mat';
end

if exist(dataset_path, 'file')
    fprintf('Loading dataset: %s\n', dataset_path);
    [views, gt] = load_multiview_dataset(dataset_path);
else
    fprintf('Dataset not found (%s). Using synthetic demo data.\n', dataset_path);
    [views, gt] = synthetic_two_view();
end

opts = struct('alpha', 1, 'beta', 1, 'gamma', 0.5, 'lambda_e', 0.1, ...
    'tau', 1, 'p', 1, 'rho', 1, 'max_iter', 50, 'tol', 1e-4, 'verbose', true);

[S_star, C, Z_set, E_set, hist] = mvsc_cd_admm(views, opts); %#ok<ASGLU>

% Build affinity and cluster
W = abs(S_star);
W = (W + W.')/2;

if ~isempty(gt)
    k = numel(unique(gt));
else
    k = 3;
end
[labels, ~] = spectral_clustering(W, k);

fprintf('\n=== Demo Results ===\n');
fprintf('Clusters: %d | Samples: %d | Views: %d\n', k, size(W,1), numel(views));

if ~isempty(gt)
    metrics = eval_clustering_metrics(labels, gt);
    disp(metrics);
else
    fprintf('No ground truth available; labels preview:\n');
    disp(labels(1:min(10,numel(labels))));
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
