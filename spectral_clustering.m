function [labels, eigvecs] = spectral_clustering(W, k, opts)
%SPECTRAL_CLUSTERING Normalized spectral clustering wrapper with stability tweaks.
%   [LABELS, EIGVECS] = SPECTRAL_CLUSTERING(W, K, OPTS)
%   Inputs:
%     W    - affinity matrix (NxN, symmetric)
%     k    - number of clusters
%     OPTS - struct with optional fields:
%            .type ('normalized' | 'unnormalized')
%            .n_init (k-means replicates)
%            .max_iter (k-means max iterations)
%            .seed (rng seed)

if nargin < 3
    opts = struct();
end
opts = fill_defaults(opts);

if isfield(opts, 'seed') && ~isempty(opts.seed)
    rng(opts.seed);
else
    rng(0);
end

if size(W,1) ~= size(W,2)
    error('Affinity matrix must be square.');
end

W = max(W, W'); % ensure symmetry
D = diag(sum(W,2));

switch lower(opts.type)
    case 'normalized'
        D_inv_sqrt = diag(1./sqrt(diag(D)+eps));
        L = eye(size(W)) - D_inv_sqrt * W * D_inv_sqrt;
    otherwise
        L = D - W;
end

% Compute k smallest eigenvectors
[eigvecs, eigvals] = eig(L, 'vector');
[~, idx] = sort(real(eigvals), 'ascend');
eigvecs = eigvecs(:, idx(1:k));

% Row-normalize eigenvectors
norms = sqrt(sum(eigvecs.^2,2));
norms(norms==0) = 1;
Y = eigvecs ./ norms;

labels = kmeans(Y, k, 'Replicates', opts.n_init, 'MaxIter', opts.max_iter, ...
    'Start', 'plus', 'Display','off');
end

function opts = fill_defaults(opts)
defaults = struct( ...
    'type', 'normalized', ...
    'n_init', 30, ...
    'max_iter', 1000, ...
    'seed', 0 ...
);
fields = fieldnames(defaults);
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(opts, f) || isempty(opts.(f))
        opts.(f) = defaults.(f);
    end
end
end
