function [W, info] = build_affinity_from_S(S, k, opts)
%BUILD_AFFINITY_FROM_S Graph polishing from S* to affinity W.
%   [W, INFO] = BUILD_AFFINITY_FROM_S(S, K, OPTS)
%   applies post-processing steps (diag removal, symmetrization, Top-K,
%   normalization, optional diffusion). This does not change the core
%   MVSC-CD model; it only affects the spectral clustering input.

if nargin < 3 || isempty(opts)
    opts = struct();
end
opts = fill_graph_defaults(opts);

N = size(S,1);
info = struct();

% 1) remove diagonal
S = S - diag(diag(S));

% 2) base graph
W_raw = abs(S) + abs(S)';
info.density_raw = nnz(W_raw > 0) / max(1, N*(N-1));

% 3) Top-K sparsify (with auto K)
if opts.graph.topKEnabled
    if strcmpi(opts.graph.topK, 'auto')
        [K, topk_info] = choose_topk(W_raw, k, opts.graph);
    else
        K = opts.graph.topK;
        topk_info = struct('Kcands', K, 'metrics', struct('K', K), 'chosen', K);
    end
    if isfield(opts.graph, 'topKOverride') && ~isempty(opts.graph.topKOverride)
        K = opts.graph.topKOverride;
    end
    Wk = topk_sparsify(W_raw, K);
else
    K = 0;
    topk_info = struct('Kcands', [], 'metrics', [], 'chosen', []);
    Wk = W_raw;
end
info.topK = K;
info.topk_info = topk_info;

% 4) Symmetrization (max/avg/auto)
sym_mode = opts.graph.symmetrizeMode;
if strcmpi(sym_mode, 'auto')
    density_k = nnz(Wk > 0) / max(1, N*(N-1));
    A = (Wk > 0) | (Wk' > 0);
    lcr = graph_largest_component_ratio(A);
    if density_k > 0.08
        sym_mode = 'max';
    elseif lcr < 0.90 || density_k < 0.01
        sym_mode = 'avg';
    else
        sym_mode = 'max';
    end
    info.sym_auto = struct('density_k', density_k, 'largeCompRatio', lcr);
end
switch lower(sym_mode)
    case 'max'
        W = max(Wk, Wk');
    otherwise % 'avg'
        W = 0.5 * (Wk + Wk');
end
info.symmetrizeMode = sym_mode;

% 5) remove diagonal again
W(1:N+1:end) = 0;

% 6) column normalization (optional) with re-symmetrization
if opts.graph.colNormalize
    W = W ./ (max(W, [], 1) + eps);
    if strcmpi(sym_mode, 'max')
        W = max(W, W');
    else
        W = 0.5 * (W + W');
    end
    W(1:N+1:end) = 0;
end

% 7) optional diffusion
if opts.graph.diffusion
    W = W * W;
    W = W ./ (max(W, [], 1) + eps);
    if strcmpi(sym_mode, 'max')
        W = max(W, W');
    else
        W = 0.5 * (W + W');
    end
    W(1:N+1:end) = 0;
end

% Ensure non-negative symmetry
W = max(W, 0);
W = (W + W.')/2;
W(1:N+1:end) = 0;

end

function opts = fill_graph_defaults(opts)
if ~isfield(opts, 'graph') || isempty(opts.graph)
    opts.graph = struct();
end
g = opts.graph;

defaults = struct( ...
    'topKEnabled', true, ...
    'topK', 'auto', ...
    'topKOverride', [], ...
    'symmetrizeMode', 'auto', ...
    'colNormalize', true, ...
    'diffusion', false ...
);
fields = fieldnames(defaults);
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(g, f) || isempty(g.(f))
        g.(f) = defaults.(f);
    end
end
opts.graph = g;
end
