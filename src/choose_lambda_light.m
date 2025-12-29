function [lambda_best, diag_info] = choose_lambda_light(Xcell, gt, k, base_opts, lambda_cands)
%CHOOSE_LAMBDA_LIGHT Lightweight lambda selection via short warmup runs.

if nargin < 5 || isempty(lambda_cands)
    N = size(Xcell{1},1);
    if N < 1500
        lambda_cands = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1];
    else
        lambda_cands = [1e-2, 3e-2, 1e-1, 3e-1];
    end
end

% ensure graph struct exists
if ~isfield(base_opts, 'graph') || isempty(base_opts.graph)
    base_opts.graph = struct('topKEnabled', true);
end

warmupIter = 12;
if size(Xcell{1},1) > 2200
    warmupIter = 8;
end

diag_info = struct('lambda', {}, 'density_raw', {}, 'largeCompRatio', {}, 'acc', {}, 'nmi', {});

best_idx = 1;
best_acc = -inf;
best_nmi = -inf;

for i = 1:numel(lambda_cands)
    lam = lambda_cands(i);
    opts_try = base_opts;
    opts_try.lambda_e = lam;
    opts_try.max_iter = warmupIter;
    opts_try.verbose = false;
    opts_try.graph.topKEnabled = false; % use raw S for density

    [S_try, ~, ~, ~, ~] = mvsc_cd_admm(Xcell, opts_try);
    W_raw = abs(S_try);
    W_raw = W_raw - diag(diag(W_raw));
    W_raw = W_raw + W_raw';
    density_raw = nnz(W_raw > 0) / max(1, numel(W_raw) - size(W_raw,1));
    A = (W_raw > 0);
    lcr = graph_largest_component_ratio(A);

    acc_i = []; nmi_i = [];
    if ~isempty(gt)
        [labels_i, ~] = spectral_clustering(W_raw, k, struct('n_init', 5, 'max_iter', 100));
        m = eval_clustering_metrics(labels_i, gt);
        acc_i = m.ACC; nmi_i = m.NMI;
    end

    diag_info(i).lambda = lam;
    diag_info(i).density_raw = density_raw;
    diag_info(i).largeCompRatio = lcr;
    diag_info(i).acc = acc_i;
    diag_info(i).nmi = nmi_i;

    if ~isempty(gt)
        if acc_i > best_acc || (acc_i == best_acc && nmi_i > best_nmi)
            best_idx = i; best_acc = acc_i; best_nmi = nmi_i;
        end
    else
        % pick density in [0.01,0.10] with better connectivity
        target_density = (density_raw >= 0.01 && density_raw <= 0.10);
        if target_density && (lcr > best_nmi) % reuse best_nmi as best connectivity tracker
            best_idx = i; best_nmi = lcr;
        elseif ~target_density && lcr > best_nmi
            best_idx = i; best_nmi = lcr;
        end
    end
end

lambda_best = lambda_cands(best_idx);
end
