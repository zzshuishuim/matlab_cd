function [K, info] = choose_topk(W_raw, k, opts)
%CHOOSE_TOPK Adaptive Top-K selection based on connectivity/density.

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'topK'), opts.topK = 'auto'; end

N = size(W_raw,1);
K0 = max(8, ceil(N/(2*k)));
if N >= 2000
    K0 = max(12, ceil(N/(3*k)));
end
Kcand = unique(round([0.6,0.8,1.0,1.2,1.5]*K0));
Kcand = max(5, Kcand);
Kcand = min(min(N-1,200), Kcand);

metrics = struct('K', {}, 'zeroDegRatio', {}, 'largeCompRatio', {});

bestK = Kcand(1);
bestLCR = -inf;

for i = 1:numel(Kcand)
    Kc = Kcand(i);
    Wk = topk_sparsify(W_raw, Kc);
    deg = sum(Wk>0,2);
    zeroDegRatio = mean(deg==0);
    A = (Wk>0) | (Wk'>0);
    lcr = graph_largest_component_ratio(A);
    metrics(i).K = Kc;
    metrics(i).zeroDegRatio = zeroDegRatio;
    metrics(i).largeCompRatio = lcr;

    if lcr >= 0.95 && zeroDegRatio <= 0.01
        if Kc < bestK || bestLCR < 0.95
            bestK = Kc; bestLCR = lcr;
        end
    elseif lcr > bestLCR || (lcr == bestLCR && Kc > bestK)
        bestK = Kc; bestLCR = lcr;
    end
end

K = bestK;
info = struct('Kcands', Kcand, 'metrics', metrics, 'chosen', K);
end
