function Wk = topk_sparsify(W, K)
%TOPK_SPARSIFY Keep top-K per column (excluding diagonal), zero the rest.
%   Wk = topk_sparsify(W, K) returns a sparse matrix retaining, for each
%   column j, the K largest off-diagonal entries of W(:,j). Symmetry is not
%   enforced here; call a symmetrization step afterward if needed.

if K <= 0
    Wk = W;
    return;
end

N = size(W,1);
Wk = zeros(size(W));

for j = 1:size(W,2)
    col = W(:,j);
    col(j) = -inf; % exclude diagonal
    [~, idx] = maxk(col, min(K, N-1));
    Wk(idx, j) = W(idx, j);
end

Wk(1:N+1:end) = 0;
end
