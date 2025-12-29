function tests = test_graph_tools
tests = functiontests(localfunctions);
end

function test_topk_and_symmetrize_max(t)
N = 20;
W = rand(N);
Wk = topk_sparsify(W, 5);
Wsym = max(Wk, Wk');
t.verifyEqual(Wsym, Wsym', 'AbsTol', 1e-12);
t.verifyEqual(diag(Wsym), zeros(N,1));
t.verifyGreaterThanOrEqual(Wsym(:), 0);
end

function test_choose_topk_info(t)
N = 50; k = 5;
W = rand(N);
[K, info] = choose_topk(W, k, struct());
t.verifyGreaterThanOrEqual(K, 5);
t.verifyLessThanOrEqual(K, min(N-1,200));
t.verifyTrue(isfield(info, 'metrics'));
t.verifyTrue(~isempty(info.metrics));
end

function test_build_affinity(t)
N = 30; k = 3;
S = randn(N);
[W, info] = build_affinity_from_S(S, k, struct());
t.verifyEqual(W, W', 'AbsTol', 1e-12);
t.verifyEqual(diag(W), zeros(N,1));
t.verifyGreaterThanOrEqual(min(W(:)), 0);
t.verifyTrue(isfield(info, 'topK'));
t.verifyTrue(isfield(info, 'symmetrizeMode'));
end
