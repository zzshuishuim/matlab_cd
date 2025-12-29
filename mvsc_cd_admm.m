function [S_star, C, Z_set, E_set, history] = mvsc_cd_admm(views, opts)
%MVSC_CD_ADMM Multi-view Subspace Clustering with Consensus + Diversity (ADMM).
%   [S_STAR, C, Z_SET, E_SET, HISTORY] = MVSC_CD_ADMM(VIEWS, OPTS)
%   runs the ADMM optimizer. Inputs:
%     VIEWS : 1xV cell, each N-by-D view matrix (columns normalized).
%     OPTS  : struct with fields (defaults shown):
%         alpha      - Schatten-p weight on S (default 1)
%         beta       - consensus strength for C (default 1)
%         gamma      - diversity complement weight (default 1)
%         lambda_e   - sparsity weight for E (default 1e-1)
%         tau        - nuclear/Schatten threshold for S (default 1)
%         p          - Schatten exponent in (0,1] (default 1)
%         rho        - ADMM penalty parameter (default 1)
%         max_iter   - maximum iterations (default 100)
%         tol        - convergence tolerance (default 1e-4)
%
%   Updates:
%     - Z via Sylvester equation
%     - C via closed-form linear equation
%     - S via Schatten-p proximal (p=1 soft, p<1 GST)
%     - E via block coordinate weighted L1 (ISTA-like)
%     - Y dual ascent with C-S
%
%   The final affinity for spectral clustering is |S_star|.

if nargin < 2 || isempty(opts)
    opts = struct();
end
opts = fill_defaults(opts);

V = numel(views);
if V == 0
    error('No views provided.');
end
N = size(views{1},1);

Z_set = cell(1,V);
E_set = cell(1,V);
for v = 1:V
    Z_set{v} = zeros(N);
    E_set{v} = zeros(size(views{v}));
end
C = zeros(N);
S_star = zeros(N);
Y = zeros(N);

history = struct('primal', [], 'dual', [], 'obj', []);

alpha = opts.alpha; beta = opts.beta; gamma = opts.gamma;
lambda_e = opts.lambda_e; tau = opts.tau; p = opts.p;
rho = opts.rho; max_iter = opts.max_iter; tol = opts.tol;
verbose = opts.verbose; innerEIter = opts.innerEIter;

    % Use sample-as-rows formulation: X is N x d, so XtX is N x N
    XtX = cellfun(@(X) X * X.', views, 'UniformOutput', false);

for iter = 1:max_iter
    Z_prev = Z_set;
    C_prev = C;
    S_prev = S_star;

    % === Z update (Sylvester) ===
    for v = 1:V
        Xv = views{v};
        Ev = E_set{v};
        A = XtX{v} + (beta + gamma*(V-1))*eye(N);
        rhs = (Xv - Ev) * Xv.' + beta*C + gamma*sum_except(Z_prev, v);
        % Sylvester form A*Z + Z*0 = rhs
        Zv = sylvester(A, zeros(N), rhs);
        Zv(1:N+1:end) = 0; % enforce diag(Z)=0
        Z_set{v} = Zv;
    end

    % === C update (linear equation) ===
    sumZ = zeros(N);
    for v = 1:V
        sumZ = sumZ + Z_set{v};
    end
    C = (beta*sumZ + rho*(S_star - Y)) / (beta*V + rho);
    C(1:N+1:end) = 0;

    % === S update (Schatten-p proximal) ===
    S_input = (gamma * sumZ + rho * (C + Y)) / (rho + gamma*(V-1));
    S_star = proximal_schatten(S_input, tau/(rho + gamma*(V-1)), p);
    S_star = (S_star + S_star.')/2; % symmetrize for stability
    S_star(1:N+1:end) = 0;

    % === E update (weighted L1 via ISTA with inner loops) ===
    for v = 1:V
        Xv = views{v};
        Zv = Z_set{v};
        E_v = E_set{v};
        for ii = 1:innerEIter
            residual = Xv - Zv*Xv - E_v;
            W = 1 ./ (abs(E_v) + 1e-6);
            grad = -residual;
            step = 1.0; % simple ISTA step; could be tuned by Lipschitz est
            E_temp = E_v - step * grad;
            E_v = soft_threshold(E_temp, lambda_e * step .* W);
        end
        E_set{v} = E_v;
    end

    % === Dual update ===
    Y = Y + (C - S_star);

    % === Convergence diagnostics ===
    primal_res = 0;
    for v = 1:V
        primal_res = primal_res + norm(Z_set{v} - C, 'fro')^2;
    end
    primal_res = sqrt(primal_res) + norm(C - S_star, 'fro');
    dual_res = norm(C - C_prev, 'fro') + norm(S_star - S_prev, 'fro');
    obj_val = objective_value(views, Z_set, E_set, C, S_star, alpha, beta, gamma, lambda_e, tau, p);

    history.primal(end+1) = primal_res;
    history.dual(end+1) = dual_res;
    history.obj(end+1) = obj_val;

    if verbose && mod(iter, 5) == 0
        r_cs = norm(C - S_star, 'fro') / max(1, norm(S_star, 'fro'));
        fprintf('Iter %03d | obj %.4e | primal %.3e | dual %.3e | r(C-S)=%.3e\n', ...
            iter, obj_val, primal_res, dual_res, r_cs);
    end

    if primal_res < tol && dual_res < tol
        break;
    end
end

function Zsum = sum_except(Z_set, idx)
Zsum = zeros(size(Z_set{1}));
for i = 1:numel(Z_set)
    if i == idx, continue; end
    Zsum = Zsum + Z_set{i};
end
end

function X = proximal_schatten(M, tau, p)
[U,S,V] = svd(M, 'econ');
s = diag(S);
if isempty(s)
    X = M;
    return;
end
if p >= 1
    s_new = max(s - tau, 0);
else
    s_new = arrayfun(@(val) gst_scalar(val, tau, p), s);
end
S_new = diag(s_new);
X = U * S_new * V';
end

function x = gst_scalar(s, tau, p)
if s <= 0
    x = 0;
    return;
end
threshold = (2*tau*(1-p))^(1/(2-p));
if s <= threshold
    x = 0;
    return;
end
% Fixed-point iteration
x = s;
for i = 1:50
    x_old = x;
    x = s - tau * p * x^(p-1);
    if abs(x - x_old) < 1e-6
        break;
    end
    x = max(x, 0);
end
end

function E = soft_threshold(X, T)
E = sign(X) .* max(abs(X) - T, 0);
end

function val = objective_value(views, Z_set, E_set, C, S, alpha, beta, gamma, lambda_e, tau, p)
V = numel(views);
val = 0;
for v = 1:V
    Xv = views{v};
    Zv = Z_set{v};
    Ev = E_set{v};
    val = val + 0.5 * norm(Xv - Zv*Xv - Ev, 'fro')^2 + lambda_e * sum(abs(Ev), 'all') ...
        + 0.5*beta*norm(Zv - C, 'fro')^2 + 0.5*gamma*norm(Zv - S, 'fro')^2;
end
% Schatten term on S
[~, Sval, ~] = svd(S, 'econ');
sing = diag(Sval);
if p >= 1
    val = val + tau * sum(sing);
else
    val = val + tau * sum(sing.^p);
end

end

function opts = fill_defaults(opts)
defaults = struct( ...
    'alpha', 1, ...
    'beta', 1, ...
    'gamma', 1, ...
    'lambda_e', 1e-1, ...
    'tau', 1, ...
    'p', 1, ...
    'rho', 1, ...
    'max_iter', 100, ...
    'tol', 1e-4, ...
    'verbose', false, ...
    'innerEIter', 5 ...
);
fields = fieldnames(defaults);
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(opts, f) || isempty(opts.(f))
        opts.(f) = defaults.(f);
    end
end
end
