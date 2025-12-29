function metrics = eval_clustering_metrics(pred, gt)
%EVAL_CLUSTERING_METRICS Compute ACC, NMI, Purity, ARI, Precision/Recall/Fscore.
%   METRICS = EVAL_CLUSTERING_METRICS(PRED, GT) where PRED and GT are
%   1-by-N integer label vectors. ACC uses Hungarian best mapping.
%   Precision/Recall/Fscore are pairwise definitions.

pred = pred(:);
gt = gt(:);
n = numel(gt);

if n == 0 || numel(pred) ~= n
    metrics = struct('ACC', [], 'NMI', [], 'Purity', [], 'ARI', [], ...
        'Precision', [], 'Recall', [], 'Fscore', []);
    return;
end

% ACC via Hungarian matching
conf = confusion_matrix(gt, pred);
cost = max(conf, [], 'all') - conf;
[assign, ~] = hungarian(cost);
acc = sum(conf(sub2ind(size(conf), 1:size(conf,1), assign))) / n;

% NMI
nmi = nmi_score(gt, pred);

% Purity
purity = sum(max(conf, [], 2)) / n;

% ARI
ari = adjusted_rand_index(gt, pred);

% Pairwise precision/recall/F1
[precision, recall, fscore] = pairwise_prf(gt, pred);

metrics = struct('ACC', acc, 'NMI', nmi, 'Purity', purity, ...
    'ARI', ari, 'Precision', precision, 'Recall', recall, 'Fscore', fscore);
end

function conf = confusion_matrix(gt, pred)
gt_classes = unique(gt);
pred_classes = unique(pred);
conf = zeros(numel(gt_classes), numel(pred_classes));
for i = 1:numel(gt_classes)
    for j = 1:numel(pred_classes)
        conf(i,j) = sum(gt == gt_classes(i) & pred == pred_classes(j));
    end
end
end

function nmi = nmi_score(gt, pred)
gt_classes = unique(gt);
pred_classes = unique(pred);
n = numel(gt);

% probabilities
Pgt = zeros(numel(gt_classes),1);
Ppred = zeros(numel(pred_classes),1);
for i = 1:numel(gt_classes)
    Pgt(i) = sum(gt == gt_classes(i)) / n;
end
for j = 1:numel(pred_classes)
    Ppred(j) = sum(pred == pred_classes(j)) / n;
end

MI = 0;
for i = 1:numel(gt_classes)
    for j = 1:numel(pred_classes)
        pij = sum(gt == gt_classes(i) & pred == pred_classes(j)) / n;
        if pij > 0
            MI = MI + pij * log(pij / (Pgt(i)*Ppred(j)));
        end
    end
end
Hgt = -sum(Pgt .* log(Pgt + eps));
Hpred = -sum(Ppred .* log(Ppred + eps));
nmi = MI / sqrt(Hgt * Hpred + eps);
end

function ari = adjusted_rand_index(gt, pred)
gt = gt(:); pred = pred(:);
n = numel(gt);

% contingency table
classes1 = unique(gt);
classes2 = unique(pred);
cont = zeros(numel(classes1), numel(classes2));
for i = 1:numel(classes1)
    for j = 1:numel(classes2)
        cont(i,j) = sum(gt == classes1(i) & pred == classes2(j));
    end
end

sum_comb_c = sum(nchoosek_vector(sum(cont,2), 2));
sum_comb_k = sum(nchoosek_vector(sum(cont,1), 2));
sum_comb = sum(nchoosek_vector(cont(:), 2));

total_comb = nchoosek(n,2);
expected = (sum_comb_c * sum_comb_k) / total_comb;
max_index = 0.5 * (sum_comb_c + sum_comb_k);
ari = (sum_comb - expected) / (max_index - expected + eps);
end

function v = nchoosek_vector(x, k)
x = x(:);
v = arrayfun(@(t) nchoosek_safe(t, k), x);
end

function val = nchoosek_safe(n, k)
if n < k
    val = 0;
else
    val = nchoosek(n, k);
end
end

function [precision, recall, fscore] = pairwise_prf(gt, pred)
n = numel(gt);
same_gt = false(n);
same_pred = false(n);
for i = 1:n
    same_gt(i,:) = gt(i)==gt.';
    same_pred(i,:) = pred(i)==pred.';
end
% Only upper triangular pairs
triu_mask = triu(true(n),1);
tp = sum(same_gt(triu_mask) & same_pred(triu_mask));
fp = sum(~same_gt(triu_mask) & same_pred(triu_mask));
fn = sum(same_gt(triu_mask) & ~same_pred(triu_mask));

precision = tp / (tp + fp + eps);
recall = tp / (tp + fn + eps);
fscore = 2*precision*recall / (precision + recall + eps);
end

function [assignment, total_cost] = hungarian(costMat)
%HUNGARIAN Simple Hungarian algorithm for square matrices.
%   Returns column assignment for each row.
costMat = double(costMat);
[nRows, nCols] = size(costMat);
n = max(nRows, nCols);
pad = zeros(n);
pad(1:nRows, 1:nCols) = costMat;

% Step 1: Row and column reduction
pad = pad - min(pad, [], 2);
pad = pad - min(pad, [], 1);

% Masks
starZ = false(n);
primeZ = false(n);
coveredRows = false(n,1);
coveredCols = false(n,1);

% Initial starring
for i = 1:n
    for j = 1:n
        if pad(i,j) == 0 && ~coveredRows(i) && ~coveredCols(j)
            starZ(i,j) = true;
            coveredRows(i) = true;
            coveredCols(j) = true;
        end
    end
end
coveredRows(:) = false; coveredCols(:) = false;

% Cover columns with starred zeros
coveredCols(any(starZ,1)) = true;

while ~all(coveredCols)
    [r,c] = find_zero(pad, coveredRows, coveredCols);
    if r == -1
        % adjust matrix
        minval = min(pad(~coveredRows, ~coveredCols), [], 'all');
        pad(~coveredRows, ~coveredCols) = pad(~coveredRows, ~coveredCols) - minval;
        pad(coveredRows, coveredCols) = pad(coveredRows, coveredCols) + minval;
        continue;
    end
    primeZ(r,c) = true;
    star_col = find(starZ(r,:),1);
    if isempty(star_col)
        starZ = augment_path(starZ, primeZ, r, c);
        primeZ(:) = false;
        coveredRows(:) = false;
        coveredCols(:) = false;
        coveredCols(any(starZ,1)) = true;
    else
        coveredRows(r) = true;
        coveredCols(star_col) = false;
    end
end

assignment = zeros(1, nRows);
for i = 1:nRows
    j = find(starZ(i,1:nCols),1);
    if ~isempty(j)
        assignment(i) = j;
    else
        assignment(i) = 1;
    end
end
total_cost = sum(costMat(sub2ind(size(costMat), 1:nRows, assignment)));
end

function [r,c] = find_zero(pad, coveredRows, coveredCols)
r = -1; c = -1;
for i = 1:size(pad,1)
    if coveredRows(i), continue; end
    for j = 1:size(pad,2)
        if coveredCols(j), continue; end
        if pad(i,j) == 0
            r = i; c = j; return;
        end
    end
end
end

function starZ = augment_path(starZ, primeZ, r, c)
path = [r c];
done = false;
while ~done
    star_row = find(starZ(:, path(end,2)),1);
    if isempty(star_row)
        done = true;
        break;
    end
    path = [path; star_row path(end,2)]; %#ok<AGROW>
    prime_col = find(primeZ(path(end,1),:),1);
    path = [path; path(end,1) prime_col]; %#ok<AGROW>
end
% flip stars and primes along path
for k = 1:size(path,1)
    if mod(k,2)==1
        starZ(path(k,1), path(k,2)) = ~starZ(path(k,1), path(k,2));
    else
        starZ(path(k,1), path(k,2)) = false;
    end
end
end
