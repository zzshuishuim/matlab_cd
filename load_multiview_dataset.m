function [views, gt, meta] = load_multiview_dataset(mat_path, opts)
%LOAD_MULTIVIEW_DATASET Robust multi-view loader with normalization and metadata.
%   [VIEWS, GT, META] = LOAD_MULTIVIEW_DATASET(MAT_PATH) loads a .mat file
%   supporting multiple common multiview formats:
%     - data + truelabel
%     - X + Y
%     - fea + gt
%     - data + label
%
%   Each view is returned as an N-by-D_v double matrix with column-wise
%   L2-normalization. Ground-truth labels GT are remapped to 1..K if
%   present. META records original fields, shapes, and remapping details.

if nargin < 2 || isempty(opts)
    opts = struct();
end
opts = fill_loader_defaults(opts);

if ~exist(mat_path, 'file')
    error('File not found: %s', mat_path);
end

raw = load(mat_path);
meta = struct();
meta.file = mat_path;
meta.original_fields = fieldnames(raw);

[views_raw, gt_raw, meta] = parse_fields(raw, meta);

% If labels are stored as per-view cells (common in some datasets), pick a
% representative label vector when lengths match.
gt_raw = standardize_label_container(gt_raw);
label_count_hint = infer_label_length(gt_raw);

[views, meta.view_shapes] = normalize_views(views_raw, gt_raw, label_count_hint, opts);

target_n = 0;
if ~isempty(views)
    target_n = size(views{1},1);
end
[gt, meta.label_map] = remap_labels(gt_raw, target_n);
meta.num_views = numel(views);
if ~isempty(views)
    meta.num_samples = size(views{1}, 1);
else
    meta.num_samples = 0;
end
meta.label_present = ~isempty(gt);

% Align label length to sample count
if meta.label_present && numel(gt) ~= meta.num_samples
    if numel(gt) > meta.num_samples
        warning('Truncating labels from %d to %d to match samples.', numel(gt), meta.num_samples);
        gt = gt(1:meta.num_samples);
    else
        warning('Label count (%d) smaller than samples (%d). Dropping gt.', numel(gt), meta.num_samples);
        gt = [];
        meta.label_present = false;
        meta.label_map = struct('original', [], 'mapped', []);
    end
end

end

function [views, gt, meta] = parse_fields(raw, meta)
views = {};
gt = [];

if isfield(raw, 'data')
    views = wrap_views(raw.data);
    if isfield(raw, 'truelabel')
        gt = raw.truelabel;
    elseif isfield(raw, 'label')
        gt = raw.label;
    end
elseif isfield(raw, 'X')
    views = wrap_views(raw.X);
    if isfield(raw, 'Y')
        gt = raw.Y;
    end
elseif isfield(raw, 'fea')
    views = wrap_views(raw.fea);
    if isfield(raw, 'gt')
        gt = raw.gt;
    end
end

if isempty(views)
    error('Unsupported dataset format in %s. Expected data/X/fea fields.', meta.file);
end

meta.detected_format = detect_format(raw);
end

function fmt = detect_format(raw)
if isfield(raw, 'data') && isfield(raw, 'truelabel')
    fmt = 'data+truelabel';
elseif isfield(raw, 'data') && isfield(raw, 'label')
    fmt = 'data+label';
elseif isfield(raw, 'X') && isfield(raw, 'Y')
    fmt = 'X+Y';
elseif isfield(raw, 'fea') && isfield(raw, 'gt')
    fmt = 'fea+gt';
elseif isfield(raw, 'data')
    fmt = 'data+?';
else
    fmt = 'unknown';
end
end

function views = wrap_views(obj)
if iscell(obj)
    views = obj(:).';
elseif isnumeric(obj)
    views = {obj};
elseif isstruct(obj)
    keys = fieldnames(obj);
    views = cellfun(@(k) obj.(k), keys, 'UniformOutput', false);
else
    error('Unrecognized view container type: %s', class(obj));
end
end

function [views, shapes] = normalize_views(raw_views, gt, label_count_hint, opts)
num_views = numel(raw_views);
views = cell(1, num_views);
shapes = zeros(num_views, 2);

if nargin < 3
    label_count_hint = numel(gt);
end
num_samples_from_gt = label_count_hint;
for v = 1:num_views
    X = double(raw_views{v});
    if ndims(X) > 2
        X = reshape(X, size(X,1), []);
	    end

    X = orient_view(X, num_samples_from_gt);
    X = apply_preprocess(X, opts);
    X = column_normalize(X);
    views{v} = X;
	    shapes(v,:) = size(X);
	end
	
	rows_all = cellfun(@(x) size(x,1), views);
	cols_all = cellfun(@(x) size(x,2), views);
	
	% If views disagree on rows but all share the same column count (typical
	% when samples are stored as columns), transpose all to samples-as-rows.
if numel(unique(cols_all)) == 1 && numel(unique(rows_all)) > 1
    for v = 1:num_views
        views{v} = column_normalize(apply_preprocess(views{v}.', opts));
        shapes(v,:) = size(views{v});
    end
    rows_all = cellfun(@(x) size(x,1), views);
    cols_all = cellfun(@(x) size(x,2), views);
end
	
	% sanity check on sample counts
	if ~isempty(views)
	    n = size(views{1},1);
	    for v = 2:num_views
	        if size(views{v},1) ~= n
	            error('View %d sample size mismatch: %d vs %d (shapes: %s)', v, size(views{v},1), n, mat2str(shapes));
	        end
	    end
    if num_samples_from_gt > 0 && n ~= num_samples_from_gt
        % transpose every view if the other dimension matches gt
        if size(views{1},2) == num_samples_from_gt
            for v = 1:num_views
                views{v} = column_normalize(apply_preprocess(views{v}.', opts));
                shapes(v,:) = size(views{v});
            end
        else
            warning('Label count (%d) does not match samples (%d). Dropping gt.', num_samples_from_gt, n);
            num_samples_from_gt = 0;
        end
    end
end
end

function X = orient_view(X, n_labels)
% Heuristic: if columns match labels or rows are smaller than cols,
% assume samples are columns and transpose to samples-as-rows.
if isempty(X)
    return;
end
if n_labels > 0 && size(X,2) == n_labels && size(X,1) ~= n_labels
    X = X.';
elseif size(X,1) < size(X,2)
    X = X.';
end
end

function X = column_normalize(X)
if isempty(X)
    return;
end
norms = sqrt(sum(X.^2,1));
norms(norms==0) = 1;
X = X ./ norms;
end

function X = apply_preprocess(X, opts)
% Optional row-wise zscore then energy alignment
if opts.preprocess.rowZScore
    X = zscore(X, 0, 2);
end
if opts.preprocess.viewEnergyAlign
    X = X ./ (norm(X,'fro') + eps);
end
end

function [gt, label_map] = remap_labels(raw_gt, target_n)
gt = raw_gt;
label_map = struct('original', [], 'mapped', []);
if isempty(raw_gt)
    return;
end

% Orient labels if possible using target sample count
if nargin < 2
    target_n = 0;
end

if isnumeric(gt) || islogical(gt) || isstring(gt) || ischar(gt)
    if ismatrix(gt) && target_n > 0
        if size(gt,1) ~= target_n && size(gt,2) == target_n
            gt = gt.';
        elseif numel(gt) == target_n
            gt = gt(:);
        end
    end
elseif iscell(gt)
    if ismatrix(gt) && target_n > 0
        if size(gt,1) ~= target_n && size(gt,2) == target_n
            gt = gt.';
        elseif numel(gt) == target_n
            gt = gt(:);
        end
    end
else
    % fallback: vectorize
    gt = gt(:);
end

gt = gt(:);
if iscell(gt)
    % Convert mixed-type cells to strings
    gt_str = cellfun(@convert_to_str, gt, 'UniformOutput', false);
    [cats, ~, ic] = unique(gt_str);
    label_map.original = cats;
    label_map.mapped = 1:numel(cats);
    gt = ic;
else
    [uniq_vals, ~, ic] = unique(gt, 'stable');
    label_map.original = uniq_vals;
    label_map.mapped = 1:numel(uniq_vals);
    gt = ic;
end
end

function gt = standardize_label_container(gt_raw)
% Handle cases where labels are stored per-view in a cell array.
gt = gt_raw;
if iscell(gt_raw) && numel(gt_raw) > 1
    % Check if every cell is a numeric/char/vector with consistent length
    lengths = cellfun(@numel, gt_raw);
    if numel(unique(lengths)) == 1 && lengths(1) > 1
        gt = gt_raw{1};
    end
end
end

function n = infer_label_length(gt_raw)
% Estimate intended sample count from label container
if isempty(gt_raw)
    n = 0; return;
end
if iscell(gt_raw)
    lengths = cellfun(@numel, gt_raw);
    n = max(lengths);
    if n == 1 && ~isempty(gt_raw{1}) && min(size(gt_raw{1})) > 1
        % e.g., matrix stored in one cell
        sz = size(gt_raw{1});
        n = max(sz);
    end
elseif isnumeric(gt_raw) || islogical(gt_raw) || isstring(gt_raw) || ischar(gt_raw)
    sz = size(gt_raw);
    n = max(sz);
else
    n = numel(gt_raw);
end
end
    
function out = convert_to_str(x)
if ischar(x)
    out = reshape(x, 1, []);
elseif isstring(x)
    sx = x(:)';
	    out = char(join(sx, "|"));
	elseif isnumeric(x) && isscalar(x)
	    out = sprintf('%.15g', x);
	elseif isnumeric(x)
	    out = sprintf('%.15g ', x(:));
	    out = strtrim(out);
	else
    out = char(string(x));
end
end

function opts = fill_loader_defaults(opts)
defaults = struct('preprocess', struct('rowZScore', false, 'viewEnergyAlign', true));
if ~isfield(opts, 'preprocess') || isempty(opts.preprocess)
    opts.preprocess = defaults.preprocess;
else
    if ~isfield(opts.preprocess, 'rowZScore') || isempty(opts.preprocess.rowZScore)
        opts.preprocess.rowZScore = defaults.preprocess.rowZScore;
    end
    if ~isfield(opts.preprocess, 'viewEnergyAlign') || isempty(opts.preprocess.viewEnergyAlign)
        opts.preprocess.viewEnergyAlign = defaults.preprocess.viewEnergyAlign;
    end
end
end
