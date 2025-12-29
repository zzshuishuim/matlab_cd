function [views, gt, meta] = load_multiview_dataset(mat_path)
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

arguments
    mat_path (1,:) char
end

if ~exist(mat_path, 'file')
    error('File not found: %s', mat_path);
end

raw = load(mat_path);
meta = struct();
meta.file = mat_path;
meta.original_fields = fieldnames(raw);

[views_raw, gt_raw, meta] = parse_fields(raw, meta);

[views, meta.view_shapes] = normalize_views(views_raw, gt_raw);

[gt, meta.label_map] = remap_labels(gt_raw);
meta.num_views = numel(views);
if ~isempty(views)
    meta.num_samples = size(views{1}, 1);
else
    meta.num_samples = 0;
end
meta.label_present = ~isempty(gt_raw);

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

function [views, shapes] = normalize_views(raw_views, gt)
num_views = numel(raw_views);
views = cell(1, num_views);
shapes = zeros(num_views, 2);

num_samples_from_gt = numel(gt);
for v = 1:num_views
    X = double(raw_views{v});
    if ndims(X) > 2
        X = reshape(X, size(X,1), []);
    end

    X = orient_view(X, num_samples_from_gt);
    X = column_normalize(X);
    views{v} = X;
    shapes(v,:) = size(X);
end

% sanity check on sample counts
if ~isempty(views)
    n = size(views{1},1);
    for v = 2:num_views
        if size(views{v},1) ~= n
            error('View %d sample size mismatch: %d vs %d', v, size(views{v},1), n);
        end
    end
    if num_samples_from_gt > 0 && n ~= num_samples_from_gt
        % transpose every view if the other dimension matches gt
        if size(views{1},2) == num_samples_from_gt
            for v = 1:num_views
                views{v} = views{v}.';
                views{v} = column_normalize(views{v});
                shapes(v,:) = size(views{v});
            end
        else
            warning('Label count (%d) does not match samples (%d). Proceeding.', num_samples_from_gt, n);
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

function [gt, label_map] = remap_labels(raw_gt)
gt = raw_gt;
label_map = struct('original', [], 'mapped', []);
if isempty(raw_gt)
    return;
end

gt = raw_gt(:);
if iscell(gt)
    % Convert string cells to categorical numbers
    [cats, ~, ic] = unique(gt);
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
