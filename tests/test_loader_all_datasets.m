function tests = test_loader_all_datasets
tests = functiontests(localfunctions);
end

function test_all_datasets(~)
paths = {
    'datasets/3sources.mat'
    'datasets/100leaves.mat'
    'datasets/BBC.mat'
    'datasets/BBCSport.mat'
    'datasets/Caltech101-7.mat'
    'datasets/Caltech101-20.mat'
    'datasets/coil-20.mat'
    'datasets/Hdigit.mat'
    'datasets/HW.mat'
    'datasets/HW2sources.mat'
    'datasets/Mfeat.mat'
    'datasets/MSRC_v1.mat'
    'datasets/NGs.mat'
    'datasets/NUS.mat'
    'datasets/ORL.mat'
    'datasets/UCI_Digits.mat'
    'datasets/WebKB.mat'
    'datasets/YaleB10_3_650.mat'
    };

for i = 1:numel(paths)
    path = paths{i};
    if ~exist(path, 'file')
        warning('Skipping missing dataset: %s', path);
        continue;
    end
    [views, gt] = load_multiview_dataset(path);
    num_views = numel(views);
    assert(num_views >= 1, 'No views detected for %s', path);
    n = size(views{1},1);
    for v = 1:num_views
        assert(size(views{v},1) == n, 'Sample size mismatch in %s view %d', path, v);
    end
    if ~isempty(gt)
        assert(numel(gt) == n, 'Label size mismatch in %s', path);
    end
end
end
