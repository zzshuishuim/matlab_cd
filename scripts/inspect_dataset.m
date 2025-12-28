function inspect_dataset()
%INSPECT_DATASET Iterate through predefined datasets and report shapes.

datasets = {
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

fprintf('Inspecting %d datasets...\n', numel(datasets));
for i = 1:numel(datasets)
    path = datasets{i};
    fprintf('\n[%02d/%02d] %s\n', i, numel(datasets), path);
    if ~exist(path, 'file')
        warning('File missing: %s (skipping)', path);
        continue;
    end
    try
        [views, gt, meta] = load_multiview_dataset(path);
        fprintf('  Views: %d | Samples: %d | Labels: %d\n', meta.num_views, meta.num_samples, numel(unique(gt)));
        for v = 1:numel(views)
            fprintf('    View %d shape: [%d x %d]\n', v, size(views{v},1), size(views{v},2));
        end
    catch ME
        warning('Failed to load %s: %s', path, ME.message);
    end
end
end
