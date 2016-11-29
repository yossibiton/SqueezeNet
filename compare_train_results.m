close all; clc; clearvars;

output_names = {'grayscale_128px', 'grayscale_128px_BN', 'grayscale_128px_BN_try01'};
output_dir = 'output/SqueezeNet_v1.1';
output_dirs = cellfun(@(x)fullfile(output_dir, x), output_names, ...
    'UniformOutput', false);

field_requested = {'TrainingLoss' , 'TestLoss'};
filenames = {'train', 'test'};
losses = cell(1, 2);
for k = 1:length(output_dirs)
    loss = struct;
    for m = 1:length(field_requested)
       filename = ['log.' filenames{m}];
       log_path = fullfile(output_dirs{k}, filename);

       fid = fopen(log_path);
       line = fgetl(fid);
       field_names = strsplit(line);
       field_index = find(strcmp(field_names, field_requested{m}));
       values = textscan(fid, repmat('%f ', 1, length(field_names)));
       fclose(fid);

       loss.(filenames{m}) = struct('iter', values{1}, ...
           'loss', values{field_index});
    end
    losses{k} = loss;
end

% graphs
output_names_ = cellfun(@(x)strrep(x, '_', '-'), output_names, 'UniformOutput', false);

figure; title('train loss'); hold on;
for k = 1:length(output_names_), plot(losses{k}.train.iter, losses{k}.train.loss);  end
legend(output_names_);

figure; title('test - top5 accuracy'); hold on;
for k = 1:length(output_names_), plot(losses{k}.test.iter, losses{k}.test.loss);  end
legend(output_names_);