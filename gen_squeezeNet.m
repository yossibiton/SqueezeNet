clearvars;
addpath('../matcaffe_tools');
imagenet_data_dir = '/home/gpu-admin/data/ImageNet_preprocessed';

%% constants (we use grayscale & resized images with maximal height/width of 128)
output_dir_model = fullfile('SqueezeNet_v1.1', 'grayscale_128px');
add_batch_norm = true;
add_mult = false;
if add_batch_norm, output_dir_model = [output_dir_model '_BN']; end

crop_size = 128;

% layer parameters
pool_layers = [1 3 5];

% fire modules parameters
E_i = [128 128 192 192 256 256 512 512];
SR = 0.125;
pct_3x3 = 0.5;
E_1x1 = (1 - pct_3x3) * E_i;
E_3x3 = pct_3x3 * E_i;
S_1x1 = SR * E_i;

%% some preparations
net_name = strrep(output_dir_model, '/', '_');
output_dir_model_depth = 1 + length(strfind(output_dir_model, '/'));

% paths relative to output_dir_model (where the training process will run from)
output_dir_train = fullfile(repmat('../', 1, output_dir_model_depth), 'output', output_dir_model); 
caffe_dir = [repmat('../', 1, 1 + output_dir_model_depth) 'caffe_Austriker/build/tools'];

mkdir(output_dir_train);
mkdir(output_dir_model);

%% create model SqueezeNet_v1.1 for ImageNet classification task
net_descr = struct('head', struct('name', net_name), 'body', []);
net_descr.head_deploy = net_descr.head;
net_descr.body = {};

% data_layers (ImageNet train & test set)
net_descr.head.layer{1} = caffe_layer_data_def('lmdb', 1); %Train
net_descr.head.layer{1}.data_param.source = fullfile(imagenet_data_dir, 'ilsvrc12_train_lmdb');
net_descr.head.layer{1}.data_param.batch_size = 512;
net_descr.head.layer{1}.transform_param.crop_size = crop_size;
net_descr.head.layer{1}.transform_param.mean_value = 117;
    
net_descr.head.layer{2} = caffe_layer_data_def('lmdb', 0); %Validation
net_descr.head.layer{2}.data_param.source = fullfile(imagenet_data_dir, 'ilsvrc12_val_lmdb');
net_descr.head.layer{2}.data_param.batch_size = 512;
net_descr.head.layer{2}.transform_param.crop_size = crop_size;
net_descr.head.layer{2}.transform_param.mean_value = 117;

% --- Data deploy
net_descr.head_deploy.input = 'data';
net_descr.head_deploy.input_shape.dim = {1, 1, crop_size, crop_size};

% --- CNN body (fire modules)
net_descr.body = gen_squeezenet_core(S_1x1, E_1x1, E_3x3, pool_layers, add_mult, add_batch_norm);

% --- CNN classification branch (conv --> average pooling)
last_layer_blob = net_descr.body{end}{end}.top;
net_descr.body{end+1} = caffe_layer_drop_def('drop9', last_layer_blob);
if add_batch_norm
    % use weaker regularization in dropout
    net_descr.body{end}.dropout_param.dropout_ratio = 0.2;
end
net_descr.body{end+1} = caffe_layer_conv_def('conv10', 'fire9/concat', 1, 1000, false);
net_descr.body{end}.convolution_param.weight_filler = ...
    struct('type', 'gaussian', 'mean', 0.0, 'std', 0.01);
blob_name = net_descr.body{end}.top;
if add_batch_norm
    bn_bottom = blob_name;
    blob_name = [bn_bottom '_bn'];
    net_descr.body{end+1} = caffe_layer_batchnorm_def(blob_name, bn_bottom);
end
net_descr.body{end+1} = caffe_layer_relu_def('relu_conv10', blob_name);
net_descr.body{end+1} = caffe_layer_pool_def('pool10', 'conv10', 0);
net_descr.body{end}.pooling_param = struct('pool', 'AVE', 'global_pooling', 'true');

%% --- Footer (losses and accuracies)
% --- Train/Val
net_descr.loss{1} = caffe_layer_loss_def('loss', {'pool10', 'label'});
net_descr.loss{2} = caffe_layer_accuracy_def('accuracy', 'pool10');
net_descr.loss{2} = rmfield(net_descr.loss{2}, 'include');
net_descr.loss{3} = caffe_layer_accuracy_def('accuracy_top5', 'pool10');
net_descr.loss{3} = rmfield(net_descr.loss{3}, 'include');
net_descr.loss{3}.accuracy_param = struct('top_k', 5);
% --- Deploy
net_descr.loss_deploy{1} = caffe_layer_loss_def('prob', {'pool10'});
net_descr.loss_deploy{1}.type = 'Softmax';

%% Saving the whole thing
% Generating training/validation net model
caffe_save_net(fullfile(output_dir_model, 'train_val.prototxt'), ...
    net_descr.head, [net_descr.body(:); net_descr.loss(:)] );
% Generating deployment model
caffe_save_net(fullfile(output_dir_model, 'deploy.prototxt'), ...
    net_descr.head_deploy, [net_descr.body(:); net_descr.loss_deploy(:)] );

%% Generating solver file
solver_props = caffe_read_solverprototxt('solver_template.prototxt');
solver_props.snapshot_prefix = fullfile(output_dir_train, 'train');
caffe_write_solverprototxt(fullfile(output_dir_model, 'solver.prototxt'), solver_props);

%% Generating train.sh script
fid = fopen(fullfile(output_dir_model, 'train.sh'), 'w');
fprintf(fid, 'OUTPUT=%s\n', fullfile(output_dir_train, 'log'));
fprintf(fid, '%s/caffe train -gpu 0 -solver solver.prototxt 2>&1 | tee $OUTPUT', caffe_dir);
fclose(fid);

fprintf('\n\nEverything is ready, just go to :\n%s\nand run train.sh.\nEnjoy!\n', output_dir_model);