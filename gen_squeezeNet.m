clearvars;
addpath('../matcaffe_tools');
imagenet_data_dir = '/home/gpu-admin/data/ImageNet_preprocessed';

%% constants (we use grayscale & resized images with maximal height/width of 128)
output_dir_model = fullfile('SqueezeNet_v1.1', 'grayscale_128px');
add_batch_norm = true;
add_mult = false;
batch_size = 512;
batch_size_test = 512;
if add_batch_norm
    output_dir_model = [output_dir_model '_BN']; 
    resize_batch_factor = 0.25;
    resize_batch_factor_test = 0.25;
    batch_size = batch_size*resize_batch_factor;
    batch_size_test = batch_size_test*resize_batch_factor_test;
end

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

output_dir_train = fullfile('output', output_dir_model); 
caffe_dir = '../caffe_Austriker/build/tools';

mkdir(output_dir_train);
mkdir(output_dir_model);

% make paths relative to output_dir_model (where the training process will run from)
output_dir_model_depth = 1 + length(strfind(output_dir_model, '/'));
output_dir_train = fullfile(repmat('../', 1, output_dir_model_depth), output_dir_train); 
caffe_dir        = fullfile(repmat('../', 1, output_dir_model_depth), caffe_dir);
%% create model SqueezeNet_v1.1 for ImageNet classification task
net_descr = struct('head', struct('name', net_name), 'body', []);
net_descr.head_deploy = net_descr.head;
net_descr.body = {};

% data_layers (ImageNet train & test set)
net_descr.head.layer{1} = caffe_layer_data_def('lmdb', 1); %Train
net_descr.head.layer{1}.data_param.source = fullfile(imagenet_data_dir, 'ilsvrc12_train_lmdb');
net_descr.head.layer{1}.data_param.batch_size = batch_size;
net_descr.head.layer{1}.transform_param.crop_size = crop_size;
net_descr.head.layer{1}.transform_param.mean_value = 117;
    
net_descr.head.layer{2} = caffe_layer_data_def('lmdb', 0); %Validation
net_descr.head.layer{2}.data_param.source = fullfile(imagenet_data_dir, 'ilsvrc12_val_lmdb');
net_descr.head.layer{2}.data_param.batch_size = batch_size_test;
net_descr.head.layer{2}.transform_param.crop_size = crop_size;
net_descr.head.layer{2}.transform_param.mean_value = 117;

% --- Data deploy
net_descr.head_deploy.input = 'data';
net_descr.head_deploy.input_shape.dim = {1, 1, crop_size, crop_size};

% --- CNN body (fire modules)
net_descr.body = gen_squeezenet_core(S_1x1, E_1x1, E_3x3, pool_layers, add_mult, add_batch_norm);

% --- CNN classification branch (conv --> average pooling)
last_layer_blob = net_descr.body{end}{end}.top;
if ~add_batch_norm
    net_descr.body{end+1} = caffe_layer_drop_def('drop9', last_layer_blob);
%if add_batch_norm
    % use weaker regularization in dropout
%    net_descr.body{end}.dropout_param.dropout_ratio = 0.2;
end
net_descr.body{end+1} = caffe_layer_conv_def('conv10', last_layer_blob, 1, 1000, false);
net_descr.body{end}.convolution_param.weight_filler = ...
    struct('type', 'gaussian', 'mean', 0.0, 'std', 0.01);
blob_name = net_descr.body{end}.top;
if add_batch_norm
    bn_bottom = blob_name;
    blob_name = [bn_bottom '_bn'];
    net_descr.body{end+1} = caffe_layer_batchnorm_def(blob_name, bn_bottom);
end
net_descr.body{end+1} = caffe_layer_relu_def('relu_conv10', blob_name);
net_descr.body{end+1} = caffe_layer_pool_def('pool10', blob_name, 0);
net_descr.body{end}.pooling_param = struct('pool', 'AVE', 'global_pooling', 'true');

%% --- Footer (losses and accuracies)
% --- Train/Val
last_blob_name = net_descr.body{end}.top;
net_descr.loss{1} = caffe_layer_loss_def('loss', {last_blob_name, 'label'});
net_descr.loss{2} = caffe_layer_accuracy_def('accuracy', {last_blob_name, 'label'});
net_descr.loss{3} = caffe_layer_accuracy_def('accuracy_top5', {last_blob_name, 'label'});
net_descr.loss{3}.accuracy_param = struct('top_k', 5);
% --- Deploy
net_descr.loss_deploy{1} = caffe_layer_loss_def('prob', {last_blob_name});
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
if add_batch_norm
    % batch size is smaller so we need to adjust some values
    solver_props.test_iter = num2str(round(str2double(solver_props.test_iter) / resize_batch_factor_test));
    solver_props.iter_size = num2str(1/resize_batch_factor);
    
    base_lr = str2double(solver_props.base_lr);
    num_iters = str2double(solver_props.max_iter);
    
    base_lr_new = 0.05;
    num_iters_new = 100e3;
    figure; title('learning rate decay'); hold on;
    plot(base_lr*(1-(0:num_iters)/num_iters));
    plot(base_lr_new*(1-(0:num_iters_new)/num_iters_new).^2);
    plot(base_lr*0.999985.^(0:num_iters));
    legend({'poly1', 'poly1.5', 'exp'});
    % recommendations from original BatchNorm paper
    % 1. Increase leraning rate (0.04 --> 0.05)
    solver_props.base_lr = '0.05'; 
    % 2. Remove Dropout (done earlier)
    % 3. Reduce the L2 weight regularization (x5)
    solver_props.weight_decay = '0.00004';
    % 5. Accelerate the learning rate decay
    %    we will use 100k iters instead of 170k, so the decay is about x2 faster
    %    we will also use polynomial decat with power 2.0
    solver_props.max_iter = '100000';
    solver_props.power = '2.0'; 
end
caffe_write_solverprototxt(fullfile(output_dir_model, 'solver.prototxt'), solver_props);

%% Generating train.sh script
fid = fopen(fullfile(output_dir_model, 'train.sh'), 'w');
fprintf(fid, 'OUTPUT=%s\n', fullfile(output_dir_train, 'log'));
fprintf(fid, '%s/caffe train -gpu 0 -solver solver.prototxt 2>&1 | tee $OUTPUT', caffe_dir);
fclose(fid);

fprintf('\n\nEverything is ready, just go to :\n%s\nand run train.sh.\nEnjoy!\n', output_dir_model);