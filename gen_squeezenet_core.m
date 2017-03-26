function body = gen_squeezenet_core(S_1x1, E_1x1, E_3x3, pool_layers, add_mult, add_batch_norm)

if ~exist('inspect', 'var')
    inspect = false;
end

conv1 = [3 64];
body = {};

block_idx = 1;
% conv1 --> pool1
body{end+1} = caffe_layer_conv_def(block_idx, 'data', conv1(1), conv1(2), add_mult, 'stride', 2, 'pad', 1);
blob_name = body{end}.top;
if add_batch_norm
    % disable bias in previous conv layer
    body{end}.convolution_param.bias_term = 'false';
    body{end}.convolution_param = ...
        rmfield(body{end}.convolution_param, 'bias_filler');
    
    body{end+1} = caffe_layer_batchnorm_def(block_idx, body{end}.top);
    blob_name = body{end}{end}.top;
end
body{end+1} = caffe_layer_relu_def(block_idx, blob_name);
body{end+1} = caffe_layer_pool_def(block_idx, blob_name, 3, 'stride', 2, 'pad', 1);
block_idx = block_idx + 1;

% Fire modules
fire_layer_num = block_idx - 1;
last_layer_blob = body{end}.top;
while (fire_layer_num <= length(S_1x1))
    body{end+1} = caffe_layer_fire_def(block_idx, last_layer_blob, ...
        S_1x1(fire_layer_num), E_1x1(fire_layer_num), E_3x3(fire_layer_num), add_mult, add_batch_norm);
    last_layer_blob = body{end}{end}.top;
    if ismember(block_idx, pool_layers)
        body{end+1} = caffe_layer_pool_def(block_idx, ...
            last_layer_blob, 3, 'stride', 2);
        last_layer_blob = body{end}.top;
    end
    block_idx = block_idx + 1;
    fire_layer_num = block_idx - 1;
end