function body = gen_squeezenet_core(S_1x1, E_1x1, E_3x3, pool_layers, add_mult, add_batch_norm)

conv1 = [3 64];
body = {};

% conv1 --> pool1
blob_name = 'conv1';
body{end+1} = caffe_layer_conv_def(blob_name, 'data', conv1(1), conv1(2), add_mult, 'stride', 2);
if add_batch_norm
    % disable bias in previous conv layer
    % TODO : we might take care of bias_filler and mult
    body{end}.convolution_param.bias_term = 'false';
    
    bn_bottom = blob_name;
    blob_name = [blob_name '_bn'];
    body{end+1} = caffe_layer_batchnorm_def(blob_name, bn_bottom);
end
body{end+1} = caffe_layer_relu_def('relu_conv1', blob_name);
body{end+1} = caffe_layer_pool_def('pool1', blob_name, 3, 'stride', 2);

% Fire modules
fire_layer_num = 1;
last_layer_blob = 'pool1';
while (fire_layer_num <= length(S_1x1))
    layer_index = fire_layer_num + 1;
    body{end+1} = caffe_layer_fire_def(sprintf('fire%d', layer_index), last_layer_blob, ...
        S_1x1(fire_layer_num), E_1x1(fire_layer_num), E_3x3(fire_layer_num), add_mult, add_batch_norm);
    last_layer_blob = body{end}{end}.top;
    if ismember(layer_index, pool_layers)
        pool_name = sprintf('pool%d', layer_index);
        body{end+1} = caffe_layer_pool_def(pool_name, ...
            last_layer_blob, 3, 'stride', 2);
        last_layer_blob = pool_name;
    end
    fire_layer_num = fire_layer_num + 1;
end