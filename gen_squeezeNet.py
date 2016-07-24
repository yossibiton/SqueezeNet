#!/usr/bin/env python
import sys
"""
Generate the SqueezeNet network.
If you find SqueezeNet useful in your research, please consider citing the SqueezeNet paper:

@article{SqueezeNet,
    Author = {Forrest N. Iandola and Matthew W. Moskewicz and Khalid Ashraf and Song Han and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
}
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_val_file',
                        help='Output train_val.prototxt file')
    parser.add_argument('--layer_number', nargs='*',
                        help=('Layer number for each layer stage.'),
                        default=[3, 8, 36, 3])
    parser.add_argument('-t', '--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer():
    data_layer_str = '''
name: "large_SqueezeNet"
layer {
    name: "data"
    type: "Data"
    top: "data"
    top: "label"
    include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "/ssd/dataset/ilsvrc12_train_lmdb/"
    batch_size: {{batch_size_train}}
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "/ssd/dataset/ilsvrc12_val_lmdb/"
    batch_size: {{batch_size_val}}
    backend: LMDB
  }
}
    '''
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="xavier"):
    tmp =''
    if filler == 'gaussian':
      tmp = '''      
      mean: 0.0
      std: 0.01
      '''
    conv_layer_str = '''
layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
     lr_mult: 2
     decay_mult: 0
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
    ''' %(layer_name, bottom, top, kernel_num, pad, kernel_size, stride, filler) + tmp + '''}
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}'''
    return conv_layer_str

def generate_pooling_layer(kernel_size, stride, pad, pool_type, layer_name, bottom, top):
    pool_layer_str = '''
layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str

def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
    eltwise_layer_str = '''
layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  eltwise_param {
    operation: %s
  }
}'''%(layer_name, bottom_1, bottom_2, top, op_type)
    return eltwise_layer_str

def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''
layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "accuracy_top1"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "accuracy_top1"
  include {
    # phase: TEST
  }
}
layer {
  name: "accuracy_top5"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "accuracy_top5"
  include {
    # phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}'''%(bottom, bottom, bottom)
    return softmax_loss_str

def generate_bn_layer(layer_name, bottom, top):
    bn_layer_str = '''
layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
}'''%(layer_name, bottom, top)
    return bn_layer_str

def generate_concat_layer(layer_name, bottom1, bottom2, top):
    concat_layer_str = '''
    layer{
    name: "%s"
    type: "Concat"
    bottom: "%s"
    bottom: "%s"
    top: "%s"
    }'''%(layer_name, bottom1, bottom2, top)
    return concat_layer_str

def generate_dropout_layer(layer_name, bottom):
    drop_str = '''
    layer{
      name: "%s"
      type: "Dropout"
      bottom: "%s"
      top: "%s"
      dropout_param {
        dropout_ratio: {{dropout}}
      }
    }'''%(layer_name, bottom, bottom)
    return drop_str

def generate_typeA(layer_name, bottom, top, first_size, middle_size, batchNorm):

    a_str = ""
    a_str += generate_conv_layer(1, first_size, 1, 0, '%d/squeeze1x1'%layer_name, bottom, '%d/squeeze1x1'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/squeeze1x1_bn'%layer_name, '%d/squeeze1x1'%layer_name, '%d/squeeze1x1_bn'%layer_name)
      a_str += generate_activation_layer('%d/squeeze1x1_relu'%layer_name, '%d/squeeze1x1_bn'%layer_name, '%d/squeeze1x1_bn'%layer_name)
      end_of_squeeze = '%d/squeeze1x1_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/squeeze1x1_relu'%layer_name, '%d/squeeze1x1'%layer_name, '%d/squeeze1x1'%layer_name)
      end_of_squeeze = '%d/squeeze1x1'%layer_name

    a_str += generate_conv_layer(1, middle_size, 1,  0, '%d/expand1x1'%layer_name, end_of_squeeze, '%d/expand1x1'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/expand1x1_bn'%layer_name, '%d/expand1x1'%layer_name, '%d/expand1x1_bn'%layer_name)
      a_str += generate_activation_layer('%d/expand1x1_relu'%layer_name, '%d/expand1x1_bn'%layer_name, '%d/expand1x1_bn'%layer_name)
      end_of_branch1 = '%d/expand1x1_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/expand1x1_relu'%layer_name, '%d/expand1x1'%layer_name, '%d/expand1x1'%layer_name)
      end_of_branch1 = '%d/expand1x1'%layer_name

    a_str += generate_conv_layer(3, middle_size, 1,  1, '%d/expand3x3'%layer_name, end_of_squeeze, '%d/expand3x3'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/expand3x3_bn'%layer_name, '%d/expand3x3'%layer_name, '%d/expand3x3_bn'%layer_name)
      a_str += generate_activation_layer('%d/expand3x3_relu'%layer_name, '%d/expand3x3_bn'%layer_name, '%d/expand3x3_bn'%layer_name)
      end_of_branch3 = '%d/expand3x3_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/expand3x3_relu'%layer_name, '%d/expand3x3'%layer_name, '%d/expand3x3'%layer_name)
      end_of_branch3 = '%d/expand3x3'%layer_name
    a_str += generate_concat_layer('%d/concat'%layer_name, end_of_branch3, end_of_branch1, '%d/concat'%layer_name)

    a_str += generate_conv_layer(1, 2*middle_size, 1, 0, '%d/bypassConv'%layer_name, bottom, '%d/bypassConv'%layer_name)
    a_str += generate_activation_layer('%d/bypassRelu'%layer_name, '%d/bypassConv'%layer_name, '%d/bypassConv'%layer_name)

    if batchNorm:
      a_str += generate_eltwise_layer('bypass_%d'%layer_name, '%d/concat'%layer_name, '%d/bypassConv'%layer_name, '%d/Elt'%layer_name)
      a_str += generate_bn_layer('%d/EltBN'%layer_name, '%d/Elt'%layer_name, top)
    else:
      a_str += generate_eltwise_layer('bypass_%d'%layer_name, '%d/concat'%layer_name, '%d/bypassConv'%layer_name, top)

    return a_str

def generate_typeB(layer_name, bottom, top, first_size, middle_size, batchNorm):

    a_str = ""
    a_str += generate_conv_layer(1, first_size, 1, 0, '%d/squeeze1x1'%layer_name, bottom, '%d/squeeze1x1'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/squeeze1x1_bn'%layer_name, '%d/squeeze1x1'%layer_name, '%d/squeeze1x1_bn'%layer_name)
      a_str += generate_activation_layer('%d/squeeze1x1_relu'%layer_name, '%d/squeeze1x1_bn'%layer_name, '%d/squeeze1x1_bn'%layer_name)
      end_of_squeeze = '%d/squeeze1x1_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/squeeze1x1_relu'%layer_name, '%d/squeeze1x1'%layer_name, '%d/squeeze1x1'%layer_name)
      end_of_squeeze = '%d/squeeze1x1'%layer_name

    a_str += generate_conv_layer(1, middle_size, 1,  0, '%d/expand1x1'%layer_name, end_of_squeeze, '%d/expand1x1'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/expand1x1_bn'%layer_name, '%d/expand1x1'%layer_name, '%d/expand1x1_bn'%layer_name)
      a_str += generate_activation_layer('%d/expand1x1_relu'%layer_name, '%d/expand1x1_bn'%layer_name, '%d/expand1x1_bn'%layer_name)
      end_of_branch1 = '%d/expand1x1_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/expand1x1_relu'%layer_name, '%d/expand1x1'%layer_name, '%d/expand1x1'%layer_name)
      end_of_branch1 = '%d/expand1x1'%layer_name

    a_str += generate_conv_layer(3, middle_size, 1,  1, '%d/expand3x3'%layer_name, end_of_squeeze, '%d/expand3x3'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/expand3x3_bn'%layer_name, '%d/expand3x3'%layer_name, '%d/expand3x3_bn'%layer_name)
      a_str += generate_activation_layer('%d/expand3x3_relu'%layer_name, '%d/expand3x3_bn'%layer_name, '%d/expand3x3_bn'%layer_name)
      end_of_branch3 = '%d/expand3x3_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/expand3x3_relu'%layer_name, '%d/expand3x3'%layer_name, '%d/expand3x3'%layer_name)
      end_of_branch3 = '%d/expand3x3'%layer_name
    a_str += generate_concat_layer('%d/concat'%layer_name, end_of_branch3, end_of_branch1, '%d/concat'%layer_name)


    if batchNorm:
      a_str += generate_eltwise_layer('bypass_%d'%layer_name, '%d/concat'%layer_name, bottom, '%d/Elt'%layer_name)
      a_str += generate_bn_layer('%d/EltBN'%layer_name, '%d/Elt'%layer_name, top)
    else:
      a_str += generate_eltwise_layer('bypass_%d'%layer_name, '%d/concat'%layer_name, bottom, top)

    return a_str

def generate_typeC(layer_name, bottom, top, first_size, middle_size, batchNorm):
    # nor residual
    a_str = ""
    a_str += generate_conv_layer(1, first_size, 1, 0, '%d/squeeze1x1'%layer_name, bottom, '%d/squeeze1x1'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/squeeze1x1_bn'%layer_name, '%d/squeeze1x1'%layer_name, '%d/squeeze1x1_bn'%layer_name)
      a_str += generate_activation_layer('%d/squeeze1x1_relu'%layer_name, '%d/squeeze1x1_bn'%layer_name, '%d/squeeze1x1_bn'%layer_name)
      end_of_squeeze = '%d/squeeze1x1_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/squeeze1x1_relu'%layer_name, '%d/squeeze1x1'%layer_name, '%d/squeeze1x1'%layer_name)
      end_of_squeeze = '%d/squeeze1x1'%layer_name

    a_str += generate_conv_layer(1, middle_size, 1,  0, '%d/expand1x1'%layer_name, end_of_squeeze, '%d/expand1x1'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/expand1x1_bn'%layer_name, '%d/expand1x1'%layer_name, '%d/expand1x1_bn'%layer_name)
      a_str += generate_activation_layer('%d/expand1x1_relu'%layer_name, '%d/expand1x1_bn'%layer_name, '%d/expand1x1_bn'%layer_name)
      end_of_branch1 = '%d/expand1x1_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/expand1x1_relu'%layer_name, '%d/expand1x1'%layer_name, '%d/expand1x1'%layer_name)
      end_of_branch1 = '%d/expand1x1'%layer_name

    a_str += generate_conv_layer(3, middle_size, 1,  1, '%d/expand3x3'%layer_name, end_of_squeeze, '%d/expand3x3'%layer_name)
    if batchNorm:
      a_str += generate_bn_layer('%d/expand3x3_bn'%layer_name, '%d/expand3x3'%layer_name, '%d/expand3x3_bn'%layer_name)
      a_str += generate_activation_layer('%d/expand3x3_relu'%layer_name, '%d/expand3x3_bn'%layer_name, '%d/expand3x3_bn'%layer_name)
      end_of_branch3 = '%d/expand3x3_bn'%layer_name
    else:
      a_str += generate_activation_layer('%d/expand3x3_relu'%layer_name, '%d/expand3x3'%layer_name, '%d/expand3x3'%layer_name)
      end_of_branch3 = '%d/expand3x3'%layer_name
    a_str += generate_concat_layer('%d/concat'%layer_name, end_of_branch3, end_of_branch1, top)

    return a_str

def generate_fully_train_val_residual(BatchNorm, a, ratio):
    
    network_str = generate_data_layer()

    network_str += generate_conv_layer(3, 64, 2, 1, 'conv1', 'data', 'conv1')
    if (BatchNorm):
      network_str += generate_bn_layer('conv1_bn', 'conv1', 'conv1_bn')
      network_str += generate_activation_layer('conv1_relu', 'conv1_bn', 'conv1_bn', 'ReLU')
      network_str += generate_pooling_layer(3, 2, 1, 'MAX', 'pool1', 'conv1_bn', 'pool1')
    else:
      network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
      network_str += generate_pooling_layer(3, 2, 1, 'MAX', 'pool1', 'conv1', 'pool1')

    network_str += generate_typeA(2, 'pool1', '2/end', a[0]/ratio, a[0], BatchNorm)
    network_str += generate_typeB(3, '2/end', '3/end', a[0]/ratio, a[0], BatchNorm)
    network_str += generate_typeA(4, '3/end', '4/end', a[1]/ratio, a[1], BatchNorm)
    network_str += generate_pooling_layer(3,2,1,'MAX', 'pool4', '4/end', 'pool4')
    network_str += generate_typeB(5, 'pool4', '5/end', a[1]/ratio, a[1], BatchNorm)
    network_str += generate_typeA(6, '5/end', '6/end', a[2]/ratio, a[2], BatchNorm)
    network_str += generate_typeB(7, '6/end', '7/end', a[2]/ratio, a[2], BatchNorm)
    network_str += generate_typeA(8, '7/end', '8/end', a[3]/ratio, a[3], BatchNorm) 
    network_str += generate_pooling_layer(3,2,1,'MAX', 'pool8', '8/end', 'pool8')
    network_str += generate_typeB(9, 'pool8', '9/end', a[3]/ratio, a[3], BatchNorm)

    if BatchNorm=='0' or BatchNorm==0:
      network_str += generate_dropout_layer('drop9', '9/end')
    network_str += generate_conv_layer(1, 1000, 1, 0, 'conv_final', '9/end', 'conv_final', filler = 'gaussian')
    
    if BatchNorm:
      network_str += generate_bn_layer('conv_final_bn', 'conv_final', 'conv_final_bn')
      network_str += generate_activation_layer('relu10', 'conv_final_bn', 'conv_final_bn')
      extra='_bn'
    else:
      network_str += generate_activation_layer('relu10', 'conv_final', 'conv_final')
      extra=''    


    network_str += '''
layer {
      name: "pool10"
      type: "Pooling"
      bottom: "conv_final%s"
      top: "pool10"
      pooling_param {
        pool: AVE
        global_pooling: true
      }
    }
    ''' %extra
    network_str += generate_softmax_loss('pool10')

    return network_str

def generate_fully_train_val(BatchNorm, a, ratio):
    # generates SqueezeNet1.1 model (original params : a=[64, 128, 192, 256], ratio = 4.0)
    network_str = generate_data_layer()

    network_str += generate_conv_layer(3, 64, 2, 1, 'conv1', 'data', 'conv1')
    if (BatchNorm):
      network_str += generate_bn_layer('conv1_bn', 'conv1', 'conv1_bn')
      network_str += generate_activation_layer('conv1_relu', 'conv1_bn', 'conv1_bn', 'ReLU')
      network_str += generate_pooling_layer(3, 2, 1, 'MAX', 'pool1', 'conv1_bn', 'pool1')
    else:
      network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
      network_str += generate_pooling_layer(3, 2, 1, 'MAX', 'pool1', 'conv1', 'pool1')

    network_str += generate_typeC(2, 'pool1', '2/end', a[0]/ratio, a[0], BatchNorm)
    network_str += generate_typeC(3, '2/end', '3/end', a[0]/ratio, a[0], BatchNorm)
    network_str += generate_pooling_layer(3,2,1,'MAX', 'pool3', '3/end', 'pool3')
    
    network_str += generate_typeC(4, 'pool3', '4/end', a[1]/ratio, a[1], BatchNorm)
    network_str += generate_typeC(5, '4/end', '5/end', a[1]/ratio, a[1], BatchNorm)
    network_str += generate_pooling_layer(3,2,1,'MAX', 'pool5', '5/end', 'pool5')
    
    network_str += generate_typeC(6, 'pool5', '6/end', a[2]/ratio, a[2], BatchNorm)
    network_str += generate_typeC(7, '6/end', '7/end', a[2]/ratio, a[2], BatchNorm)
    network_str += generate_typeC(8, '7/end', '8/end', a[3]/ratio, a[3], BatchNorm) 
    network_str += generate_typeC(9, '8/end', '9/end', a[3]/ratio, a[3], BatchNorm)

    if BatchNorm=='0' or BatchNorm==0:
      network_str += generate_dropout_layer('drop9', '9/end')
    network_str += generate_conv_layer(1, 1000, 1, 0, 'conv_final', '9/end', 'conv_final', filler = 'gaussian')
    
    if BatchNorm:
      network_str += generate_bn_layer('conv_final_bn', 'conv_final', 'conv_final_bn')
      network_str += generate_activation_layer('relu10', 'conv_final_bn', 'conv_final_bn')
      extra='_bn'
    else:
      network_str += generate_activation_layer('relu10', 'conv_final', 'conv_final')
      extra=''    


    network_str += '''
layer {
      name: "pool10"
      type: "Pooling"
      bottom: "conv_final%s"
      top: "pool10"
      pooling_param {
        pool: AVE
        global_pooling: true
      }
    }
    ''' %extra
    network_str += generate_softmax_loss('pool10')

    return network_str

def main():
    # a = E_3x3 (=E_1x1)
    a=[64, 128, 192, 256]
    
    #base=64,inc=64
    #a=[32,64,96,128]
    
    # a*ratio = S_1x1
    # ratio = SR / 2
    if sys.argv[2] == 'large':
      ratio = 1.0 
    if sys.argv[2] == 'small':
      ratio = 4.0
      
    #
    network_str = generate_fully_train_val(int(sys.argv[1]), a, ratio)
    print "BatchNorm =", sys.argv[1], sys.argv[2]

    fp = open('trainval.template', 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
