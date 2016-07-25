OUTPUT=../../output/SqueezeNet_v1.1/grayscale_128px/log
~/Dev/faster_rcnn/external/caffe/build/tools/caffe train -gpu 0 -solver solver.prototxt 2>&1 | tee $OUTPUT