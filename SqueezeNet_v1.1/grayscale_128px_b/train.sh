OUTPUT=../../output/SqueezeNet_v1.1/grayscale_128px_b/log
~/Dev/faster_rcnn/external/caffe/build/tools/caffe train -gpu 0 -solver solver.prototxt 2>&1 | tee $OUTPUT