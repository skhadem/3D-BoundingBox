# 3D Bounding Box Estimation Using Deep Learning and Geometry
If interested, join the slack workspace where the paper is discussed, issues are worked through, and more! Click this [link](https://join.slack.com/t/3dboundingbox-oun9186/shared_invite/enQtNDk4Njg2NzYyNzY5LWVlZWRlMjNhZmZlYjVmNGY3NWVlNDA4MmY2ZWQ3ZmUyY2Q4OWIxMmY4NzU4YmViM2ViZWI5YjgxOTIyOTI4ZjI) to join.

## Introduction
PyTorch implementation for this [paper](https://arxiv.org/abs/1612.00496).

![example](http://soroushkhadem.com/img/2d-top-3d-bottom1.png)

## Usage
Download the weights:
```
cd weights/
./get_weights.sh
```
To run in evaluation:
```
python Run.py
```
This will visualize the 3d box for all images in eval/. For these, the label is only used
to obtain the object class and the 2D bounding box. This will soon come from YOLO once the integration is done!
Camera projection matrices are also needed for every image (given by Kitti).

## Training
First, you must download the data from [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
You will need the left color images, the training labels, and the camera calibration matrices. Total is ~13GB.
Put these folders into the Kitti/ directory.
```
python Train.py
```
By default, the model is saved every 10 epochs in weights/.
The loss is printed every 10 batches. The loss should not converge to 0! The loss function for
the orientation is driven to -1, so a negative loss is expected. The hyper-parameters to tune
are alpha and w (see paper). I obtained good results after just 10 epochs, but the training
script will run until 100.

## How it works

## Future Goals
- YOLO integration to get 2D boxes
- Cuda optimization to run frame by frame on video feed
- ROS node to publish positions

## Credit
I originally started from a fork of this [repo](https://github.com/fuenwang/3D-BoundingBox), and some of the original code still exists in the training script.
