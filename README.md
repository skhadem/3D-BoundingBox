# 3D Bounding Box Estimation Using Deep Learning and Geometry
If interested, join the slack workspace where the paper is discussed, issues are worked through, and more! Click this [link](https://join.slack.com/t/3dboundingbox-oun9186/shared_invite/enQtNDk4Njg2NzYyNzY5LWVlZWRlMjNhZmZlYjVmNGY3NWVlNDA4MmY2ZWQ3ZmUyY2Q4OWIxMmY4NzU4YmViM2ViZWI5YjgxOTIyOTI4ZjI) to join.

## Introduction
PyTorch implementation for this [paper](https://arxiv.org/abs/1612.00496).

![example-image](http://soroushkhadem.com/img/2d-top-3d-bottom1.png)

At the moment, it takes approximately 0.4s per frame, depending on the number of objects
detected. An improvement will be speed upgrades soon. Here is the current fastest possible:
![example-video](eval/example/3d-bbox-vid.gif)

## Requirements
- PyTorch
- Cuda
- OpenCV >= 3.4.3

## Usage
In order to download the weights:
```
cd weights/
./get_weights.sh
```
This will download pre-trained weights for the 3D BoundingBox net and also YOLOv3 weights from the
official yolo [source](https://pjreddie.com/darknet/yolo/).

>If script is not working: [pre trained weights](https://drive.google.com/open?id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA) and 
[YOLO weights](https://pjreddie.com/media/files/yolov3.weights)

To see all the options:
```
python Run.py --help
```

Run through all images in default directory (eval/image_2/), optionally with the 2D
bounding boxes also drawn. Press SPACE to proceed to next image, and any other key to exit.
```
python Run.py [--show-yolo]
```
>Note: See [training](#training) for where to download the data from

There is also a script provided to download the default video from Kitti in ./eval/video. Or,
download any Kitti video and corresponding calibration and use `--image-dir` and `--cal-dir` to
specify where to get the frames from.
```
python Run.py --video [--hide-debug]
```

## Training
First, the data must be downloaded from [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
Download the left color images, the training labels, and the camera calibration matrices. Total is ~13GB.
Unzip the downloads into the Kitti/ directory.

```
python Train.py
```
By default, the model is saved every 10 epochs in weights/.
The loss is printed every 10 batches. The loss should not converge to 0! The loss function for
the orientation is driven to -1, so a negative loss is expected. The hyper-parameters to tune
are alpha and w (see paper). I obtained good results after just 10 epochs, but the training
script will run until 100.

## How it works
The PyTorch neural net takes in images of size 224x224 and predicts the orientation and
relative dimension of that object to the class average. Thus, another neural net must give
the 2D bounding box and object class. I chose to use YOLOv3 through OpenCV.
Using the orientation, dimension, and 2D bounding box, the 3D location is calculated, and then
back projected onto the image.

There are 2 key assumptions made:
1. The 2D bounding box fits very tightly around the object
2. The object has ~0 pitch and ~0 roll (valid for cars on the road)

## Future Goals
- Train custom YOLO net on the Kitti dataset
- Some type of Pose visualization (ROS?)

## Credit
I originally started from a fork of this [repo](https://github.com/fuenwang/3D-BoundingBox), and some of the original code still exists in the training script.
