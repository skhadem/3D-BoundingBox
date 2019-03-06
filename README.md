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
This will download the weights I have trained and also the YOLOv3 weights from the
official yolo [site](https://pjreddie.com/darknet/yolo/).

To run in evaluation:
```
python Run.py
```
>Note: This script expects images in `./Kitti/testing/image_2/` and corresponding projection matricies
in `./Kitti/testing/calib/`. See [training](#training) for where to download data from.

Press SPACE to process next image, and any other key to exit.
This will visualize the 3d bounding box for all the images in Kitti/testing/. The image is passed
through YOLOv3 pre-trained on the COCO dataset to make 2D bounding box detections, which is then
used to crop the image and pass portions through the PyTorch neural net. The obtained location
is used to project a 3D box onto the image.
Camera projection matrices are also needed for every image (given by Kitti).

```
python Run_no_yolo.py
```
Will process all the images in eval/, using the label to get 2D box instead of YOLO.

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
The PyTorch neural net takes in images of size 224x224 and predicts the orientation and
relative dimension of that object to the class average. Thus, another neural net must give
the 2D bounding box and object class, this repo uses YOLOv3 to do this using OpenCV.
Using the orientation, dimension, and 2D bounding box, the 3D location is calculated.
(more in depth math explanation coming soon...)

There are 2 key assumptions made:
1. The 2D bounding box fits very tightly around the object
2. The object has ~0 pitch and ~0 roll (valid for cars on the road)


## Future Goals
- Train custom YOLO net on the Kitti dataset
- Cuda optimization to run frame by frame on video feed
- ROS node to publish positions

## Credit
I originally started from a fork of this [repo](https://github.com/fuenwang/3D-BoundingBox), and some of the original code still exists in the training script.
