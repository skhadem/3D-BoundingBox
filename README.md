The original repo did not do things well so I am working on improving evaluation and batch data classes.
Run.py will be working evaluation script eventually.
# 3D Bounding Box Estimation Using Deep Learning and Geometry

## Introduction
This repo is PyTorch implementation for this [paper](https://arxiv.org/abs/1612.00496). In this paper, they collect
[KITTI 2D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and introduce a flow to
estimate object pose and dimension. If you are looking for TensorFlow implementation,
[here](https://github.com/smallcorgi/3D-Deepbox) is a great repo.

## Dependency
* [numpy](http://www.numpy.org/)
* [opencv](https://opencv.org/)
* [yaml](https://pypi.python.org/pypi/PyYAML)
* [PyTorch](http://pytorch.org/docs/master/)
* [torchvision](https://pypi.python.org/pypi/torchvision/0.1.9)
* [CUDA](https://developer.nvidia.com/cuda-downloads)

## Usage
Before using this code, you need download data from
[KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and unzip it.
After that, you need to add the kitti path of dataset to **config.yaml**.
```yaml
kitti_path: somewhere # Root of kitti, where contrain trainning/ and testing/   
```
Also, you can set up parameters for training and weight of loss as described in paper.
```yaml
epochs: 8 # How many epoch for training?
bins: 2  # How many bins you want to split?
w: 0.8
alpha: 0.8
batches: 8             
```
After setting up, just type it for training
```cmd
python Train.py
```
It will store model in ./models. For simple evaluation, type
```cmd
python Eval.py
```
This will calculate average orientation and dimension error (in degree and meters).

## Reference
* [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
* [PyTorch](http://pytorch.org/docs/master/)
* [KITTI](http://www.cvlibs.net/datasets/kitti)
