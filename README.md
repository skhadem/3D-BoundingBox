# 3D Bounding Box Estimation Using Deep Learning and Geometry

## Introduction
This repo is pytorch implementation for this [paper](https://arxiv.org/abs/1612.00496). In this paper, they collect 
[KITTI 2D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and introduce a flow to
estimate object pose and dimension.

## Usage
Before using this code, you need download data from 
[KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and unzip it. 
After that, you need to add the kitti path of dataset to *config.yaml*.
```yaml
kitti_path: somewhere # Root of kitti, where contrain trainning/ and testing/   
```
Also, you can set up parameters for training and weight of loss as describded in paper.
```yaml
epochs: 8 # How many epoch for training?
bins: 2  # How many bins you want to split?
w: 0.8
alpha: 0.8
batches: 8             
```
After setting up, just type
```cmd
python Train.py
```
It will store model in ./models
