"""
Big Picture:

- use a 2D box of an object in scene (can get it from label or yolo eventually)
- pass image cropped to object through the model
- net outputs dimension and oreintation, then calculate the location (T) using camera
    cal and lots of math
- put the calculated 3d location onto 2d image using plot_regressed_3d_bbox
- visualize

Plan:

[x] reformat data structure to understand it better
[x] use purely truth values from label for dimension and orient to test math
[ ] use the label 2d_box to get dimension and orient from net
[ ] use yolo or rcnn to get the 2d box and class, so run from just an image (and cal)
[ ] Try and optimize to be able to run on video
[ ] Ros node eventually

Random TODOs:

[ ] loops inside of plotting functions
[ ] Move alot of functions to a library and import it


Notes:

- The net outputs an angle (actually a sin and cos) relative to an angle defined
    by the # of bins, thus the # of bins used to train model should be known
- Everything should be using radians, just for consistancy (old version used degrees, so careful!)
- Is class ever used? Could this be an improvement?

"""

# some global debug options
single_car = True
debug_corners = True


import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/Library')
import cv2
import yaml
import time
import datetime
from enum import Enum
import numpy as np
import itertools
import random

import Model
import Dataset
from library.Plotting import *
from library.File import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

# plot from net output. The orient should be global
# after done testing math, can remove label param
def plot_regressed_3d_bbox(img, net_output, calib_file, label, truth_img):
    cam_to_img = get_calibration_cam_to_image(calib_file)
    K = get_K(os.path.abspath(os.path.dirname(__file__)) + '/eval/calib/calib_cam_to_cam.txt')
    box_2d = net_output['Box_2D']

    dims = net_output['Dimension']
    orient = net_output['Orientation']

    # use truth for now
    truth_dims = label['Dimension']
    truth_orient = label['Ry']

    # the math! returns X, the corners used for constraint
    center, X = calc_location(truth_dims, cam_to_img, box_2d, label['Alpha'], net_output['ThetaRay'])

    center = [center[0][0], center[1][0], center[2][0]]

    truth_pose = label['Location']

    print "Estimated pose:"
    print center
    print "Truth pose:"
    print truth_pose
    print "-------------"

    # plot_2d_box(img, box_2d)
    plot_2d_box(truth_img, box_2d)

    plot_3d_box(img, cam_to_img, truth_orient, truth_dims, center) # 3d boxes



    # plot the corners that were used
    # these corners returned are the ones that are unrotated, because they were
    # in the calculation. We must find the indicies of the corners used, then generate
    # the roated corners and visualize those

    if debug_corners:

        left = X[0]
        right = X[1]
        # DEBUG with left and right as different colors
        corners = create_corners(truth_dims) # unrotated

        left_corner_indexes = [corners.index(i) for i in left] # get indexes
        right_corner_indexes = [corners.index(i) for i in right] # get indexes

        # get the rotated version
        R = rotation_matrix(truth_orient)
        # corners = create_corners(truth_dims, location=center, R=R)
        # corners_used = [corners[i] for i in corner_indexes]
        #
        # # plot
        # plot_3d_pts(img, corners_used, truth_pose, cam_to_img=cam_to_img, relative=False)

        corners = create_corners(truth_dims, location=truth_pose, R=R)
        left_corners_used = [corners[i] for i in left_corner_indexes]
        right_corners_used = [corners[i] for i in right_corner_indexes]

        # plot
        for i, pt in enumerate(left_corners_used):
            plot_3d_pts(truth_img, [pt], truth_pose, cam_to_img=cam_to_img, relative=False, constraint_idx=0)

        for i, pt in enumerate(right_corners_used):
            plot_3d_pts(truth_img, [pt], truth_pose, cam_to_img=cam_to_img, relative=False, constraint_idx=2)

        plot_3d_box(truth_img, cam_to_img, truth_orient, truth_dims, truth_pose) # 3d boxes


        # the 4 corners used
        # corners = create_corners(truth_dims) # unrotated
        #
        # corner_indexes = [corners.index(i) for i in X] # get indexes
        #
        # # get the rotated version
        # R = rotation_matrix(truth_orient)
        #
        # corners = create_corners(truth_dims, location=truth_pose, R=R)
        # corners_used = [corners[i] for i in corner_indexes]
        #
        # # plot
        # for i, pt in enumerate(corners_used):
        #     plot_3d_pts(truth_img, [pt], truth_pose, cam_to_img=cam_to_img, relative=False, constraint_idx=i)

    return img

# From KITTI : x = P2 * R0_rect * Tr_velo_to_cam * y
# Velodyne coords
def plot_truth_3d_bbox(img, label_info, calib_file):
    Ry = label_info['Ry']
    dims = label_info['Dimension']
    center = label_info['Location']

    plot_3d(img, calib_file, Ry, dims, center)

    return img

def draw_truth_boxes(img, info, calib_file):

    for item in info:
        if item['Class'] == 'DontCare':
            continue
        img = plot_truth_3d_bbox(img, item, calib_file)

    return img

# using config # of bins, create array of 0..2pi split into bins
# radians
def generate_angle_bins(bins):
    interval = 2 * np.pi / bins

    angle_bins = np.zeros(bins)

    for i in range(1, bins):
        angle_bins[i] = i * interval

    return angle_bins


#https://math.stackexchange.com/questions/1320285/convert-a-pixel-displacement-to-angular-rotation
# helpful:
#https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
def calc_theta_ray(img, box_2d):
    K = get_K(os.path.abspath(os.path.dirname(__file__)) + '/eval/calib/calib_cam_to_cam.txt')
    width = img.shape[1]
    fovx = 2 * np.arctan(width / (2 * K[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    dx = center - (width / 2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
    angle = angle * mult

    return angle



def format_net_output(box_2d, orient, dim, theta_ray):
    net_output = {}
    net_output['Box_2D'] = box_2d
    net_output['Orientation'] = orient
    net_output['Dimension'] = dim
    net_output['ThetaRay'] = theta_ray

    return net_output


def main():

    # option to do a single image if passed as arg, or do all in ./eval
    try:
        single_img_id = sys.argv[1]
    except:
        single_img_id = False

    # get models
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        print 'No folder named \"models/\"'
        exit()

    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    # get net's config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    path = config['kitti_path']
    epochs = config['epochs']
    batches = config['batches']
    bins = config['bins'] # important
    alpha = config['alpha']
    w = config['w']

    # load model
    if len(model_lst) == 0:
        print 'No previous model found, please check it'
        exit()
    else:
        print 'Find previous model %s'%model_lst[-1]
        my_vgg = vgg.vgg19_bn(pretrained=False)
        model = Model.Model(features=my_vgg.features, bins=bins).cuda()
        params = torch.load(store_path + '/%s'%model_lst[-1])
        model.load_state_dict(params)
        model.eval()


    img_data = Dataset.MyImageDataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    data = Dataset.MyBatchDataset(img_data, batches, bins, mode = 'eval')

    angle_bins = generate_angle_bins(bins)

    # main loop through the images

    for img_idx in range(0,data.num_images):

        img_ID = img_data[img_idx]['ID']

        # single mode, image name is passed in
        if single_img_id:
            if img_ID != single_img_id:
                continue

        calib_file = os.path.abspath(os.path.dirname(__file__)) + '/eval/calib/%s.txt' % img_ID

        truth_img = img_data.GetRawImage(img_idx)
        info = img_data[img_idx]['Label']
        # draw_truth_boxes(truth_img, info, calib_file)

        img = img_data.GetRawImage(img_idx)

        # batches is the all objects in image, cropped, i.e. crop with just a car
        # loop through each object in image
        batches, infos = data.formatForModel(img_idx)
        for i, batch in enumerate(batches):
            # get 2d box and class from label, should be all we need in eval
            info = infos[i] # this should be better for eval
            obj_class = info['Class']
            box_2d = info['Box_2D']

            if obj_class == 'DontCare':
                continue

            # create tensor
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

            # run through the net, format output
            [orient, conf, dim] = model(batch)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            # How this works:
            # the net outputs the angle relative to a bin of angles, as defined
            # by the number of bins in the config file, so take the bin
            # with the highest confidence and use that as the angle
            argmax = np.argmax(conf)
            orient_max = orient[argmax, :]
            cos = orient_max[0]
            sin = orient_max[1]
            alpha = np.arctan2(sin, cos) # should be radians, but double check
            alpha = alpha + angle_bins[argmax]

            theta_ray = calc_theta_ray(img, box_2d) # get horiz angle to the center of this box

            # format outputs
            net_output = format_net_output(box_2d, alpha, dim, theta_ray)


            # project 3d into 2d to visualize
            img = plot_regressed_3d_bbox(img, net_output, calib_file, info, truth_img)

            if single_car:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                cv2.imshow('Truth on top, Prediction on bottom for image', numpy_vertical)
                cv2.waitKey(0)




    # single image
    # cv2.imshow('Net output', img)
    # cv2.waitKey(0)

        # put truth image on top
        if not single_car:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
            cv2.waitKey(0)



if __name__ == '__main__':
    main()
