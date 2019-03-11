"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg


def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done
    angle_bins = generate_bins(2)

    img_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/image_2/'

    # using P from each frame
    calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    # using P_rect from global calibration file
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/camera_cal/'
    # calib_file = calib_path + "calib_cam_to_cam.txt"

    ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]

    for id in ids:

        start_time = time.time()

        img_file = img_path + id + ".png"

        # P for each frame
        calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        detections = yolo.detect(yolo_img)

        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                object = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = object.theta_ray
            input_img = object.img
            proj_matrix = object.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(img, truth_img, proj_matrix, box_2d, dim, alpha, theta_ray)

            print('Estimated pose: %s'%location)

        numpy_vertical = np.concatenate((truth_img, img), axis=0)
        cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)

        print("\n")
        print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
        print('-------------')
        if cv2.waitKey(0) != 32: # space bar
            exit()

if __name__ == '__main__':
    main()
