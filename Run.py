"""
Goal:

- use the 2D box from the label (cause we can get that from yolo)
- resize said box in the EvalBatch method
- pass that through the model (it's a tensor)
- using output dimension and oreintation calculate the location ( based on camera cal )
    this can be done by the method in the paper
- back propogate the 3d location to a 2d location using plot_3d_bbox
- draw on image, look at output

"""


import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/Library')
import cv2
import yaml
import time
import datetime
from enum import Enum

import Model
import Dataset
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

from math import sin, cos

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)

# read camera cal file and get intrinsic params
def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# need to double check the coordinate system used, I believe it is Velodyne
# using this math: https://en.wikipedia.org/wiki/Rotation_matrix
def rotation_matrix(yaw, pitch=0, roll=0):
   tx = roll
   ty = pitch
   tz = yaw

   Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
   Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
   Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])

   return np.dot(Rz, np.dot(Ry, Rx))


# this should be based on the paper. Math!
# orientation is car's local yaw angle ?, dimension is a 1x3 vector
# calib is a 3x4 matrix, box_2d is [[x, y], ... (x4)]
def calc_location(orient, dimension, calib, box_2d):
    # variables just like the equation
    K = calib
    # R = rotation_matrix(orient)




def plot_3d_bbox(img, net_output, calib_file):
    cam_to_img = get_calibration_cam_to_image(calib_file)

    alpha = net_output['ThetaRay'] # ???? some angle
    # theta_ray = label_info['theta_ray']

    box_2d = net_output['Box_2D']
    dims = net_output['Dimension']
    orient = net_output['Orientation']

    # center = label_info['Location']
    center = calc_location(orient, dims, cam_to_img, box_2d)

    # print(box_2d)

    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)


    cv2.line(img, pt1, pt2, (255,0,0), 2)
    cv2.line(img, pt2, pt3, (255,0,0), 2)
    cv2.line(img, pt3, pt4, (255,0,0), 2)
    cv2.line(img, pt4, pt1, (255,0,0), 2)


    return img

    box_3d = []


    rot_y = alpha / 180 * np.pi  + np.arctan(center[0]/center[2])


    for i in [1,-1]:
        for j in [1,-1]:
            for k in [0,1]:
                point = np.copy(center)
                point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)
                point[1] = center[1] - k * dims[0]

                point = np.append(point, 1)
                point = np.dot(cam_to_img, point)
                point = point[:2]/point[2]
                point = point.astype(np.int16)
                box_3d.append(point)

    front_mark = []
    for i in range(4):
        point_1_ = box_3d[2*i]
        point_2_ = box_3d[2*i+1]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), cv_colors.GREEN.value, 1)
        if i == 0 or i == 3:
            front_mark.append((point_1_[0], point_1_[1]))
            front_mark.append((point_2_[0], point_2_[1]))

    cv2.line(img, front_mark[0], front_mark[-1], cv_colors.BLUE.value, 1)
    cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)

    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i+2)%8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), cv_colors.GREEN.value, 1)

    return img


# From KITTI : x = P2 * R0_rect * Tr_velo_to_cam * y
# Velodyne coords
def plot_truth_3d_bbox(img, label_info, calib_file):
    cam_to_img = get_calibration_cam_to_image(calib_file)


    # seems to be the car's orientation
    # I think this is the red angle, which is regressed
    alpha = label_info['ThetaRay']

    dims = label_info['Dimension']
    center = label_info['Location']



    box_3d = []


    # radians (of the camera angle I think)
    # this angle is the same for every object in the scene
    rot_y = alpha / 180 * np.pi  + np.arctan(center[0]/center[2])

    # 3d corners
    for i in [1,-1]:
        for j in [1,-1]:
            for k in [0,1]:
                point = np.copy(center)
                point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)
                point[1] = center[1] - k * dims[0]

                point = np.append(point, 1)
                point = np.dot(cam_to_img, point)
                point = point[:2]/point[2]
                point = point.astype(np.int16)
                box_3d.append(point)

    front_mark = []
    for i in range(4):
        point_1_ = box_3d[2*i]
        point_2_ = box_3d[2*i+1]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), cv_colors.GREEN.value, 1)

        # get the front of the box
        if alpha > 90:
            if i == 0 or i == 3:
                front_mark.append((point_1_[0], point_1_[1]))
                front_mark.append((point_2_[0], point_2_[1]))
        if alpha <= 90:
            if i == 1 or i == 2:
                front_mark.append((point_1_[0], point_1_[1]))
                front_mark.append((point_2_[0], point_2_[1]))

    # x on front of object
    cv2.line(img, front_mark[0], front_mark[-1], cv_colors.BLUE.value, 1)
    cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)

    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i+2)%8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), cv_colors.GREEN.value, 1)

    return img



def draw_truth_boxes(img_idx, img_dataset, calib_file):
    # visualize with truth data
    img = img_dataset.GetRawImage(img_idx)

    info = img_dataset[img_idx]['Label']

    for item in info:
        img = plot_truth_3d_bbox(img, item, calib_file)

    return img



def main():
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
    bins = config['bins']
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




    img_data = Dataset.MyImageDataset(path + '/eval')
    data = Dataset.MyBatchDataset(img_data, batches, bins, mode = 'eval')

    # hard code for now, eventually should be based on image id gotten in loop
    image_id = '000010'
    calib_file = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/eval/calib/%s.txt' % image_id



    truth_img = draw_truth_boxes(0,img_data, calib_file)
    # cv2.imshow('Thruth data for index %s'%0,truth_img)
    # cv2.waitKey(0)

    # for error
    # angle_error = []
    # dimension_error = []

    # for i in range(data.num_of_patch):
    # angle = info['LocalAngle'] / np.pi * 180
    # Ry = info['Ry'] # ????

    for img_idx in range(0,data.num_images): # through the images
        # get the raw image, for visualization purposes
        img = img_data.GetRawImage(img_idx)

        # batches is the all objects in image, cropped
        batches, centerAngles, infos = data.formatForModel(img_idx)

        for i, batch in enumerate(batches):
            info = infos[i]

            # create tensor
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

            # run through the net, format output
            [orient, conf, dim] = model(batch)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            # wtf is this for????
            argmax = np.argmax(conf)
            orient_max = orient[argmax, :]
            cos = orient_max[0]
            sin = orient_max[1]
            theta = np.arctan2(sin, cos) / np.pi * 180
            theta = theta + centerAngles[0][argmax] / np.pi * 180
            theta = 360 - info['ThetaRay'] - theta
            if theta > 0: theta -= int(theta / 360) * 360
            elif theta < 0: theta += (int(-theta / 360) + 1) * 360


            # print(orient)
            # print(orient_max)
            # print(theta)
            # print '---'
            Ry = info['Ry']
            if Ry > 0: Ry -= int(Ry / 360) * 360
            elif Ry < 0: Ry += (int(-Ry / 360) + 1) * 360
            # print(Ry)
            # exit()


            # format to pass into *math* functions and visualize
            net_output = {}
            net_output['Orientation'] = orient
            net_output['Dimension'] = dim
            net_output['ThetaRay'] = theta
            net_output['Box_2D'] = info['Box_2D'] # from label, will eventually be from yolo

            # project 3d into 2d to visualize
            # img = img_data.GetImage(0)
            img = plot_3d_bbox(img, net_output, calib_file)


    # cv2.imshow('Net output', img)
    # cv2.waitKey(0)

    numpy_vertical = np.concatenate((truth_img, img), axis=0)
    cv2.imshow('Truth on top, Prediction on bottom', numpy_vertical)
    cv2.waitKey(0)


    exit()








if __name__ == '__main__':
    main()


# all for error
#
#     argmax = np.argmax(conf)
#     orient = orient[argmax, :]
#     cos = orient[0]
#     sin = orient[1]
#
#     theta = np.arctan2(sin, cos) / np.pi * 180
#     theta = theta + centerAngle[argmax] / np.pi * 180
#     theta = 360 - info['ThetaRay'] - theta
#
#     if theta > 0: theta -= int(theta / 360) * 360
#     elif theta < 0: theta += (int(-theta / 360) + 1) * 360
#
#     if Ry > 0: Ry -= int(Ry / 360) * 360
#     elif Ry < 0: Ry += (int(-Ry / 360) + 1) * 360
#
#     theta_error = abs(Ry - theta)
#     if theta_error > 180: theta_error = 360 - theta_error
#     angle_error.append(theta_error)
#
#     dim_error = np.mean(abs(np.array(dimGT) - dim))
#     dimension_error.append(dim_error)
#
#
#     print(info)
#     exit()
#
#     #if i % 60 == 0:
#     #    print (theta, Ry)
#     #    print (dim.tolist(), dimGT)
#     if i % 1000 == 0:
#         now = datetime.datetime.now()
#         now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
#         print '------- %s %.5d -------'%(now_s, i)
#         print 'Angle error: %lf'%(np.mean(angle_error))
#         print 'Dimension error: %lf'%(np.mean(dimension_error))
#         print '-----------------------------'
# print 'Angle error: %lf'%(np.mean(angle_error))
# print 'Dimension error: %lf'%(np.mean(dimension_error))
