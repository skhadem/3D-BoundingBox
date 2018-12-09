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
import numpy as np
import itertools

import Model
import Dataset


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

   # yaw comes out of the net as a 2x2, seems to be confidence and angle?
   # get angle of highest confidence, (rad2deg?)
   tz = yaw[np.argmax(yaw[:,0]), :][1]

   Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
   Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
   Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])

   return np.dot(Rz, np.dot(Ry, Rx))


# this should be based on the paper. Math!
# orientation is car's local yaw angle ?, dimension is a 1x3 vector
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def calc_location(orient, dimension, calib, box_2d):

    # variables just like the equation
    K = calib
    R = rotation_matrix(orient)
    # [xmin, ymin, xmax, ymax]. This can be hard-coded. YOLO, etc. is consistant
    # box_corners = [box_2d[0][0], box_2d[0][1], box_2d[1][0], box_2d[1][1]]

    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    box_corners = [xmin, ymin, xmax, ymax]

    # print dimension
    # return None

    # check the order on these, Velodyne coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2


    corners = []

    # get all the corners
    # this gives all 8 corners with respect to 0,0,0 being center of box
    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                corners.append([dx*i, dy*j, dz*k])

    # print(corners)

    # need to get 64 possibilities for the order (xmin, ymin, xmax, ymax)
    # TODO:How to do this??

    # this should be 64 long, each possibility has 4 3d points
    # [ [ [3D corner for xmin], [for ymin] ... x4 ], ... x64 ]
    constraints = [[] for i in range(64)]


    constraints = list(itertools.product(corners, repeat=4))

    # from paper:
    # each vertical side of the 2D detection box can correspond to [+/- dx/2, . , +/- dz/2]
    # each horizontal side of the 2D detection box can correspond to [., +/- dx , +/- dz/2]
    # this gives 256, which ones to remove for zero pitch/roll ?
    # for i in range(0,)



    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = 1e09

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    for constraint in constraints:

        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)


        indicies = [0,1,0,1]
        X_array = [Xa, Xb, Xc, Xd]
        # print X_array
        # return None
        repeat = False
        test_x = list(itertools.combinations(X_array, 2))

        for x in test_x:
            if x[0] == x[1]:
                repeat = True
                break

        if repeat:
            continue

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)

            M[:3,3] = RX.reshape(3)

            # print(X)
            # print(M)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[0,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b)
        # print(A)
        # return None

        if error < best_error:
            best_loc = loc
            best_error = error


    # [X,Y,Z] in 3D coords
    return best_loc

def plot_3d_bbox(img, net_output, calib_file, truth_pose):
    cam_to_img = get_calibration_cam_to_image(calib_file)

    alpha = net_output['ThetaRay'] # ???? some angle
    # theta_ray = label_info['theta_ray']

    box_2d = net_output['Box_2D']
    dims = net_output['Dimension']
    orient = net_output['Orientation']

    # center = label_info['Location']
    center = calc_location(orient, dims, cam_to_img, box_2d)
    print "Estimated pose:"
    print center
    print "--"
    print "Truth pose:"
    print truth_pose
    print "-------------"

    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)

    return img # for now just 2d boxes

    # below will draw 3d box once the location is found with math (center)

    img = plot_3d(img, cam_to_img, alpha, dims, center)

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

    img = plot_3d(img, cam_to_img, alpha, dims, center)

    return img


def plot_3d(img, cam_to_img, alpha, dims, center):
    # radians (of the camera angle I think)
    # this angle is the same for every object in the scene
    rot_y = alpha / 180 * np.pi  + np.arctan(center[0]/center[2])

    box_3d = []
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
    calib_file = os.path.abspath(os.path.dirname(__file__)) + '/eval/calib/%s.txt' % image_id



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
            net_output['ThetaRay'] = theta
            net_output['Box_2D'] = info['Box_2D'] # from label, will eventually be from yolo

            truth_pose = info['Location']

            truth_dim = info['Dimension']

            net_output['Dimension'] = truth_dim

            # print "====="
            #
            # print truth_dim
            # print dim
            #
            # print "====="

            # project 3d into 2d to visualize
            # img = img_data.GetImage(0)
            img = plot_3d_bbox(img, net_output, calib_file, truth_pose)


    # cv2.imshow('Net output', img)
    # cv2.waitKey(0)

    numpy_vertical = np.concatenate((truth_img, img), axis=0)
    cv2.imshow('Truth on top, Prediction on bottom', numpy_vertical)
    cv2.waitKey(0)


    exit()








if __name__ == '__main__':
    main()


# create A
# A[0,:] = Ma[0,:3] - xmin * Ma[2,:3]
# A[1,:] = Ma[1,:3] - ymin * Mb[2,:3]
# A[2,:] = Ma[0,:3] - xmax * Mc[2,:3]
# A[3,:] = Ma[1,:3] - ymax * Md[2,:3]
#
# create b
# b[0] = xmin * Ma[2,3] - Ma[0,3]
# b[0] = ymin * Mb[2,3] - Mb[0,3]
# b[0] = xmax * Mc[2,3] - Mc[0,3]
# b[0] = ymax * Md[2,3] - Md[0,3]


# below is for error
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
