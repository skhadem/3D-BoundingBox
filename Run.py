"""
Goals:

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
import random

import Model
import Dataset


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

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


# need to double check the coordinate system used, I believe it is from camera coords
# using this math: https://en.wikipedia.org/wiki/Rotation_matrix
def rotation_matrix(yaw, pitch=0, roll=0):
    # print yaw
    tx = roll
    ty = pitch

    # from net:
    # yaw comes out of the net as a 2x2, seems to be confidence and angle?
    # get angle of highest confidence, (rad2deg?)
    # tz = yaw[np.argmax(yaw[:,0]), :][1]

    # from truth:
    tz = yaw # should be in radians

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])


    # return Rz.reshape([3,3]) # do we use this ?
    return np.dot(Rz, np.dot(Ry, Rx))


# this should be based on the paper. Math!
# orientation is car's local yaw angle ?, dimension is a 1x3 vector
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def calc_location(orient, dimension, calib, box_2d):

    # variables with same names as the equation
    K = calib
    R = rotation_matrix(np.deg2rad(orient))

    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    box_corners = [xmin, ymin, xmax, ymax]

    # TODO: check the order on these
    # print dimension
    dx = dimension[1] / 2
    dy = dimension[0] / 2
    dz = dimension[2] / 2


    corners = []

    # get all the corners
    # this gives all 8 corners with respect to 0,0,0 being center of box
    for i in [1, -1]:
        for j in [0,-2]:
            for k in [1,-1]:
                corners.append([dx*i, dy*j, dz*k])


    # need to get 64 possibilities for the order (xmin, ymin, xmax, ymax)
    # this should be 64 long, each possibility has 4 3d points
    # [ [ [3D corner for xmin], [for ymin] ... x4 ], ... x64 ]
    # from paper:
    # each vertical side of the 2D detection box can correspond to [+/- dx/2, . , +/- dz/2]
    # each horizontal side of the 2D detection box can correspond to [., +/- dx , +/- dz/2]
    # this gives 256, which ones to remove for zero pitch/roll ?
    # TODO:How to do this??
    constraints = [[] for i in range(64)]

    # for now, use all 4096 possible
    constraints = list(itertools.product(corners, repeat=4))

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        # we don't want ones with the 4 of the same corners
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

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # print count

    return best_loc, best_X

def plot_3d_pt(img, pts, center, calib_file=None, cam_to_img=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        for i in range(3):
            pt[i] = pt[i] + center[i]

        point = np.array(pt)
        point = np.append(point, 1)
        point = np.dot(cam_to_img, point)
        point = point[:2]/point[2]
        point = point.astype(np.int16)

        cv2.circle(img, (point[0], point[1]), 3, cv_colors.RED.value, thickness=-1)


# plot from net output
def plot_3d_bbox(img, net_output, cam_to_img):
    cam_to_img = get_calibration_cam_to_image(cam_to_img)

    alpha = net_output['ThetaRay'] # Still confused on this angle

    box_2d = net_output['Box_2D']
    dims = net_output['Dimension']
    orient = net_output['Orientation']

    # center = label_info['Location']
    center, X = calc_location(orient, dims, cam_to_img, box_2d)

    center = [center[0][0], center[1][0], center[2][0]]

    truth_pose = net_output['Truth_Pose']

    print "Estimated pose:"
    print center
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

    # for now visualize truth pose
    center = truth_pose
    img = plot_3d(img, cam_to_img, alpha, dims, center) # 3d boxes
    plot_3d_pt(img, X, center, cam_to_img=cam_to_img) # red corner points that were used

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
        if item['Class'] == 'DontCare':
            continue
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

            if info['Class'] == 'DontCare':
                continue

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
            net_output['ThetaRay'] = info['ThetaRay']
            net_output['Box_2D'] = info['Box_2D'] # from label, will eventually be from yolo

            truth_dim = info['Dimension']

            # TODO: this seems like the correct truth angle, but double check
            truth_orient = info['Ry']

            net_output['Orientation'] = truth_orient # orient
            net_output['Dimension'] = truth_dim # dim

            net_output['Truth_Pose'] = info['Location']


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
