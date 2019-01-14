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
[ ] use purely truth values from label for dimension and orient to test math
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
# this is actually the projection matrix
def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            # cam_to_img[:,3] = 1
            return cam_to_img

def get_K(cab_f):
    for line in open(cab_f):
        if 'K_02' in line:
            cam_K = line.strip().split(' ')
            cam_K = np.asarray([float(cam_K) for cam_K in cam_K[1:]])
            return_matrix = np.zeros((3,4))
            return_matrix[:,:-1] = cam_K.reshape((3,3))

    return return_matrix

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))

            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0

            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,:4] = Tr

            return Tr_to_velo


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
    ty = yaw
    tz = pitch

    # from net:
    # yaw comes out of the net as a 2x2, seems to be confidence and angle?
    # get angle of highest confidence, (rad2deg?)
    # tz = yaw[np.argmax(yaw[:,0]), :][1]

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])


    return Ry.reshape([3,3]) # do we use this ?
    # return np.dot(np.dot(Rz,Ry), Rx)


# this should be based on the paper. Math!
# orientation is car's local yaw angle ?, dimension is a 1x3 vector
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def calc_location(orient, dimension, calib, box_2d):

    K = calib # actually P, but works ok

    # this one didn't work well
    # K = get_K(os.path.abspath(os.path.dirname(__file__)) + '/eval/calib/calib_cam_to_cam.txt')

    R = rotation_matrix(orient)

    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]


    box_corners = [xmin, ymin, xmax, ymax]

    # need to get 64 possibilities for the order (xmin, ymin, xmax, ymax)
    # this should be 64 long, each possibility has 4 3d points
    # [ [ [3D corner for xmin], [for ymin] ... x4 ], ... x64 ]
    # from paper:
    # each vertical side of the 2D detection box can correspond to [+/- dx/2, . , +/- dz/2]
    # each horizontal side of the 2D detection box can correspond to [., +/- dx , +/- dz/2]
    # this gives 256, which ones to remove for zero pitch/roll ?
    # TODO:How to do this correctly

    # for now, splitting into left, right, top, bottom of box
    # 4^4 = 256 combinations

    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    for i in (-1,1):
        for j in (-1,1):
            left_constraints.append([i*dx, j*dy, -dz])

    for i in (-1,1):
        for j in (-1,1):
            right_constraints.append([i*dx, j*dy, dz])

    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])

    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # car is facing opposite way, swap left/ right
    if orient < 0:
        temp = left_constraints
        left_constraints = right_constraints
        right_constraints = temp

    # 256 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

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
            print ("REPEAT")
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

            M = np.dot(K, M)

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
    # print best_error

    return best_loc, best_X

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners

# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point

# take in 3d points and plot them on image as red circles
def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        cv2.circle(img, (point[0], point[1]), 3, cv_colors.RED.value, thickness=-1)

def plot_3d(img, calib_file, ry, dimension, center):

    cam_to_img = get_calibration_cam_to_image(calib_file)

    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []

    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)

        box_3d.append(point)

    #TODO put into loop

    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.GREEN.value, 1)


    # TODO: put in loop
    # front_mark = []
    # front_mark.append((box_3d[0][0], box_3d[0][1]))
    # front_mark.append((box_3d[1][0], box_3d[1][1]))
    # front_mark.append((box_3d[2][0], box_3d[2][1]))
    # front_mark.append((box_3d[3][0], box_3d[3][1]))

    front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]


    cv2.line(img, front_mark[0], front_mark[3], cv_colors.BLUE.value, 1)
    cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)


# plot from net output. The orient should be global
# after done testing math, can remove label param
def plot_regressed_3d_bbox(img, net_output, calib_file, label, truth_img):
    cam_to_img = get_calibration_cam_to_image(calib_file)
    box_2d = net_output['Box_2D']

    # center of 2d box
    box_2d_center = [(box_2d[1][0] + box_2d[0][0]) / 2, (box_2d[1][1] + box_2d[0][1]) / 2]
    # what is this compared to theta_ray, don't think it's necessary here
    alpha = np.arctan(box_2d_center[0] / box_2d_center[1])

    dims = net_output['Dimension']
    orient = net_output['Orientation']

    # use truth for now
    truth_dims = label['Dimension']
    truth_orient = label['Ry']

    # the math! returns X, the corners used for constraint
    center, X = calc_location(truth_orient, truth_dims, cam_to_img, box_2d)

    center = [center[0][0], center[1][0], center[2][0]]

    truth_pose = label['Location']

    print "Estimated pose:"
    print center
    print "Truth pose:"
    print truth_pose
    print "-------------"

    plot_2d_box(truth_img, box_2d)

    # for now visualize truth pose, soon this should come from the calculated center
    # plot_3d(img, calib_file, truth_orient, truth_dims, truth_pose) # 3d boxes
    plot_3d(img, calib_file, truth_orient, truth_dims, center) # 3d boxes

    # plot the corners that were used
    # these corners returned are the ones that are unrotated, because they were
    # in the calculation. We must find the indicies of the corners used, then generate
    # the roated corners and visualize those

    corners = create_corners(truth_dims) # unrotated

    corner_indexes = [corners.index(i) for i in X] # get indexes

    # get the rotated version
    R = rotation_matrix(truth_orient)
    corners = create_corners(truth_dims, location=truth_pose, R=R)
    corners_used = [corners[i] for i in corner_indexes]

    # plot
    plot_3d_pts(truth_img, corners_used, truth_pose, cam_to_img=cam_to_img, relative=False)

    return img

# From KITTI : x = P2 * R0_rect * Tr_velo_to_cam * y
# Velodyne coords
def plot_truth_3d_bbox(img, label_info, calib_file):
    Ry = label_info['Ry']
    dims = label_info['Dimension']
    center = label_info['Location']

    plot_3d(img, calib_file, Ry, dims, center)

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

# using config # of bins, create array of 0..2pi split into bins
# radians
def generate_angle_bins(bins):
    interval = 2 * np.pi / bins

    angle_bins = np.zeros(bins)

    for i in range(1, bins):
        angle_bins[i] = i * interval

    return angle_bins

#TODO: implement this:
#https://math.stackexchange.com/questions/1320285/convert-a-pixel-displacement-to-angular-rotation
# helpful:
#https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
def calc_theta_ray(box_2d):
    pass

def format_net_output(box_2d, orient, dim):
    net_output = {}
    net_output['Box_2D'] = box_2d
    net_output['Orientation'] = orient
    net_output['Dimension'] = dim

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

        truth_img = draw_truth_boxes(img_idx, img_data, calib_file)
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
            theta = np.arctan2(sin, cos) # should be radians, but double check
            theta = theta + angle_bins[argmax]

            theta_ray = calc_theta_ray(box_2d) # get horiz angle to the center of this box
            # theta = theta + theta_ray

            theta = 360 - info['ThetaRay'] - theta # this should be done with math on pixel center of box

            # why format like this?
            # if theta > 0: theta -= int(theta / 360) * 360
            # elif theta < 0: theta += (int(-theta / 360) + 1) * 36

            # why this?
            Ry = info['Ry']
            if Ry > 0: Ry -= int(Ry / 360) * 360
            elif Ry < 0: Ry += (int(-Ry / 360) + 1) * 360

            # format outputs
            net_output = format_net_output(box_2d, orient, dim)

            # project 3d into 2d to visualize
            img = plot_regressed_3d_bbox(img, net_output, calib_file, info, truth_img)

    # single image
    # cv2.imshow('Net output', img)
    # cv2.waitKey(0)

        # put truth image on top
        numpy_vertical = np.concatenate((truth_img, img), axis=0)
        cv2.imshow('Truth on top, Prediction on bottom for image', numpy_vertical)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
