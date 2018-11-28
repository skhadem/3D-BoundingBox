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

import Model
import Dataset
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

# read camera cal file and get intrinsic params
def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

# from the 2 corners, return the 4 corners of a box in CCW order
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# this should be based on the paper
# orientation is a quaternion, dimension is a 1x3 vector
# calib is a 3x4 matrix, box_2d is [[x, y], ... (x4) ]
def calc_location(orient, dimension, calib, box_2d):
    pass



def plot_3d_bbox(img, net_output, calib_file):
    cam_to_img = get_calibration_cam_to_image(calib_file)

    alpha = net_output['ThetaRay'] # ???? some angle
    # theta_ray = label_info['theta_ray']

    box_2d = net_output['Box_2D']
    dims = net_output['Dimension']
    orient = net_output['Orientation']

    # center = label_info['Location']
    center = calc_location(orient, dims, cam_to_img, box_2d)

    print(box_2d)

    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)


    cv2.line(img, pt1, pt2, (255,0,0), 2)
    cv2.line(img, pt2, pt3, (255,0,0), 2)
    cv2.line(img, pt3, pt4, (255,0,0), 2)
    cv2.line(img, pt4, pt1, (255,0,0), 2)


    return img

    box_3d = []

    # calib = label_info['Calib']
    # cam_to_img = calib['P2']
    rot_y = alpha / 180 * np.pi  + np.arctan(center[0]/center[2])
    # import pdb; pdb.set_trace()

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
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255,0,0), 2)
        if i == 0 or i == 3:
            front_mark.append((point_1_[0], point_1_[1]))
            front_mark.append((point_2_[0], point_2_[1]))

    cv2.line(img, front_mark[0], front_mark[-1], (255,0,0), 2)
    cv2.line(img, front_mark[1], front_mark[2], (255,0,0), 2)

    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i+2)%8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255,0,255), 2)

    return img



if __name__ == '__main__':
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        print 'No folder named \"models/\"'
        exit()

    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    path = config['kitti_path']
    epochs = config['epochs']
    batches = config['batches']
    bins = config['bins']
    alpha = config['alpha']
    w = config['w']





    img_data = Dataset.MyImageDataset(path + '/eval')

    # visualize with truth data
    # img = data.GetImage(0)
    #
    # info = data[0]['Label']
    #
    # for item in info:
    #     img = plot_3d_bbox(img, item, calib_file)
    #
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    #
    # exit()




    data = Dataset.MyBatchDataset(img_data, batches, bins, mode = 'eval')

    if len(model_lst) == 0:
        print 'No previous model found, please check it'
        exit()
    else:
        print 'Find previous model %s'%model_lst[-1]
        vgg = vgg.vgg19_bn(pretrained=False)
        model = Model.Model(features=vgg.features, bins=bins).cuda()
        params = torch.load(store_path + '/%s'%model_lst[-1])
        model.load_state_dict(params)
        model.eval()

    angle_error = []
    dimension_error = []
    print(data.num_of_patch)
    for i in range(data.num_of_patch):
        batch, centerAngle, info = data.EvalBatch() # this should be called for each item in an image
        dimGT = info['Dimension']

        # angle = info['LocalAngle'] / np.pi * 180

        Ry = info['Ry'] # ????

        batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

        # run through the net here
        [orient, conf, dim] = model(batch)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        # wtf is this ????
        argmax = np.argmax(conf)
        orient_max = orient[argmax, :]
        cos = orient_max[0]
        sin = orient_max[1]
        theta = np.arctan2(sin, cos) / np.pi * 180

        # print info

        #
        # print orient
        # print dim
        # print theta



        net_output = {}
        net_output['Orientation'] = orient
        net_output['Dimension'] = dim
        net_output['ThetaRay'] = theta

        net_output['Box_2D'] = info['Box_2D'] # from label, will eventually be from yolo



        # project 3d into 2d to visualize
        img = img_data.GetImage(0)
        calib_file = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/eval/calib/000010.txt'
        annotated_img = plot_3d_bbox(img, net_output, calib_file)
        cv2.imshow('Net output', annotated_img)
        cv2.waitKey(0)

        exit()




        # is this all for error???

        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]

        theta = np.arctan2(sin, cos) / np.pi * 180
        theta = theta + centerAngle[argmax] / np.pi * 180
        theta = 360 - info['ThetaRay'] - theta

        if theta > 0: theta -= int(theta / 360) * 360
        elif theta < 0: theta += (int(-theta / 360) + 1) * 360

        if Ry > 0: Ry -= int(Ry / 360) * 360
        elif Ry < 0: Ry += (int(-Ry / 360) + 1) * 360

        theta_error = abs(Ry - theta)
        if theta_error > 180: theta_error = 360 - theta_error
        angle_error.append(theta_error)

        dim_error = np.mean(abs(np.array(dimGT) - dim))
        dimension_error.append(dim_error)


        print(info)
        exit()

        #if i % 60 == 0:
        #    print (theta, Ry)
        #    print (dim.tolist(), dimGT)
        if i % 1000 == 0:
            now = datetime.datetime.now()
            now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
            print '------- %s %.5d -------'%(now_s, i)
            print 'Angle error: %lf'%(np.mean(angle_error))
            print 'Dimension error: %lf'%(np.mean(dimension_error))
            print '-----------------------------'
    print 'Angle error: %lf'%(np.mean(angle_error))
    print 'Dimension error: %lf'%(np.mean(dimension_error))
