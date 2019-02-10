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


def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
    return cam_to_img


def plot_3d_bbox(img, label_info, calib):
    alpha = label_info['ThetaRay']
    # theta_ray = label_info['theta_ray']
    box_3d = []
    center = label_info['Location']
    dims = label_info['Dimension']
    # calib = label_info['Calib']
    # cam_to_img = calib['P2']
    cam_to_img = calib
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

    calib_file = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training/calib/000000.txt'
    camera_cal = get_calibration_cam_to_image(calib_file)


    data = Dataset.ImageDataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    # img = data.GetImage(10)
    #
    # info = data[10]['Label']
    #
    # for item in info:
    #     img = plot_3d_bbox(img, item, camera_cal)
    #
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    #
    # exit()
    #



    data = Dataset.BatchDataset(data, batches, bins, mode = 'eval')

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

    for i in range(data.num_of_patch):
        batch, centerAngle, info = data.EvalBatch()
        dimGT = info['Dimension']

        # print info
        # print batch
        # exit()

        angle = info['LocalAngle'] / np.pi * 180
        Ry = info['Ry']
        batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()


        [orient, conf, dim] = model(batch)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        # print dimGT
        # print dim

        # exit()

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


        print theta
        print Ry
        exit()

        theta_error = abs(Ry - theta)
        if theta_error > 180: theta_error = 360 - theta_error
        angle_error.append(theta_error)

        dim_error = np.mean(abs(np.array(dimGT) - dim))
        dimension_error.append(dim_error)


        # print(info)
        # exit()



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
