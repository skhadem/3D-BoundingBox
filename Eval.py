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

    data = Dataset.ImageDataset(path + '/training')
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
        angle = info['LocalAngle'] / np.pi * 180
        Ry = info['Ry']
        batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()


        [orient, conf, dim] = model(batch)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]
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
