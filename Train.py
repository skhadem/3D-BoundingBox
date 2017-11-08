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
        os.mkdir(store_path)

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
    data = Dataset.BatchDataset(data, batches, bins)
    
    if len(model_lst) == 0:
        print 'No previous model found, start training'
        vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=vgg.features, bins=bins).cuda()
    else:
        print 'Find previous model %s'%model_lst[-1]
        vgg = vgg.vgg19_bn(pretrained=False)
        model = Model.Model(features=vgg.features, bins=bins).cuda()
        params = torch.load(store_path + '/%s'%model_lst[-1])
        model.load_state_dict(params)

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    dim_LossFunc = nn.MSELoss().cuda()
    conf_LossFunc = nn.CrossEntropyLoss().cuda()

    iter_each_time = round(float(data.num_of_patch) / batches)
    for epoch in range(epochs):
        for i in range(int(iter_each_time)):
            batch, confidence, confidence_multi, angleDiff, dimGT = data.Next()

            confidence_arg = np.argmax(confidence, axis = 1)
            batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()
            confidence = Variable(torch.LongTensor(confidence.astype(np.int)), requires_grad=False).cuda()
            confidence_multi = Variable(torch.LongTensor(confidence_multi.astype(np.int)), requires_grad=False).cuda()
            angleDiff = Variable(torch.FloatTensor(angleDiff), requires_grad=False).cuda()
            dimGT = Variable(torch.FloatTensor(dimGT), requires_grad=False).cuda()
            confidence_arg = Variable(torch.LongTensor(confidence_arg.astype(np.int)), requires_grad=False).cuda()

            [orient, conf, dim] = model(batch)
            conf_loss = conf_LossFunc(conf, confidence_arg)
            orient_loss = Model.OrientationLoss(orient, angleDiff, confidence_multi)
            dim_loss = dim_LossFunc(dim, dimGT)
            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta

            if i % 1000 == 0:
                c_l = conf_loss.cpu().data.numpy()
                o_l = orient_loss.cpu().data.numpy()
                d_l = dim_loss.cpu().data.numpy()
                t_l = loss.cpu().data.numpy()
                now = datetime.datetime.now()
                now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
                print '------- %s Epoch %.2d -------'%(now_s, epoch)
                print 'Confidence Loss: %lf'%c_l
                print 'Orientation Loss: %lf'%o_l
                print 'Dimension Loss: %lf'%d_l
                print 'Total Loss: %lf'%t_l
                print '-----------------------------'
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
        now = datetime.datetime.now()
        now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
        name = store_path + '/model_%s.pkl'%now_s
        torch.save(model.state_dict(), name)







