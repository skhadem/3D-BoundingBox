from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data

import os

def main():

    # hyper parameters
    epochs = 10
    batch_size = 8
    alpha = 1
    w = 1


    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)


    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    # device = torch.device('cuda:0')

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).cuda()

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # conf_loss_func = nn.CrossEntropyLoss().cuda()
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(epochs):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            # orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            # dim_loss = dim_loss_func(dim, truth_dim)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            # first, get only confidence to converge
            # loss_theta = conf_loss + orient_loss * w
            # loss = alpha * dim_loss + loss_theta
            loss = conf_loss

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 10 == 0:
                print "--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item())
                passes = 0

            passes += 1
            curr_batch += 1

        name = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
        name += 'epoch_%s.pkl' % epoch
        print "===================="
        print "Done with epoch %s!" % epoch
        print "Saving weights as %s ..." % name
        torch.save(model.state_dict(),name)
        print "===================="











if __name__=='__main__':
    main()
