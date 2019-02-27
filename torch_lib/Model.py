import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# TODO optimize using torch functions
def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    total_loss = torch.tensor(0).float().cuda()
    batch_size = orient_batch.size()[0]

    for row in range(0, batch_size):
        row_loss = 0
        confGT = confGT_batch[row]
        orientGT = orientGT_batch[row]
        orient = orient_batch[row]

        n_theta = torch.sum(confGT).float().cuda()

        # for each bin that covers GT angle
        for conf_arg in range(0, len(confGT)):
            if confGT[conf_arg] != 1:
                continue

            # recover GT angle diff
            #TODO use arctan2 instead
            theta_diff = torch.acos(orientGT[conf_arg][0]).float().cuda()
            estimated_theta_diff = torch.acos(orient[conf_arg][0]).float().cuda()
            row_loss += torch.cos(theta_diff - estimated_theta_diff)

        total_loss += ( 1/n_theta ) * row_loss

    return -total_loss/batch_size

class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                    # nn.Softmax()
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension
