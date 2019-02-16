from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg as V

import os

def main():

    epochs = 3
    each_iter = 10
    batch_size = 50

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path, batch_size)

    batch_split = dataset.generate_batch_splits(dataset.num_objects, batch_size)
    min_idx = 0
    max_idx = 1



    for i in range(0, epochs):
        # split up the dataset
        for j in range(0,len(batch_split)):
            min = batch_split[min_idx]
            max = batch_split[max_idx]
            dataset.new_batch(min, max)
            # iterate over each batch many times
            for k in range(0, each_iter):
                print k
                objects = dataset.shuffle_batch() # get shuffled list of objects
                # pass in cropped image to net
                for obj in objects:
                    pass
                    #forward
                    #loss
                    #backward
            min+=1
            max+=1

            print "--------------------"
            print "done with batch %s/%s"%(min, len(batch_split))
            print "loss: "
            print "--------------------"











if __name__=='__main__':
    main()
