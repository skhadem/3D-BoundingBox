"""
Big Picture:
- use the 2D box of an object in scene
- pass cropped image of object through the model
- net outputs dimension and oreintation, then calculate the location (T) using camera
    cal and lots of math
- put the calculated 3d location onto 2d image using plot_regressed_3d_bbox
- visualize
Plan:
[x] reformat data structure to understand it better
[x] use purely truth values from label for dimension and orient to test math
[x] regress dimension and orient from net
[ ] use yolo or rcnn to get the 2d box and class, so run from just an image (and cal)
[ ] Try and optimize to be able to run on video
[ ] Ros node eventually
Random TODOs:
[ ] loops inside of plotting functions
[x] Move alot of functions to a library and import it
Notes:
- The net outputs an angle (actually a sin and cos) relative to an angle defined
    by the # of bins, thus the # of bins used to train model should be known
- Everything should be using radians, just for consistancy
- output dimension is actually difference from the class average
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
import numpy as np

# to run car by car
single_car = False

def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print 'No previous model found, please train first!'
        exit()
    else:
        print 'Using previous model %s'%model_lst[-1]
        my_vgg = vgg.vgg19_bn(pretrained=True)
        #TODO model in Cuda throws an error
        model = Model.Model(features=my_vgg.features, bins=2)
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    averages = ClassAverages.ClassAverages()

    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    exit()

    all_images = dataset.all_objects()
    for key in sorted(all_images.keys()):
        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for object in objects:
            label = object.label
            theta_ray = object.theta_ray
            input_img = object.img

            input_tensor = torch.zeros([1,3,224,224])
            input_tensor[0,:,:,:] = input_img
            input_tensor.cuda()

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(label['Class'])

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += dataset.angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

            print 'Estimated pose: %s'%location
            print 'Truth pose: %s'%label['Location']
            print '-------------'

            # plot car by car
            if single_car:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
                cv2.waitKey(0)

        # plot image by image
        if not single_car:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()
