import cv2
import numpy as np
import os
import random

from library.File import *


"""
Will hold all the ImageData objbects. Should only be used when evaluating or
training. When running, pass image and array of 2d boxes directly into an ImageData
object
"""
class Dataset:
    def __init__(self, path, batch_size=100):
        self.top_label_path = path + "/label_2"
        self.top_img_path = path + "/image_2"
        self.top_calib_path = path + "/calib"

        self.k = self.get_K(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)
        self.num_objects = self.total_num_objects(self.ids)


        self.current = 0



    def total_num_objects(self, ids):
        total = 0
        for id in self.ids:
            total += len(self.parse_label(self.top_label_path + '/%s.txt'%id))

        return total

    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry
                    })
        return buf

    def all_objects(self):
        data = {}
        for id in self.ids:
            data[id] = {}
            img_path = self.top_img_path + '/%s.png'%id
            img = cv2.imread(img_path)
            data[id]['Image'] = img

            calib_path = self.top_calib_path + '/%s.txt'%id
            data[id]['Calib'] = get_calibration_cam_to_image(calib_path)


            label_path = self.top_label_path + '/%s.txt'%id
            labels = self.parse_label(label_path)
            objects = []
            for label in labels:
                box_2d = label['Box_2D']
                detection_class = label['Class']
                objects.append(DetectedObject(img, detection_class, box_2d, self.k, label=label))

            data[id]['Objects'] = objects

        return data


    def generate_batch_splits(self, total, batch_size):
        splits = [x for x in range(0, total, batch_size)]
        splits.append(total)

        return splits


    def new_batch(self, min_idx, max_idx):
        self.current_ids = self.ids[min_idx:max_idx]

    # shuffle the current batch and return the objects
    def shuffle_batch(self):
        random.shuffle(self.current_ids)
        objects = []

        for id in self.current_ids:
            for obj in self.generate_objects(id):
                objects.append(obj)

        return objects

    # from a filename generate a DetectedObject object
    def generate_objects(self, id):
        img_path = self.top_img_path + '/%s.png'%id
        img = cv2.imread(img_path)

        calib_path = self.top_calib_path + '/%s.txt'%id
        label_path = self.top_label_path + '/%s.txt'%id
        labels = self.parse_label(label_path)

        objects = []
        for label in labels:
            box_2d = label['Box_2D']
            detection_class = label['Class']
            objects.append(DetectedObject(img, detection_class, box_2d, self.k, label=label))

        return objects



    def get_K(self, cab_f):
        for line in open(cab_f, 'r'):
            if 'K_02' in line:
                cam_K = line.strip().split(' ')
                cam_K = np.asarray([float(cam_K) for cam_K in cam_K[1:]])
                return_matrix = np.zeros((3,4))
                return_matrix[:,:-1] = cam_K.reshape((3,3))

        return return_matrix



# ------ python overrides

    def __iter__(self):
        return self

    def next(self):
        if self.current  == len(self.ids):
            raise StopIteration
        else:
            self.current += 1
            id = self.ids[self.current-1]
            return self.data[id]

    def __getitem__(self, index):
        return self.data[self.ids[index]]


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, K, label=None):
        self.theta_ray = self.calc_theta_ray(img, box_2d, K)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class


    def calc_theta_ray(self, img, box_2d, K):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * K[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        img=img.astype(np.float) / 255

        img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229

        # crop image
        batch = np.zeros([1, 3, 224, 224], np.float)
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # cv2.imshow('hello', crop) # to see the input cropped section
        # cv2.waitKey(0)

        # recolor, reformat
        batch[0, 0, :, :] = crop[:, :, 2]
        batch[0, 1, :, :] = crop[:, :, 1]
        batch[0, 2, :, :] = crop[:, :, 0]

        return batch
