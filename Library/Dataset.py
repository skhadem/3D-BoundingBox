import os
import sys
import cv2
import glob
import numpy as np


class ImageDataset:
    def __init__(self, path):
        self.img_path = path + '/image_2'
        self.label_path = path + '/label_2'

        self.IDLst = [x.split('.')[0] for x in sorted(os.listdir(self.img_path))] 

    def __getitem__(self, index):
        tmp = {}
        #img = cv2.imread(self.img_path + '/%s.png'%self.IDLst[index], cv2.IMREAD_COLOR)
        with open(self.label_path + '/%s.txt'%self.IDLst[index], 'r') as f:
            buf = []
            for line in f:
                line = line[:-1].split(' ')
                for i in range(1, len(line)):
                    line[i] = float(line[i])
                Class = line[0]
                Alpha = line[3] / np.pi * 180
                Ry = line[14] / np.pi * 180
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]
                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                ThetaRay = (np.arctan2(Location[2], Location[0])) / np.pi * 180
                #if Ry > 0:
                #    LocalAngle = (180 - Ry) + (180 - ThetaRay)
                #else:
                #    LocalAngle = 360 - (ThetaRay + Ry)
                LocalAngle = 360 - (ThetaRay + Ry)
                if LocalAngle > 360:
                    LocalAngle -= 360
                #LocalAngle = Ry - ThetaRay
                LocalAngle = LocalAngle / 180 * np.pi

                if LocalAngle < 0:
                    LocalAngle += 2 * np.pi
                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimension': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry,
                        'ThetaRay': ThetaRay,
                        'LocalAngle': LocalAngle
                    })

        tmp['ID'] = self.IDLst[index]
        #tmp['Image'] = img
        tmp['Label'] = buf
        return tmp
    def GetImage(self, idx):
        name = '%s/%s.png'%(self.img_path, self.IDLst[idx])
        img = cv2.imread(name, cv2.IMREAD_COLOR).astype(np.float) / 255
        img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
        return img

    def __len__(self):
        return len(self.IDLst)

class BatchDataset:
    def __init__(self, imgDataset, batchSize = 1, bins = 3, overlap = 25/180.0 * np.pi, mode='train'):
        self.imgDataset = imgDataset
        self.batchSize = batchSize
        self.bins = bins
        self.overlap = overlap
        self.mode = mode
        self.imgID = None
        centerAngle = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            centerAngle[i] = i * interval
        self.centerAngle = centerAngle
        #print centerAngle / np.pi * 180
        self.intervalAngle = interval
        self.info = self.getBatchInfo()
        self.Total = len(self.info)
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = 35570
        else:
            self.idx = 35570
            self.num_of_patch = 5000
        #print len(self.info)
        #print self.info
    def getBatchInfo(self):
        #
        # get info of all crop image
        #   
        data = []
        total = len(self.imgDataset)
        centerAngle = self.centerAngle
        intervalAngle = self.intervalAngle
        for idx, one in enumerate(self.imgDataset):
            ID = one['ID']
            #img = one['Image']
            allLabel = one['Label']
            for label in allLabel:
                if label['Class'] != 'DontCare':
                    #crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
                    LocalAngle = label['LocalAngle']
                    confidence = np.zeros(self.bins)
                    confidence_multi = np.zeros(self.bins)
                    for i in range(self.bins):
                        diff = abs(centerAngle[i] - LocalAngle)
                        if diff > np.pi:
                            diff = 2 * np.pi - diff
                        if diff <= intervalAngle / 2 + self.overlap:
                            confidence_multi[i] = 1
                        if diff < intervalAngle / 2:
                            confidence[i] = 1
                    n = np.sum(confidence)
                    data.append({
                                'ID': ID, # img ID
                                'Index': idx, # id in Imagedataset
                                'Box_2D': label['Box_2D'],
                                'Dimension': label['Dimension'],
                                'Location': label['Location'],
                                'LocalAngle': LocalAngle,
                                'Confidence': confidence,
                                'ConfidenceMulti': confidence_multi,
                                'Ntheta':n,
                                'Ry': label['Ry'],
                                'ThetaRay': label['ThetaRay']
                            })
        return data

    def Next(self):
        batch = np.zeros([self.batchSize, 3, 224, 224], np.float) 
        confidence = np.zeros([self.batchSize, self.bins], np.float)
        confidence_multi = np.zeros([self.batchSize, self.bins], np.float)
        ntheta = np.zeros(self.batchSize, np.float)
        angleDiff = np.zeros([self.batchSize, self.bins], np.float)
        dim = np.zeros([self.batchSize, 3], np.float)
        record = None
        for one in range(self.batchSize):
            data = self.info[self.idx]
            imgID = data['Index']
            if imgID != record:
                img = self.imgDataset.GetImage(imgID)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #cv2.namedWindow('GG')
                #cv2.imshow('GG', img)
                #cv2.waitKey(0)
            pt1 = data['Box_2D'][0]
            pt2 = data['Box_2D'][1]
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            batch[one, 0, :, :] = crop[:, :, 2]
            batch[one, 1, :, :] = crop[:, :, 1]
            batch[one, 2, :, :] = crop[:, :, 0]
            confidence[one, :] = data['Confidence'][:]
            confidence_multi[one, :] = data['ConfidenceMulti'][:]
            #confidence[one, :] /= np.sum(confidence[one, :])
            ntheta[one] = data['Ntheta']
            angleDiff[one, :] = data['LocalAngle'] - self.centerAngle
            dim[one, :] = data['Dimension']
            if self.mode == 'train':
                if self.idx + 1 < self.num_of_patch:
                    self.idx += 1
                else:
                    self.idx = 0
            else:
                if self.idx + 1 < self.Total:
                    self.idx += 1
                else:
                    self.idx = 35570
        return batch, confidence, confidence_multi, angleDiff, dim

    def EvalBatch(self):
        batch = np.zeros([1, 3, 224, 224], np.float) 
        info = self.info[self.idx]
        imgID = info['Index']
        if imgID != self.imgID:
            self.img = self.imgDataset.GetImage(imgID)
            self.imgID = imgID
        pt1 = info['Box_2D'][0]
        pt2 = info['Box_2D'][1]
        crop = self.img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        batch[0, 0, :, :] = crop[:, :, 2]
        batch[0, 1, :, :] = crop[:, :, 1]
        batch[0, 2, :, :] = crop[:, :, 0]

        if self.mode == 'train':
            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0
        else:
            if self.idx + 1 < self.Total:
                self.idx += 1
            else:
                self.idx = 35570
        return batch, self.centerAngle, info 
