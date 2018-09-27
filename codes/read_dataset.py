import cv2
import tensorflow as tf
import numpy as np
import os
import config
import random
from progress.bar import Bar
import sys

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

class DatasetReader():
    def __init__(self, list_file_path):
        self.trainset  = []
        self.testset   = []

        self.make_file_list(list_file_path)

        self.train_batch_pointer = 0
        self.test_batch_pointer  = 0

    def make_file_list(self, list_file_path):
        train_data  = []
        test_data   = []
        train_list  = []
        test_list   = []

        for li in os.listdir(list_file_path):
            if li.startswith('train'):
                train_list.append(os.path.join(list_file_path, li))
            elif li.startswith('test'):
                test_list.append(os.path.join(list_file_path, li))
            else:
                idx_stream = open(os.path.join(list_file_path, li), 'r')
                while(True):
                    class_idx = idx_stream.readline()
                    if not class_idx:
                        break

                    class_idx = class_idx.split(' ')
                    config.CLASS_IDX[class_idx[0]] = class_idx[1][:-1]
                    config.REV_IDX[class_idx[1][:-1]] = class_idx[0]

        for e in train_list:
            file_stream = open(e, 'r')
            while(True):
                video_file = file_stream.readline()
                if not video_file:
                    break
                video_file = video_file.split(' ')
                train_data.append([os.path.abspath(os.path.join(list_file_path, '..', 'data', video_file[0])),
                                   int(video_file[1][:-1])])
            file_stream.close()

        for e in test_list:
            file_stream = open(e, 'r')
            while(True):
                video_file = file_stream.readline()
                if not video_file:
                    break
                test_data.append([os.path.abspath(os.path.join(list_file_path, '..', 'data', video_file[:-1])),
                                  int(config.REV_IDX[video_file.split('/')[0]])])
            file_stream.close()

        print(train_data[0])
        print(test_data[0])
        self.trainset = train_data
        self.testset  = test_data


    def next_batch(self, is_train=True):
        if is_train and self.train_batch_pointer == 0:
            random.shuffle(self.trainset)

        data  = []
        labels = []
        for i in range(config.BATCH_SIZE):
            data_loc, label = self.trainset[self.train_batch_pointer] if is_train else self.testset[self.test_batch_pointer]

            video_stream = cv2.VideoCapture(data_loc)
            video = []
            while(True):
                ret, frame = video_stream.read()
                if ret == False:
                    break

                video.append(frame)

            choice = np.random.choice(len(video), config.IMAGE_FRAMES)
            datum = []
            for num in choice:
                datum.append(video[num].copy())
            data.append(datum)
            labels.append(label)
            if is_train:
                self.train_batch_pointer += 1
                if self.train_batch_pointer >= len(self.trainset):
                    self.train_batch_pointer = 0
            else:
                self.test_batch_pointer += 1
                if self.test_batch_pointer >= len(self.testset):
                    self.test_batch_pointer = 0

        if np.shape(data) != (8, 20, 240, 320, 3):
            print('[!] BATCH SIZE ERROR: ', np.shape(data))
            print(np.shape(video))
            with open('log.txt', 'w') as stream:
                for e in data:
                    for ee in e:
                        stream.write(ee+'\n')

        video.clear()
        return data, labels