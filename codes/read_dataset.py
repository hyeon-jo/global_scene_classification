import cv2
import tensorflow as tf
import numpy as np
import os
import config
import random
from progress.bar import Bar
import sys

file_read_limit = config.FILE_READ_LIMIT
DATA_DISTRIBUTE_RATE = config.TRAIN_SET_RATE
split_size = config.FILE_READ_LIMIT // 10
train_size = int(split_size * DATA_DISTRIBUTE_RATE)

LABEL_MAP=config.LABEL_MAP

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

class DatasetReader():
    def __init__(self):
        self.train_cnt = 0
        self.test_cnt = 0
        self.cnt_temp = 0
        pass

    def convert_to_tfrecord(self, video_dir, tfrecord_dir):

        def make_tfrecord(same_labeled_video, label, i, isTest=False):
            random.shuffle(same_labeled_video)

            if isTest:
                dir = os.path.join(tfrecord_dir, 'test', path[-1])
            else:
                dir = os.path.join(tfrecord_dir, 'train', path[-1])

            if not os.path.exists(dir): os.makedirs(dir, exist_ok=True)
            #print('Write file: %s_%d.tfrecords' % (label, i // config.FILE_READ_LIMIT))

            writer = tf.python_io.TFRecordWriter(
                os.path.join(dir, '%s_%d.tfrecords' % (label, i // split_size)))

            for vid in same_labeled_video:
                video = np.array(vid)
                img_dir = os.path.join(dir, 'out')
                img_file = os.path.join(img_dir, str(self.cnt_temp) + '.avi')
                self.cnt_temp += 1
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir, exist_ok=True)
                video_writer = cv2.VideoWriter(img_file, fourcc, config.IMAGE_FRAMES, (224, 224))
                for frame in video:
                    video_writer.write(frame)
                bar.next()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[config.IMAGE_HEIGHT])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[config.IMAGE_WIDTH])),
                    'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[config.IMAGE_CHANNEL])),
                    'frames': tf.train.Feature(int64_list=tf.train.Int64List(value=[config.IMAGE_FRAMES])),
                    'video': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video.tostring()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[LABEL_MAP[path[-1]]]))
                }))
                writer.write(example.SerializeToString())
            writer.close()

        # valid_size = int(split_size * (config.TRAIN_SET_RATE + config.VALID_SET_RATE))

        for root, subdirs, files in os.walk(video_dir):
            random.shuffle(files)
            print('CWD: ' + root)
            bar = Bar('Processing', max=config.FILE_READ_LIMIT * 2, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

            path          = root.split('/')
            dataset       = []
            dataset_count = 0

            if path[-1] == 'tfrecord' or path[-1] == 'out':
                continue

            for file in files:
                if not file.endswith('.avi'):
                    continue
                video        = []
                file_name    = os.path.join(root, file)
                frame_count  = 0
                video_stream = cv2.VideoCapture(file_name)

                ret   = False
                frame = None

                while(True):
                    ret, frame = video_stream.read()

                    if ret == False:
                        break

                    frame = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
                    video.append(frame.copy())
                    frame_count += 1

                    if frame_count % config.IMAGE_FRAMES == 0:
                        dataset.append(video.copy())
                        video.clear()
                        dataset_count += 1
                        bar.next()

                        if dataset_count % split_size == 0:
                            make_tfrecord(dataset[0:train_size].copy(), path[-1], dataset_count)
                            make_tfrecord(dataset[train_size:split_size].copy(), path[-1], dataset_count, True)
                            self.train_cnt += split_size * DATA_DISTRIBUTE_RATE
                            self.test_cnt  += split_size * (1 - DATA_DISTRIBUTE_RATE)
                            dataset.clear()

                        if dataset_count == config.FILE_READ_LIMIT:
                            break

                video_stream.release()
                if dataset_count == config.FILE_READ_LIMIT:
                    break
            print(path[-1], ': ', dataset_count)
            bar.finish()

        print("Train clip: %d" % self.train_cnt + "\nTest clip: %d" % self.test_cnt)
        sys.exit(1)


    def read_dataset(self, data_path, test, batch_size=config.BATCH_SIZE):

        if not os.path.exists(os.path.join(data_path, 'tfrecord')):
            os.mkdir(os.path.join(data_path, 'tfrecord'))
            self.convert_to_tfrecord(data_path, data_path + '/tfrecord')
        feature = {'video': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        data_list = []
        if test == False:
            for root, _, files in os.walk(os.path.join(data_path, 'tfrecord', 'train')):
                for file in files:
                    if file.endswith('.tfrecords'):
                        data_list.append(os.path.join(root, file))
        else:
            for root, _, files in os.walk(os.path.join(data_path, 'tfrecord', 'test')):
                for file in files:
                    if file.endswith('.tfrecords'):
                        data_list.append(os.path.join(root, file))

        data_list.sort()
        filename_queue = tf.train.string_input_producer(data_list, num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)

        video = tf.decode_raw(features['video'], tf.uint8)
        video = tf.reshape(video, [config.IMAGE_FRAMES, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL])
        label = tf.cast(features['label'], tf.int32)

        if test == False:
            images, labels = tf.train.shuffle_batch([video, label], batch_size=batch_size, capacity=config.BATCH_SIZE*10, num_threads=4,
                                            min_after_dequeue=5)
        else:
            images, labels = tf.train.batch([video, label], batch_size=batch_size, capacity=config.BATCH_SIZE*10, num_threads=4)

        return images, labels
