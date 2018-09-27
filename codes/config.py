
import tensorflow as tf
import net.cnn as cnn
from net.resnet50 import resnet

NUM_GPUS = int(4)
DATA_FOLD_PATH = None

EVAL_SAVE_FOLDER = './results'

NETWORK = resnet()
MODEL_NETWORK = cnn.ResNetBaseline()

BATCH_SIZE = int(8)
IMAGE_WIDTH = int(320)          # need to change the values in cnn.py
IMAGE_HEIGHT = int(240)         # need to change the values in cnn.py
IMAGE_CHANNEL = int(3)          # need to change the values in cnn.py
IMAGE_FRAMES = int(20)
NUM_OF_CLASS = int(101)
NUM_OF_CLIP  = int(28747)

EPOCH = int(NUM_OF_CLIP / BATCH_SIZE)

MODEL_SAVE_FOLDER = '/disk1/hyeon/model_save/'
MODEL_SAVE_NAME = 'stacked_0926'#'ResNet_0720_scratch'#'ResNet_0609_scratch'#RCNN_0609_scratch'#'Resnet_0601'
MODEL_SAVE_INTERVAL = int(EPOCH * 5)

# ResNet_0609_scratch : BATCH_SIZE=8, FRAMES=60, NUM_OF_CLASS=7
# ResNet_0720_scratch : BATCH_SIZE=32, FRAMES=15, NUM_OF_CLASS=8
# ResNet_0821_30frames : BATCH_size=16, FRAMES=30, NUM_OF_CLASS=8

CLASS_IDX = {}
REV_IDX   = {}

TRAIN_DATA_SHUFFLE = True
TRAIN_MAX_STEPS = int(EPOCH * 100 + 1)
TRAIN_LEARNING_RATE = float(0.01)
TRAIN_DECAY = float(0.1)
TRAIN_WEIGHT_DECAY = float(0.05)
TRAIN_DECAY_INTERVAL = float(EPOCH * 30)

TRAIN_OVERWRITE = False
TEST_MODE = False


# outdoor:  6141
# panning:  3737
# pattern:  2840
# sports:  5664
# animation:  7685
# complex:  5260
# ballsports:  14070
# tfrecord:  0
# graphic:  1708

# batch:32, frame:15 -> 70.xx
# batch:16, frame:30 -> 71.25
# batch:16, frame:30, FILE_READ_LIMIT 1500(graphic 770) -> 77.82%
# batch:16, frame:30, FILE_READ_LIMIT 1500
