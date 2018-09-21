
import tensorflow as tf
import net.cnn as cnn
from net.resnet50 import resnet

NUM_GPUS = int(4)
DATA_FOLD_PATH = None

EVAL_SAVE_FOLDER = './results'

NETWORK = resnet()
MODEL_NETWORK = cnn.ResNetBaseline()

TRAIN_SET_RATE = 0.8
TEST_SET_RATE  = 0.2

BATCH_SIZE = int(16)
IMAGE_WIDTH = int(224)          # need to change the values in cnn.py
IMAGE_HEIGHT = int(224)         # need to change the values in cnn.py
IMAGE_CHANNEL = int(3)          # need to change the values in cnn.py
IMAGE_FRAMES = int(30)
FILE_READ_LIMIT = int(1500)
NUM_OF_CLASS = int(8)

EPOCH = int((FILE_READ_LIMIT * TRAIN_SET_RATE) / BATCH_SIZE * NUM_OF_CLASS)

MODEL_SAVE_FOLDER = '/media/sdc1/hyeon/model_save/'
MODEL_SAVE_NAME = 'ResNet_0921_30frames'#'ResNet_0720_scratch'#'ResNet_0609_scratch'#RCNN_0609_scratch'#'Resnet_0601'
MODEL_SAVE_INTERVAL = int(EPOCH * 5)#575)#350)#560)                            # 7 class train: 4595, test: 1005
TRAINABLE = False

LABEL_MAP={'sports':0, 'animation':1, 'ballsports':2, 'complex':3, 'outdoor':4, 'pattern':5, 'graphic':6, 'panning':7}
REV_MAP  ={0:'sports', 1:'animation', 2:'ballsports', 3:'complex', 4:'outdoor', 5:'pattern', 6:'graphic', 7:'panning'}

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
