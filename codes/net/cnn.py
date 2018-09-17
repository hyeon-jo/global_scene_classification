# -*- coding: utf-8 -*-
from common import BaseModel
import numpy as np
import tensorflow as tf

IMAGE_FRAME   =  30
IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
IMAGE_CHANNEL =   3
CLASS_NUM     =   8

"""
****************************************************************************************************************************
"""
def custom_fc_layers(input, out_dim, reuse):
    with tf.device('/cpu:0'):
        shape = int(np.prod(input.get_shape()[1:]))
        fc1w = tf.get_variable(initializer=tf.truncated_normal([shape, out_dim],
                                                               dtype=tf.float32,
                                                               stddev=1e-1), name='custom_fc_weights')

    with tf.variable_scope('custom_fc', reuse=reuse) as scope:
        input_flat = tf.reshape(input, [-1, shape])
        custom_fc = tf.matmul(input_flat, fc1w)

    return custom_fc

class RCNN_baseline(BaseModel):
    def __init__(self, depth = 6, init_channels = 16, input_frames = IMAGE_FRAME,
                 input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH, input_channels=IMAGE_CHANNEL , hiddenDim=4096):     # 512 for RCNN_Baseline_0527
        self.depth = depth
        self.input_frames = input_frames
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.vectorSize = input_height*input_width*input_frames
        self.hiddenDim = hiddenDim

    def build_RNN(self, x, reuse, num_units=2048):

        cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse)
        cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse)
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, x, dtype=tf.float32)


        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]

        return output

    def build_graph(self, input, expected, reuse, vgg16):


        input = tf.reshape(input,[-1,self.input_height,self.input_width,self.input_channels])

        # VGG-16 without BN & FC
        vgg = vgg16

        with tf.variable_scope('vgg16_out1024', reuse=reuse):
            vgg.imgs = input
            network = vgg.conv_old()

        with tf.variable_scope('reshape', reuse=reuse):

            network = tf.reshape(network, [-1, self.input_frames, 4096])

        with tf.variable_scope('mainRNN', reuse=reuse):
            network= self.build_RNN(network,reuse)
            network = custom_fc_layers(input=network, out_dim=CLASS_NUM, reuse=reuse)

        with tf.variable_scope("mainRegression", reuse=reuse):
            output = tf.nn.softmax_cross_entropy_with_logits(labels=expected, logits=network)

        cost = tf.reduce_mean(output)

        return dict(prediction=network, loss=cost, images=None, attention =None)


class ResNetBaseline(RCNN_baseline):
    def __init__(self, depth = 6, init_channels = 16, input_frames = IMAGE_FRAME,
                 input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH, input_channels=IMAGE_CHANNEL , hiddenDim=2048):     # hiddenDim=512 for Resnet_0601
        super(RCNN_baseline, self).__init__()
        self.depth = depth
        self.input_frames = input_frames
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.vectorSize = input_height * input_width * input_frames
        self.hiddenDim = hiddenDim

    def build_graph(self, input, expected, reuse, resnet):
        input = tf.reshape(input, [-1, self.input_height, self.input_width, self.input_channels])

        _, network = resnet.resnet_v2_50(inputs=input, reuse=reuse, is_training=True)                    # num_classes=512 for Resnet_0601
        with tf.variable_scope('reshape', reuse=reuse):
            network = tf.reshape(network['global_pool'], [-1, self.input_frames, 2048])#tf.reshape(network[1]['global_pool'], [-1, self.input_frames, 2048])                               #[-1, self.input_frames, 512] for Resnet_0601

        with tf.variable_scope('mainRNN', reuse=reuse):
            network = self.ConvGRU(network, reuse)
            network = custom_fc_layers(network, CLASS_NUM, reuse)

        with tf.variable_scope("mainRegression", reuse=reuse):
            softmax = tf.nn.softmax(logits=network)
            output = tf.nn.softmax_cross_entropy_with_logits(labels=expected, logits=network)#(logits=network)#_cross_entropy_with_logits(labels=expected, logits=network)

        cost = tf.reduce_mean(output)

        return dict(prediction=network, loss=cost, softmax=softmax)

    def SliceDB(self, x, sliceFrameNum):

        x = tf.slice(x, [0, self.input_frames - sliceFrameNum, 0], [-1, sliceFrameNum, -1])
        return x

    def RLC(self, filter_name, x, hiddenDim, activation, reuse, stride=1, filterSize=3, channels=1, padding='None'):
        # x : input rnn vector ( maybe [batch,timeStep,vectorDim)
        # stride : filter stride
        # nb_filter : num of channels
        # padding ; 'None' or ZeroPadding'

        timeStep = x.get_shape()[1]

        if (timeStep < filterSize):
            print("initLen > filterSize")

        # timeStep Length

        if (((timeStep - filterSize) % (stride) != 0 and padding == 'None') or
                ((timeStep) % (stride) != 0 and padding == 'ZeroPadding')):
            print("Please Fit Timestep Size : Required TimeStep(filterSize+stride*(k-1) //k=nexeTimeStep\n"
                  + "Please Fit Timestep Size(ZeroPadding) : Required TimeStep(filterSize+stride*(k) //k=nexeTimeStep")

        with tf.variable_scope(filter_name, reuse=reuse):
            cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hiddenDim, state_is_tuple=True,
                                                  activation=activation,
                                                  reuse=reuse)
            cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hiddenDim, state_is_tuple=True,
                                                  activation=activation,
                                                  reuse=reuse)

        if padding == 'None':
            nextTimeStepNum = (timeStep - filterSize) / stride + 1
            for i in range(nextTimeStepNum):
                print(
                x.get_shape())
                sliceVector = tf.slice(x, [0, i * stride, 0], [-1, filterSize, -1])
                if (i == 0):

                    self.nextLayer = self.localRNN(sliceVector, True, cell_1, cell_2, filter_name)
                else:
                    self.nextLayer = tf.concat([self.nextLayer, self.localRNN(sliceVector, True, cell_1, cell_2,
                                                                              filter_name)], 1)

            x = tf.reshape(self.nextLayer, [-1, int(nextTimeStepNum), hiddenDim * 2])

        elif (padding == 'ZeroPadding'):

            startX = -(filterSize - 1) / 2
            endX = startX + filterSize
            nextTimeStepNum = (timeStep) / stride

            for i in range(nextTimeStepNum):

                if (startX < 0):
                    sliceSt = 0
                    sliceNum = filterSize + startX
                elif (endX >= timeStep):
                    sliceSt = startX
                    sliceNum = timeStep - sliceSt
                else:
                    sliceSt = startX
                    sliceNum = filterSize

                # sliceSt= int(sliceSt)
                sliceNum = int(sliceNum)

                print(
                "startX : %d endX : %d sliceNum : %d" % (startX, endX, sliceNum))
                sliceVector = tf.slice(x, [0, sliceSt, 0], [-1, sliceNum, -1])

                if (i == 0):

                    self.nextLayer = self.localRNN(sliceVector, True, cell_1, cell_2, filter_name)
                else:
                    self.nextLayer = tf.concat([self.nextLayer, self.localRNN(sliceVector, True, cell_1, cell_2,
                                                                              filter_name)], 1)
                startX += stride
                endX += stride

            x = tf.reshape(self.nextLayer, [-1, int(nextTimeStepNum), hiddenDim * 2])
        return x

    def ConvGRU_MakaBaseLine(self, x, reuse):

        input = self.SliceDB(x, 15)

        # time step =15
        input = self.RLC('depth_1', input, 128, tf.tanh, reuse, stride=1, filterSize=3, channels=1,
                         padding='ZeroPadding')
        input = self.RLC('depth_1_pooling', input, 128, tf.tanh, reuse, stride=2, filterSize=3, channels=1,
                         padding='None')
        # actually it is stride pooling

        # time step =7
        input = self.RLC('depth_2', input, 128, tf.tanh, reuse, stride=1, filterSize=3, channels=1,
                         padding='ZeroPadding')
        input = self.RLC('depth_2_pooling', input, 128, tf.tanh, reuse, stride=2, filterSize=3, channels=1,
                         padding='None')

        # time step=3
        input = self.RLC('depth_3', input, 128, tf.tanh, reuse, stride=1, filterSize=3, channels=1,
                         padding='ZeroPadding')
        input = self.RLC('depth_3_pooling', input, 128, tf.tanh, reuse, stride=2, filterSize=3, channels=1,
                         padding='None')

        return tf.reshape(input, [-1, int(x.get_shape()[2])])

    def ConvGRU(self, x, reuse):
        print('init value')
        print(
        x.get_shape())
        filterSize = 3
        stride = 2

        # curBatchSize = int(x.get_shape()[0])
        vectorDim = int(x.get_shape()[2])
        initLen = int(x.get_shape()[1])

        ########CAUTION Only for TimeStep-20###############
        if (initLen < filterSize or (initLen - filterSize) % 2 != 0):
            ############CAUTION
            x = tf.slice(x, [0, 5, 0], [-1, 15, -1])
            initLen = int(x.get_shape()[1])

            print(
            'caution vector')
            print(
            x.get_shape())

            ############CAUTION

            # print 'impossible timestep with filterSize : %d and stride : %d'%(filterSize,stride)
            # exit()

        while (True):
            if (initLen < filterSize):
                break

            print(
            'nextStep')
            forNum = (initLen - filterSize) // stride + 1
            with tf.variable_scope("gru_layer2_" + str(initLen), reuse=reuse):

                cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True,
                                                      activation=tf.tanh,
                                                      reuse=reuse)

                cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True,
                                                      activation=tf.tanh,
                                                      reuse=reuse)
            print(
            'fornum = %d' % (forNum))
            for i in range(forNum):
                print(
                x.get_shape())
                sliceVector = tf.slice(x, [0, i * stride, 0], [-1, filterSize, -1])
                print(
                '--' + str(i))
                print(
                sliceVector.get_shape())

                if (i == 0):
                    self.nextLayer = self.localRNN(sliceVector, True, cell_1, cell_2, "gru_layer2_" + str(initLen))
                else:
                    self.nextLayer = tf.concat([self.nextLayer, self.localRNN(sliceVector, True, cell_1, cell_2,
                                                                              "gru_layer2_" + str(initLen))], 1)

            ###concat
            vectorDim = self.hiddenDim * 2
            x = tf.reshape(self.nextLayer, [-1, forNum, vectorDim])

            initLen = int(initLen / stride)

        afterDim = int(x.get_shape()[2])
        return tf.reshape(x, [-1, forNum * afterDim])

    def localRNN(self, x, reuse, cell_1, cell_2, _scope):

        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, x, dtype=tf.float32, scope=_scope)

        output = tf.concat([output[0], output[1]], axis=2)
        output = output[:, -1]

        # self attention or using last gruCell Output

        return output
