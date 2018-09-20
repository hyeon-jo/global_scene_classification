# -*- coding: utf-8 -*-
from common import BaseModel
import numpy as np
import tensorflow as tf
import tflearn

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

    def build_RNN(self, x, reuse):

        cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
                                              reuse=reuse)
        cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenDim, state_is_tuple=True, activation=tf.tanh,
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

    def frame_weights(self, network):
        network = tflearn.conv_2d(network, self.input_frames, filter_size=[1, 1],
                                 strides=1, padding='SAME', activation='linear', bias=False, weights_init='xavier')
        return network

    def build_graph(self, input, expected, reuse, resnet):
        input = tf.reshape(input, [-1, self.input_height, self.input_width, self.input_channels])

        _, network = resnet.resnet_v2_50(inputs=input, reuse=reuse, is_training=True)                    # num_classes=512 for Resnet_0601
        with tf.variable_scope('reshape', reuse=reuse):
            network = tf.reshape(network['global_pool'], [-1, self.input_frames, 2048])#tf.reshape(network[1]['global_pool'], [-1, self.input_frames, 2048])                               #[-1, self.input_frames, 512] for Resnet_0601

        with tf.variable_scope('frameWeights', reuse=reuse):
            network = self.frame_weights(network)

        with tf.variable_scope('mainRNN', reuse=reuse):
            network = self.build_RNN(network, reuse)
            network = custom_fc_layers(network, CLASS_NUM, reuse)

        with tf.variable_scope("mainRegression", reuse=reuse):
            softmax = tf.nn.softmax(logits=network)
            output = tf.nn.softmax_cross_entropy_with_logits(labels=expected, logits=network)#(logits=network)#_cross_entropy_with_logits(labels=expected, logits=network)

        cost = tf.reduce_mean(output)

        return dict(prediction=network, loss=cost, softmax=softmax)