import os

import numpy as np
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.models import Model

from siaws.nets.Resnet501 import resnet_50

# from nets.Resnet50 import ResNet50
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
#-------------------------#
#   创建孪生神经网络
#-------------------------#
def siamese_own(input_shape):
    res50_model = resnet_50()

    # ResNet50_model = ResNet50()

    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    #------------------------------------------#
    #   我们将两个输入传入到主干特征提取网络
    #------------------------------------------#
    encoded_image_1 = res50_model(input_image_1)
    encoded_image_2 = res50_model(input_image_2)

    # encoded_image_1 = ResNet50_model[input_image_1]
    # encoded_image_2 = ResNet50_model[input_image_2]

    #-------------------------#
    #   相减取绝对值
    #-------------------------#
#50 :L1,一个Dence
#501:L2,两个Dence
    distance = Lambda(euclidean_distance,
                       output_shape=eucl_dist_output_shape)([encoded_image_1, encoded_image_2])

    # l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_image_1, encoded_image_2])

    #-------------------------#
    #   进行两次全连接
    #-------------------------#
    # x = Dense(512,activation='relu')(l1_distance)
    # x = Dense(512,activation='relu')(distance)
    #---------------------------------------------#
    #   利用sigmoid函数将最后的值固定在0-1之间。
    #---------------------------------------------#
    out = Dense(1,activation='sigmoid')(distance)

    model = Model([input_image_1, input_image_2], out)
    # model.summary()
    return model









