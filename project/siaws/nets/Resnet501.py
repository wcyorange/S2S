from __future__ import absolute_import
from __future__ import print_function
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, Flatten

im_width = 105
im_height = 105

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    # x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def bottleneck_Block(inpt, nb_filters, strides=(1, 1), with_conv_shortcut=False):
    k1, k2, k3 = nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_50():
    width = im_width
    height = im_height
    channel = 1
    inpt = Input(shape=(width, height, channel))
    # x = ZeroPadding2D((3, 3))(inpt)
    x = inpt
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = bottleneck_Block(x, nb_filters=[64, 64, 256], strides=(1, 1), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64, 64, 256])
    x = bottleneck_Block(x, nb_filters=[64, 64, 256])

    # conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    # conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    # conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    # self.block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    return Model(inpt, x)