import colorsys
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from PIL import Image, ImageDraw, ImageFont
# from tensorflow.compat.v1.keras import backend as K
from sia.siamese import siamese
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#---------------------------------------------------#
#   使用自己训练好的模型预测需要修改model_path参数
#---------------------------------------------------#
class Siamesewcy(object):
    _defaults = {
        "model_path"    : './sia/model_data/ep017-loss0.000-val_loss0.621.h5',
        "input_shape"   : (256, 256, 1),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Siamese
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.generate()
        
    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        #---------------------------#
        #   载入模型与权值
        #---------------------------#
        self.model = siamese(self.input_shape)
        self.model.load_weights(self.model_path)
        # print('{} model, anchors, and classes loaded.'.format(model_path))
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (255,255,255))
        # new_image = Image.new('RGB', size, (255, 255, 255))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        #---------------------------------------------------#
        #   对输入图像进行不失真的resize
        #---------------------------------------------------#
        image_1 = self.letterbox_image(image_1,[self.input_shape[1],self.input_shape[0]])
        image_2 = self.letterbox_image(image_2,[self.input_shape[1],self.input_shape[0]])
        
        #---------------------------------------------------#
        #   对输入图像进行归一化
        #---------------------------------------------------#
        image_1 = np.asarray(image_1).astype(np.float64)/255
        image_2 = np.asarray(image_2).astype(np.float64)/255
        
        if self.input_shape[-1]==1:
            image_1 = np.expand_dims(image_1, -1)
            image_2 = np.expand_dims(image_2, -1)
            
        #---------------------------------------------------#
        #   添加上batch维度，才可以放入网络中预测
        #---------------------------------------------------#
        photo1 = np.expand_dims(image_1,0)
        photo2 = np.expand_dims(image_2,0)

        #---------------------------------------------------#
        #   获得预测结果，output输出为概率
        #---------------------------------------------------#
        output = self.model.predict([photo1,photo2])[0]
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(np.squeeze(image_1)))
        # # plt.imshow(np.array(image_1))
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(np.squeeze(image_2)))
        # # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va='bottom', fontsize=11)
        # plt.show()
        return output

    def close_session(self):
        self.sess.close()
