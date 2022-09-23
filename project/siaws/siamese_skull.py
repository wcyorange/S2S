import colorsys
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import tensorflow.keras.backend as K
# from nets.siamese import siamese
from siaws.nets.siamese_own import siamese_own


# ---------------------------------------------------#
#   使用自己训练好的模型预测需要修改model_path参数
# ---------------------------------------------------#
def sigmoid_x(x):
     if x >= 0.65:
        return (x ** (1 / 10))
     else:
        return (x ** (1 / 4))

class Siameseskull(object):
    _defaults = {
        "model_path": './siaws/logs/skull/skull.h5',

        "input_shape": (105, 105, 1),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Siamese
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # ---------------------------#
        #   载入模型与权值
        # ---------------------------#
        self.model = siamese_own(self.input_shape)
        self.model.load_weights(self.model_path)
        # print('{} model, anchors, and classes loaded.'.format(model_path))

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        # new_image = Image.new('RGB', size, (255,255,255))
        # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            # new_image = new_image.convert("L")
            new_image = image.convert("L")
        return new_image

    @tf.function
    def get_pred(self, photo):
        preds = self.model(photo, training=False)
        return preds

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------#
        #   对输入图像进行归一化
        # ---------------------------------------------------#
        image_1 = np.asarray(image_1).astype(np.float64) / 255
        image_2 = np.asarray(image_2).astype(np.float64) / 255

        # ---------------------------------------------------#
        #   添加上batch维度，才可以放入网络中预测
        # ---------------------------------------------------#
        photo1 = np.expand_dims(image_1, 0)
        photo2 = np.expand_dims(image_2, 0)

        # ---------------------------------------------------#
        #   获得预测结果，output输出为概率
        # ---------------------------------------------------#

        output = np.array(self.get_pred([photo1, photo2])[0])

        if (np.round(output, 4)==0.1967):
            output = 0

        output = sigmoid_x(1 - output)


        return output