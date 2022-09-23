from PIL import Image
import json
import matplotlib.pyplot as plt # plt 用于显示图片
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# import pcl.pcl_visualization

f = open("config/openImage1.json")
img1_path = json.load(f)
# print(img1_path['open_fold'])
img = Image.open(img1_path['open_fold'])
# 结果展示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
plt.subplot(121)
plt.imshow(img)
plt.title('图像1')
# 不显示坐标轴
plt.axis('off')



# 显示图像
plt.show()

# def join(png1, png2, flag='horizontal'):
#     """
#     :param png1: path
#     :param png2: path
#     :param flag: horizontal or vertical
#     :return:
#     """
#     img1, img2 = Image.open(png1), Image.open(png2)
#     # 统一图片尺寸，可以自定义设置（宽，高）
#     img1 = img1.resize((256, 256), Image.ANTIALIAS)
#     img2 = img2.resize((256, 256), Image.ANTIALIAS)
#     size1, size2 = img1.size, img2.size
#     if flag == 'horizontal':
#         joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
#         loc1, loc2 = (0, 0), (size1[0], 0)
#         joint.paste(img1, loc1)
#         joint.paste(img2, loc2)
#         joint.save('horizontal.png')
#     elif flag == 'vertical':
#         joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
#         loc1, loc2 = (0, 0), (0, size1[1])
#         joint.paste(img1, loc1)
#         joint.paste(img2, loc2)
#         joint.save('vertical.png')
#
#
# if __name__ == '__main__':
#     # 两张图片地址：
#     png1 = r"E:\pycharmnote\python\jiemian\self/skin.png"
#     png2 = r"E:\pycharmnote\python\jiemian\self/skull.png"
#     # 横向拼接
#     join(png1, png2, flag='horizontal')
#
#     # 纵向拼接
#     # join(png1, png2, flag='vertical')