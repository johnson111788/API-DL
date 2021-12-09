# -*- coding: utf-8 -*-
# @Time     : 2021/12/06
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : interpolate.py
# @Reference: https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
# @Reference: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9

import numpy as np


# input = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
# tensor([[[[1., 2., 3.],
#           [4., 5., 6.],
#           [7., 8., 9.]]]])

# image_upsample_tensor = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
# print(image_upsample_tensor)

# ###### mode ######

# bilinear
# tensor([[[[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.0000],
#           [1.7500, 2.0000, 2.5000, 3.0000, 3.5000, 3.7500],
#           [3.2500, 3.5000, 4.0000, 4.5000, 5.0000, 5.2500],
#           [4.7500, 5.0000, 5.5000, 6.0000, 6.5000, 6.7500],
#           [6.2500, 6.5000, 7.0000, 7.5000, 8.0000, 8.2500],
#           [7.0000, 7.2500, 7.7500, 8.2500, 8.7500, 9.0000]]]])

# bicubic
# tensor([[[[0.5781, 0.8750, 1.3516, 2.0156, 2.4922, 2.7891],
#           [1.4688, 1.7656, 2.2422, 2.9062, 3.3828, 3.6797],
#           [2.8984, 3.1953, 3.6719, 4.3359, 4.8125, 5.1094],
#           [4.8906, 5.1875, 5.6641, 6.3281, 6.8047, 7.1016],
#           [6.3203, 6.6172, 7.0938, 7.7578, 8.2344, 8.5312],
#           [7.2109, 7.5078, 7.9844, 8.6484, 9.1250, 9.4219]]]])


# ###### align_corners #####
#  True -> Pixels are regarded as a grid of points. Points at the corners are aligned.
#  像素被视为网格的格子上的点，拐角处的像素对齐。可知是点之间是等间距的。
#  False -> Pixels are regarded as 1x1 areas. Area boundaries, rather than their centers, are aligned.
#  像素被视为网格的交叉线上的点, 拐角处的点依然是原图像的拐角像素，但是差值的点间却按照上图的取法取，导致点与点之间是不等距的。

# bilinear, True
# tensor([[[[1.0000, 1.4000, 1.8000, 2.2000, 2.6000, 3.0000],
#           [2.2000, 2.6000, 3.0000, 3.4000, 3.8000, 4.2000],
#           [3.4000, 3.8000, 4.2000, 4.6000, 5.0000, 5.4000],
#           [4.6000, 5.0000, 5.4000, 5.8000, 6.2000, 6.6000],
#           [5.8000, 6.2000, 6.6000, 7.0000, 7.4000, 7.8000],
#           [7.0000, 7.4000, 7.8000, 8.2000, 8.6000, 9.0000]]]])

# bicubic, True
# tensor([[[[1.0000, 1.3160, 1.7280, 2.2720, 2.6840, 3.0000],
#           [1.9480, 2.2640, 2.6760, 3.2200, 3.6320, 3.9480],
#           [3.1840, 3.5000, 3.9120, 4.4560, 4.8680, 5.1840],
#           [4.8160, 5.1320, 5.5440, 6.0880, 6.5000, 6.8160],
#           [6.0520, 6.3680, 6.7800, 7.3240, 7.7360, 8.0520],
#           [7.0000, 7.3160, 7.7280, 8.2720, 8.6840, 9.0000]]]])


def bilinear_interpolate(im, x, y):

    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1

    # Make sure no exceeding
    x0 = np.clip(x0, 0, im.shape[1] - 1) if x1 <= im.shape[1] - 1 else x0 - 1
    x1 = np.clip(x1, 0, im.shape[1] - 1) if x1 <= im.shape[1] - 1 else x1 - 1
    y0 = np.clip(y0, 0, im.shape[0] - 1) if y1 <= im.shape[0] - 1 else y0 - 1
    y1 = np.clip(y1, 0, im.shape[0] - 1) if y1 <= im.shape[0] - 1 else y1 - 1

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    result = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return result


input = np.arange(1, 10, dtype=np.float32).reshape((3, 3))
print(input)

enlargedShape = list(map(int, [input.shape[0] * 2, input.shape[1] * 2]))
output = np.empty(enlargedShape, dtype=np.float32)

y_scale = float(input.shape[0] - 1) / float(output.shape[0] - 1)
x_scale = float(input.shape[1] - 1) / float(output.shape[1] - 1)

for y in range(output.shape[0]):
    for x in range(output.shape[1]):
        ori_x = x * x_scale
        ori_y = y * y_scale  # Find position in original image
        output[y, x] = bilinear_interpolate(input, ori_x, ori_y)

print(output)
