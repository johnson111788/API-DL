# -*- coding: utf-8 -*-
# @Time     : 2021/12/05
# @Author   : Johnson-Chou
# @Email    : johnson111788@gmail.com
# @FileName : conv2d.py

import numpy as np


def conv2D(input_2Ddata, kern, in_size, out_size, stride_h=1, stride_w=1):
    kernel_size_h, kernel_size_w = kern.shape
    (h_in, w_in) = in_size
    (h_out, w_out) = out_size
    out_2Ddata = np.zeros(shape=out_size)

    for index_h_out, index_h_in in zip(range(h_out), range(0, h_in + 2 * padding_size - kernel_size_h + 1, stride_h)):
        for index_w_out, index_w_in in zip(range(w_out), range(0, w_in + 2 * padding_size - kernel_size_w + 1, stride_w)):
            window = input_2Ddata[index_h_in:index_h_in + kernel_size_h, index_w_in:index_w_in + kernel_size_w]
            out_2Ddata[index_h_out, index_w_out] = np.sum(kern * window)
    return out_2Ddata


h_in = 5
w_in = 5
channel_in = 4

input_3Ddata = np.random.randn(h_in, w_in, channel_in)

stride = 2
kernel_size = 2
channel_out = 4
padding_size = 1

h_out = (h_in - kernel_size + 2 * padding_size) // stride + 1
w_out = (w_in - kernel_size + 2 * padding_size) // stride + 1

padding = np.zeros(shape=(h_in + 2 * padding_size, w_in + 2 * padding_size, channel_in))
padding[padding_size:-padding_size, padding_size:-padding_size] = input_3Ddata

output_3Ddata = np.zeros(shape=(h_out, w_out, channel_out))

kernel = np.random.randn(channel_out, kernel_size, kernel_size, channel_in)
bias = np.random.randn(channel_out)

for ch_out in range(channel_out):
    for ch_in in range(channel_in):
        input_2Ddata = padding[:, :, ch_in]
        kern = kernel[ch_out, :, :, ch_in]
        output_3Ddata[:, :, ch_out] += conv2D(input_2Ddata, kern, (h_in, w_in), out_size=(h_out, w_out),
                                              stride_h=stride,
                                              stride_w=stride)
    output_3Ddata[:, :, ch_out] += bias[ch_out]
