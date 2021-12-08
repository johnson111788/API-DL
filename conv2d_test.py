# -*- coding: utf-8 -*-
# @Time     : 2021/12/05
# @Author   : Johnson-Chou
# @Email    : johnson111788@gmail.com
# @FileName : conv2d_test.py

import numpy as np

channel_in = 3
h_in, w_in = 5, 5
channel_out = 4

kernel_size = 2
padding = 1
stride = 3


def conv2d(input, kernel, stride):
    output_2d = np.zeros(shape=(h_out, w_out))

    # Notice upper bound
    for index_h_in, index_h_out in zip(range(0, h_in + 2 * padding - kernel_size + 1, stride), range(0, h_out)):
        for index_w_in, index_w_out in zip(range(0, w_in + 2 * padding - kernel_size + 1, stride), range(0, w_out)):
            window = input[index_h_in:index_h_in + kernel_size, index_w_in:index_w_in + kernel_size]
            output_2d[index_h_out, index_w_out] = np.sum(window * kernel)

    return output_2d


h_out = (h_in - kernel_size + 2 * padding) // stride + 1
w_out = (w_in - kernel_size + 2 * padding) // stride + 1

input = np.random.rand(h_in, w_in, channel_in)
output = np.zeros(shape=(h_out, w_out, channel_out))

input_pad = np.zeros(shape=(h_in + 2 * padding, w_in + 2 * padding, channel_in))
if padding != 0:
    input_pad[padding:-padding, padding:-padding] = input
else:
    input_pad = input

kernel = np.random.rand(channel_out, kernel_size, kernel_size, channel_in)
bias = np.random.rand(channel_out)  # one dim bias

for index_channel_out in range(channel_out):  # out loop - in loop
    for index_channel_in in range(channel_in):
        kernel_slice = kernel[index_channel_out, :, :, index_channel_in]
        input_2d = input_pad[:, :, index_channel_in]  # input_pad
        output[:, :, index_channel_out] += conv2d(input_2d, kernel_slice, stride)

    output[:, :, index_channel_out] += bias[index_channel_out]

print(output.shape)
