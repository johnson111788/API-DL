# -*- coding: utf-8 -*-
# @Time     : 2021/12/08
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : conv3d.py

import numpy as np

channel_in = 3
h_in, w_in, t_in = 5, 5, 5
channel_out = 4

kernel_size = 2
padding = 1
stride = 3


def conv3d(input, kernel, stride):
    output_3d = np.zeros(shape=(h_out, w_out, t_out))

    # Notice upper bound
    for index_h_in, index_h_out in zip(range(0, h_in + 2 * padding - kernel_size + 1, stride), range(0, h_out)):
        for index_w_in, index_w_out in zip(range(0, w_in + 2 * padding - kernel_size + 1, stride), range(0, w_out)):
            for index_t_in, index_t_out in zip(range(0, t_in + 2 * padding - kernel_size + 1, stride), range(0, t_out)):
                window = input[index_h_in:index_h_in + kernel_size, index_w_in:index_w_in + kernel_size, index_t_in:index_t_in + kernel_size]
                output_3d[index_h_out, index_w_out] = np.sum(window * kernel)

    return output_3d


h_out = (h_in - kernel_size + 2 * padding) // stride + 1
w_out = (w_in - kernel_size + 2 * padding) // stride + 1
t_out = (t_in - kernel_size + 2 * padding) // stride + 1

input = np.random.rand(h_in, w_in, t_in, channel_in)
output = np.zeros(shape=(h_out, w_out, t_out, channel_out))

input_pad = np.zeros(shape=(h_in + 2 * padding, w_in + 2 * padding, t_in + 2 * padding, channel_in))
if padding != 0:
    input_pad[padding:-padding, padding:-padding, padding:-padding] = input
else:
    input_pad = input

kernel = np.random.rand(channel_out, kernel_size, kernel_size, kernel_size, channel_in)
bias = np.random.rand(channel_out)  # one dim bias

for index_channel_out in range(channel_out):  # out loop - in loop
    for index_channel_in in range(channel_in):
        kernel_slice = kernel[index_channel_out, :, :, :, index_channel_in]
        input_3d = input_pad[:, :, :, index_channel_in]  # input_pad
        output[:, :, :, index_channel_out] += conv3d(input_3d, kernel_slice, stride)

    output[:, :, :, index_channel_out] += bias[index_channel_out]

print('Input shape: ', input.shape)
print('Output shape: ', output.shape)