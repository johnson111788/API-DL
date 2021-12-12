# -*- coding: utf-8 -*-
# @Time     : 2021/12/09
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : batchnormalization.py
# @Reference: https://blog.csdn.net/liuxiao214/article/details/81037416

import numpy as np


def batchnorm(x, gamma, beta):
    # x_shape:[B, C, H, W]
    eps = 1e-5

    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(0, 2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta

    return results


def layernorm(x, gamma, beta):
    eps = 1e-5

    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results


def instancenorm(x, gamma, beta):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results


def groupnorm(x, gamma, beta, G=16):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5
    x = np.reshape(x, (x.shape[0], G, x.shape[1] / 16, x.shape[2], x.shape[3]))

    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results


# x_shape:[B, C, H, W]
input = np.random.rand(2, 1, 3, 3)
print('Input: \n', input)
output = batchnorm(input, gamma=1, beta=0)
print('Output: \n', output)
