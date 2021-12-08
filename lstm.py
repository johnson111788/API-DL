# -*- coding: utf-8 -*-
# @Time     : 2021/12/08
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : lstm.py
# @Reference: https://blog.varunajayasiri.com/numpy_lstm.html

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y * y


class LSTM(object):
    def __init__(self):
        self.W_f = 0
        self.b_f = 0

        self.W_i = 0
        self.b_i = 0

        self.W_C = 0
        self.b_C = 0

        self.W_o = 0
        self.b_o = 0

        # For final layer to predict the next character
        self.W_v = 0
        self.b_v = 0

    def forward(self, x, h_prev, C_prev, ):
        z = np.row_stack((h_prev, x))

        f = sigmoid(np.dot(self.W_f, z) + self.b_f)
        i = sigmoid(np.dot(self.W_i, z) + self.b_i)
        C_bar = tanh(np.dot(self.W_C, z) + self.b_C)

        c = f * C_prev + i * C_bar
        o = sigmoid(np.dot(self.W_o, z) + self.b_o)
        h = o * tanh(c)

        v = np.dot(self.W_v, h) + self.b_v

        return h, c, v