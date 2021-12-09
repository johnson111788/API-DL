# -*- coding: utf-8 -*-
# @Time     : 2021/12/09
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : dropout.py

import numpy as np


class Dropout(object):
    def __init__(self, drop_rate, is_test=False):
        self.drop_rate = drop_rate
        self.is_test = is_test

    def forward(self, x):
        if self.is_test:
            return x
        else:
            self.mask = np.random.uniform(0, 1, x.shape) > self.drop_rate
            return (x * self.mask) / (1 - self.drop_rate)  # keep expectation the same

    def backward(self, eta):
        return eta if self.is_test else eta * self.mask


dropout = Dropout(drop_rate=0.1)
for _ in range(10):
    x = np.random.rand(2, 2)
    x_out = dropout.forward(x)
    print(x_out)
