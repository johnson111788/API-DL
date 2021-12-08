# -*- coding: utf-8 -*-
# @Time     : 2021/12/08
# @Author   : Johnson-Chou
# @Email    : johnson111788@gmail.com
# @FileName : backpropagation.py

import numpy as np


def maxpooling(data, kernel_size):
    a, b = data.shape
    img_new = []
    for i in range(0, a, kernel_size):
        line = []
        for j in range(0, b, kernel_size):
            x = data[i:i + kernel_size, j:j + kernel_size]
            line.append(np.max(x))
        img_new.append(line)
    return np.array(img_new)


input = np.arange(1, 145, dtype=np.float32).reshape((12, 12))
print(input)
output = maxpooling(input, 3)
print(output)