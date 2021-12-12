# -*- coding: utf-8 -*-
# @Time     : 2021/12/07
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : backpropagation.py

import numpy as np


class Loss(object):

    def squared_loss(self, pred, truth):
        return (pred - truth) ** 2

    def squared_loss_grad(self, pred, truth):
        return 2 * (pred - truth)  # dloss/dy_pred

    def bce_loss(self, pred, truth):
        return -np.log(pred) * truth

    def bce_loss_grad(self, pred, truth):
        return truth*(1/pred*np.log(10))


class Op1(object):
    '''
    y = x^2
    '''

    def __init__(self):
        self.param = {'x': 1}

    def forward(self):
        return self.param['x'] * self.param['x']

    def backward(self, grad):
        # dy/dx=2*x
        return grad * (2 * self.param['x'])

    def update(self, g, lr):
        self.param['x'] -= lr * g


class Op2(object):
    '''
    y = (a+b)*c
    '''

    def __init__(self):
        self.param = {'a': 1, 'b': 1, 'c': 1}

    def forward(self):
        return (self.param['a'] + self.param['b']) * self.param['c']

    def backward(self, grad):
        # q = a+b dq/da=1 dq/db=1
        # y = q*c dy/dq=c dy/dc=q
        # dy/da = dy/dq * dq/da
        # dy/db = dy/dq * dq/db
        # dy/dc = q
        grad_a = grad * self.param['c'] * 1  # dL/dy * dy/dq * dq/da
        grad_b = grad * self.param['c'] * 1  # dL/dy * dy/dq * dq/db
        grad_c = grad * (self.param['a'] + self.param['b'])  # dL/dy * dy/dc
        return grad_a, grad_b, grad_c

    def update(self, grad, lr=0.01):
        grad_a, grad_b, grad_c = grad
        self.param['a'] -= lr * grad_a
        self.param['b'] -= lr * grad_b
        self.param['c'] -= lr * grad_c

        return


class Op3(object):
    '''
    y = sigmoid(x0*w0+x1*w1+w2)=1/(1+e^-(x0*w0+x1*w1+w2))
    '''

    def __init__(self):
        self.input = {'x0': 1, 'x1': 1}
        self.param = {'w0': 1, 'w1': 1, 'w2': 1}

    def forward(self):
        return 1 / (1 + 2.71828 ** -(
                self.input['x0'] * self.param['w0'] + self.input['x1'] * self.param['w1'] + self.param['w2']))

    def backward(self, grad):
        # Sigmoid: @(x)=1/(1+e^(x)), d@(x)/dx = (1-@(x))@(x)
        # dy/dw0 dy/dw1 dy/dw2
        grad_mid = (1 - self.forward()) * self.forward()
        grad_w2 = grad * grad_mid
        grad_w1 = grad * grad_mid * self.input['x1']
        grad_w0 = grad * grad_mid * self.input['x0']

        return grad_w0, grad_w1, grad_w2

    def update(self, grad, lr=0.01):
        grad_w0, grad_w1, grad_w2 = grad
        self.param['w0'] -= lr * grad_w0
        self.param['w1'] -= lr * grad_w1
        self.param['w2'] -= lr * grad_w2
        return


class Op4(object):
    '''
    y = L2(matrix_w * matrix_x)
    '''
    def __init__(self):
        self.param = [[0.1, 0.5], [-0.3, 0.8]]
        self.x = [0.2, 0.4]

    def forward(self):
        # (q0)[w00*x0 + w01*x1] (q1)[w10*x0 + w11*x1]
        # q0**2 + q1**2

        return np.sum((np.matmul(self.x, self.param))) ** 2

    def backward(self, grad):
        grad_00 = 2 * self.param[0][0] * grad
        grad_01 = 2 * self.param[0][1] * grad
        grad_10 = 2 * self.param[1][0] * grad
        grad_11 = 2 * self.param[1][1] * grad
        return grad_00, grad_01, grad_10, grad_11

    def update(self, grad, lr=0.1):
        grad_00, grad_01, grad_10, grad_11, = grad

        self.param[0][0] -= lr * grad_00
        self.param[0][1] -= lr * grad_01
        self.param[1][0] -= lr * grad_10
        self.param[1][1] -= lr * grad_11


y_truth = 1
lr = 0.01
op = Op4()
loss_func = Loss()
while True:
    y_pred = op.forward()
    loss = loss_func.squared_loss(y_pred, y_truth)

    grad = loss_func.squared_loss_grad(y_pred, y_truth)
    param_grad = op.backward(grad)

    if loss < 1e-5:
        print(op.param)
        print(op.forward())
        break
    op.update(param_grad, lr)
