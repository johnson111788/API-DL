# -*- coding: utf-8 -*-
# @Time     : 2021/12/08
# @Author   : Yu-Cheng, Chou
# @Email    : johnson111788@gmail.com
# @FileName : backpropagation.py
# @Reference: https://github.com/antsfamily/pyml/tree/master/pyml

import numpy as np


def linear(x):
    r"""linear activation

    :math:`y = x`

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """
    return x


def sigmoid(x):
    r"""sigmoid function

    .. math::
        y = \frac{e^x}{e^x + 1}


    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """
    ex = np.exp(x)

    return ex / (ex + 1)


def tanh(x):
    r"""Computes tanh of `x` element-wise.

    Specifically

    .. math::
        y = {\rm tanh}(x) = {{e^{2x} - 1} \over {e^{2x} + 1}}.

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs

    """

    e2x = np.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

    # return np.tanh(x)


def softplus(x):
    r"""Computes softplus: `log(exp(x) + 1)`.

    :math:`{\rm log}(e^x + 1)`

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs

    """

    return np.log(np.exp(x) + 1)


def softsign(x):
    r"""Computes softsign: `x / (abs(x) + 1)`.

    :math:`\frac{x} {({\rm abs}(x) + 1)}`

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs

    """

    return x / (np.abs(x) + 1)


def elu(x):
    r"""Computes exponential linear element-wise. exp(x) - 1` if x < 0, `x` otherwise

    .. math::
        y = \left\{ {\begin{array}{*{20}{c}}{x,\;\;\;\;\;\;\;\;\;x \ge 0}\\{{e^x} - 1,\;\;\;x < 0}\end{array}} \right..

    See  `Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) <http://arxiv.org/abs/1511.07289>`_

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """

    return np.where(x < 0, np.exp(x) - 1, x)


def relu(x):
    r"""Computes rectified linear: `max(x, 0)`.

    :math:`{\rm max}(x, 0)`

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """

    # x[x < 0] = 0
    # return x
    return np.where(x > 0, x, 0)


def relu6(x):
    r"""Computes Rectified Linear 6: `min(max(x, 0), 6)`.

    :math:`{\rm min}({\rm max}(x, 0), 6)`

    `Convolutional Deep Belief Networks on CIFAR-10. A. Krizhevsky <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """
    maxx = np.where(x > 0, x, 0)
    return np.where(maxx < 6, maxx, 6)


def selu(x):
    r"""Computes scaled exponential linear: `scale * alpha * (exp(x) - 1)` if < 0, `scale * x` otherwise.

    .. math::
        y = \lambda \left\{ {\begin{array}{*{20}{c}}{x,\;\;\;\;\;\;\;\;\;\;\;\;\;x \ge 0}\\{\alpha ({e^x} - 1),\;\;\;\;x < 0}\end{array}} \right.

    where, :math:`\alpha = 1.6732632423543772848170429916717` , :math:`\lambda = 1.0507009873554804934193349852946`

    See `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return np.where(x < 0, scale * alpha * (np.exp(x) - 1), scale * x)


def crelu(x):
    r"""Computes Concatenated ReLU.

    Concatenates a ReLU which selects only the positive part of the activation
    with a ReLU which selects only the *negative* part of the activation.
    Note that as a result this non-linearity doubles the depth of the activations.
    Source: `Understanding and Improving Convolutional Neural Networks via
    Concatenated Rectified Linear Units. W. Shang, et
    al. <https://arxiv.org/abs/1603.05201>`_

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs

    """

    return x


def leaky_relu(x, alpha=0.2):
    r"""Compute the Leaky ReLU activation function.

    :math:`y = \left\{ {\begin{array}{*{20}{c}}{x,\;\;\;\;\;\;x \ge 0}\\{\alpha x,\;\;\;x < 0}\end{array}} \right.`

    `Rectifier Nonlinearities Improve Neural Network Acoustic Models <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`_


    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """

    return np.where(x < 0, alpha * x, x)


def swish(x, beta=1.0):
    r"""Computes the Swish activation function: `x * sigmoid(beta*x)`.

    :math:`y = x\cdot {\rm sigmoid}(\beta x) = {e^{(\beta x)} \over {e^{(\beta x)} + 1}} \cdot x`

    See `"Searching for Activation Functions" (Ramachandran et al. 2017) <https://arxiv.org/abs/1710.05941>`_

    Arguments:
        x {lists or array} -- inputs

    Returns:
        array -- outputs
    """

    ex = np.exp(beta * x)

    return (ex / (ex + 1)) * x


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    colors = ['k', 'm', 'b', 'g', 'c', 'r',
              '-.m', '-.b', '-.g', '-.c', '-.r']
    activations = ['tanh', 'sigmoid', 'relu', 'leaky_relu', 'swish']
    # activations = ['linear', 'tanh', 'sigmoid', 'softplus', 'softsign',
    #                'elu', 'relu', 'selu', 'relu6', 'leaky_relu', 'swish']

    x = np.linspace(-10, 10, 200)


    plt.figure()
    for activation, color in zip(activations, colors):
        print("---show activation: " + activation + "---")
        y = globals()[activation](x)
        plt.plot(x, y, color)
    plt.title('activation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-1, 3)

    plt.grid()
    plt.legend(activations)
    plt.show()

