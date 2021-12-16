import numpy as np
import pylab as pl
from sklearn import svm

# 创建 40 个点
np.random.seed(0)  # 让每次运行程序生成的随机样本点不变
# 生成训练实例并保证是线性可分的
# np._r表示将矩阵在行方向上进行相连
# array(20,2) - [2,2] 相当于给每一行的两个数都减去 2
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# 两个类别 每类有 20 个点，Y 为 40 行 1 列的列向量
Y = [0] * 20 + [1] * 20

# 建立 svm 模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 获得划分超平面
# 划分超平面原方程：w0x0 + w1x1 + b = 0
# 将其转化为点斜式方程，并把 x0 看作 x，x1 看作 y，b 看作 w2
# 点斜式：y = -(w0/w1)x - (w2/w1)
w = clf.coef_[0]  # w 是一个二维数据，coef 就是 w = [w0,w1]
a = -w[0] / w[1]  # 斜率

xx = np.linspace(-5, 5)  # 从 -5 到 5 产生一些连续的值（随机的）

# .intercept_[0] 获得 bias，即 b 的值，b / w[1] 是截距
yy = a * xx - (clf.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程

# 画出和划分超平面平行且经过支持向量的两条线（斜率相同，截距不同）
b = clf.support_vectors_[0]  # 取出第一个支持向量点
yy_down = a * xx + (b[1] - a * b[0])

b = clf.support_vectors_[-1]  # 取出最后一个支持向量点
yy_up = a * xx + (b[1] - a * b[0])

# 查看相关的参数值
print("w: ", w)
print("a: ", a)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)

# 在 scikit-learin 中，coef_ 保存了线性模型中划分超平面的参数向量。形式为(n_classes, n_features)。若 n_classes > 1，则为多分类问题，(1，n_features) 为二分类问题。

# 绘制划分超平面，边际平面和样本点
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
# 圈出支持向量
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
