#!python
# encoding=utf-8
from numpy import *


def loadDataSet(filename):
    dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# i为alpha下标，m是所有alpha的数目，
# 意思为确定一个第i个alpha，然后在剩下的所有alpha中任选一个
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 用于防止alpha越过上下界h,l
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 简化版的SMO算法
'''
创建一个alpha向量并将其初始化为0向量
当迭代次数下雨最大迭代次数时（外循环）
    对数据集中的每个数据向量（内循环）：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果这两个向量都不能被优化，推出内循环
    如果所有的向量都没被优化，增加迭代数目，继续下一次循环
'''


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()
    b = 0;
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0

    while (iter < maxIter):

        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * \
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] + alphas[i])
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H: print("L==H");continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0: print("eta>=0");continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough");
                    continue

                alphas[i] += labelMat[j] * labelMat[i] * \
                             (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b2 + b2) / 2.0
                    alphaPairsChanged += 1

                print("iter: %d i:%d, pairs changed %d" % \
                      (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

# 利用完整Platt SMO算法加速优化

