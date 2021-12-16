# 初始化常数K，随机初始化k个聚类中心
# 重复计算以下以下过程，直到【聚类中心】不再改变
#   计算【每个样本】与【每个聚类中心点】的距离，将样本划分到最近的中心点
#   计算划分到每个类别中的【所有样本特征的均值】，并将该均值作为每个类【新的聚类中心】
# 输出最终的聚类中心以及每个样本所属的类别。

import numpy as np
import matplotlib.pyplot as plt


def get_random_centroids(data, k, dim):
    centroids = np.zeros((k, dim))
    for j in range(dim):
        # 给每个维度生成k个随机值
        data_min = np.min(data[:, j])
        data_max = np.max(data[:, j])
        cent = (data_max - data_min) * np.random.rand(k) + data_min
        centroids[:, j] = cent
    return centroids


def KMeans(centroids, data, k, num):
    cluster_result = np.zeros((num, 2))  # 存取类别、距离

    cluster_changed = True
    while cluster_changed:
        cluster_changed = False

        # 遍历所有数据
        for i in range(num):
            dist_min = np.inf  # 与各个中心点的距离
            index_min = -1

            # 遍历每个中心点与对应数据的距离
            for j in range(k):
                dist = np.sqrt(np.sum(np.square(data[i] - centroids[j])))

                # 如果距离小于最小距离，更新该数据对应中心点的类别
                if dist < dist_min:
                    dist_min = dist
                    index_min = j

            # 若该点的类别非距离最小者，则继续遍历，且更新其最近的类别
            if cluster_result[i, 0] != index_min:
                cluster_changed = True
                cluster_result[i, :] = index_min, dist_min

        # 遍历完所有数据后，更新中心点为该类别数据的中点
        for cent in range(k):
            pts = np.nonzero(cluster_result[:, 0] == cent)
            pts_in_cluster = data[pts]
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)

    return centroids, cluster_result


def show_cluster(data, k, num, centroids, cluster_result):
    mark = ['or', 'ob', 'og', 'oy', 'oc', 'om']
    for i in range(num):
        mark_index = int(cluster_result[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[mark_index])
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], 'o', markersize=16)
    plt.show()


k = 5
dim = 2
num = 100
data = np.random.rand(num, dim)
centroids = get_random_centroids(data, k, dim)

centroids, cluster_result = KMeans(centroids, data, k, num)
show_cluster(data, k, num, centroids, cluster_result)
