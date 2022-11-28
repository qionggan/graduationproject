# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def L2(vecXi, vecXj):
    '''
    计算欧氏距离
    para vecXi：点坐标，向量
    para vecXj：点坐标，向量
    retrurn: 两点之间的欧氏距离
    '''
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))

def kMeans(S, k, distMeas=L2):
    '''
    K均值聚类
    para S：样本集，多维数组
    para k：簇个数
    para distMeas：距离度量函数，默认为欧氏距离计算函数
    return sampleTag：一维数组，存储样本对应的簇标记
    return clusterCents：一维数组，各簇中心
    retrun SSE:误差平方和
    '''
    m = np.shape(S)[0] # 样本总数
    sampleTag = np.zeros(m)
    
    # 随机产生k个初始簇中心
    n = np.shape(S)[1] # 样本向量的特征数
    # clusterCents = np.mat([[-1.93964824,2.33260803],[7.79822795,6.72621783],[10.64183154,0.20088133]])
    clusterCents = np.mat(np.zeros((k,n)))
    for j in range(n):
       minJ = min(S[:,j])
       rangeJ = float(max(S[:,j]) - minJ)
       clusterCents[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
        
    sampleTagChanged = True
    SSE = 0.0
    while sampleTagChanged: # 如果没有点发生分配结果改变，则结束
        sampleTagChanged = False
        SSE = 0.0
        
        # 计算每个样本点到各簇中心的距离
        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(k):
                d = distMeas(clusterCents[j,:],S[i,:]) #得到数据与质心的距离
                if d < minD: #每次进行判断是否比上次距离更小
                    minD = d    #进行存储更小的距离，直至比较到最后取到最小的距离
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex   #更新类标签
            SSE += minD**2    #样本点到类质心的距离平方和
        print(clusterCents)
        plt.scatter(clusterCents[:,0].tolist(),clusterCents[:,1].tolist(),c='r',marker='^',linewidths=7)
        plt.scatter(S[:,0],S[:,1],c=sampleTag,linewidths=np.power(sampleTag+0.5, 2))
        plt.show()
        print(SSE)
        
        # 重新计算簇中心
        for i in range(k):
            ClustI = S[np.nonzero(sampleTag[:]==i)[0]]  #找到当前类质心下所有数据点，通过数组过滤得到簇中所有元素
            clusterCents[i,:] = np.mean(ClustI, axis=0) #更新质心，求平均值
    return clusterCents, sampleTag, SSE

if __name__=='__main__':
    samples = np.loadtxt("kmeansSamples.txt")
    clusterCents, sampleTag, SSE = kMeans(samples, 3)
    # plt.scatter(clusterCents[:,0].tolist(),clusterCents[:,1].tolist(),c='r',marker='^')
    # plt.scatter(samples[:,0],samples[:,1],c=sampleTag,linewidths=np.power(sampleTag+0.5, 2))
    plt.show()
    print(clusterCents)
    print(SSE)