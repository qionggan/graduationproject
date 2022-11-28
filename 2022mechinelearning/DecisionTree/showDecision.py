# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:15:29 2019

@author: WangHJ
"""

from decision_bitree import build_biTree, print_tree

def load_data(file_name):
    '''导入数据
    input:  file_name(string):训练数据保存的文件名
    output: data_train(list):训练数据
    '''
    data_train = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split(" ")
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return data_train

import matplotlib.pyplot as plt
if __name__ == "__main__":  
    
    data_train = load_data("classificationSamples.txt")

    x1 = []
    y1 = []
    x0 = []
    y0 = []
    for x in data_train:
        if x[-1] > 0.0:
            x1.append(x[0])
            y1.append(x[1])
        else:
            x0.append(x[0])
            y0.append(x[1])
    plt.scatter(x1,y1,c='r',marker='.')
    plt.scatter(x0,y0,c='b',marker='x')
    plt.show()
    
    tree = build_biTree(data_train, splitInfo="infogain")
    
    print_tree(tree)
    
    data_train = load_data("classificationSamples_noise.txt")

    x1 = []
    y1 = []
    x0 = []
    y0 = []
    for x in data_train:
        if x[-1] > 0.0:
            x1.append(x[0])
            y1.append(x[1])
        else:
            x0.append(x[0])
            y0.append(x[1])
    plt.scatter(x1,y1,c='r',marker='.')
    plt.scatter(x0,y0,c='b',marker='x')
    plt.show()
    
    tree = build_biTree(data_train, splitInfo="infogain")
    
    print_tree(tree)


