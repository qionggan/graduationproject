# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:06:21 2017

@author: wanghengjun
"""

import matplotlib.pyplot as plt

def load_data(file_name):
    '''导入数据
    input:  file_name(string):训练数据保存的文件名
    output: X1(list):正类x轴
            Y1(list):正类y轴
            X2(list):负类x轴
            Y2(list):负正类y轴
    '''
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split("\t")
        if float(lines[-1]) >= 0.0:
            X1.append(lines[0])
            Y1.append(lines[1])
        else:
            X2.append(lines[0])
            Y2.append(lines[1])
    f.close()
    return X1, Y1, X2, Y2

X1, Y1, X2, Y2 = load_data("data.txt")

plt.axis([0, 4, -2, 4])
plt.scatter(X1, Y1, c='r', marker='x')
plt.scatter(X2, Y2, c='b', marker='x')
plt.show

X1, Y1, X2, Y2 = load_data("final_result")

plt.scatter(X1, Y1, c='g', marker='.')
plt.scatter(X2, Y2, c='c', marker='.')
plt.show