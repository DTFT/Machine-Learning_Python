# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:17:58 2018

@author: DR20160002
"""
import numpy as np
import matplotlib.pyplot as plt



def file2matrix(filename):
    fr = open(filename)
    numberOfLine = len(fr.readlines())
    returnMat = np.zeros((numberOfLine,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化
# def autoNorm(dataSet):
    





fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
ax.scatter(datingDataMat[:,0], datingDataMat[:,2], 15.0*(np.array(datingLabels)), 
           15.0*(np.array(datingLabels)))