# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:29:27 2018

@author: DTFT
"""
from math import log

def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	# 计算分类标签label出现的次数
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	# 对于 label 标签的占比，求出 label 标签的香农熵
	shannonEnt = 0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob,2)
	return shannonEnt


def splitDataSet(dataSet, index, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[index] == value:
			reducedFeatVec = featVec[:index]
			reducedFeatVec.extend(featVec[index+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0])-1
	baseEntropy = calcShannonEnt(dataSet)
	bsetInfoGain, beatFeature = 0.0, -1
	for i in range(numFeatures):
		# 将dataSet中的数据先按行依次放入example中，然后取得example中的example[i]元素，放入列表featList中
		featList = [example[i] for example in dataSet]

