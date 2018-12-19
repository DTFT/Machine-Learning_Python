# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:29:27 2018

@author: DTFT
"""
from __future__ import print_function

import operator
from collections import Counter
from math import log

import decisionTreePlot as dtPlot




def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算香浓熵
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

'''	splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
'''
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
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subdataSet = splitDataSet(dataSet, i, value)
			prob = len(subdataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subdataSet)
		
		infoGain =baseEntropy - newEntropy
		print("infoGain=", infoGain, "Feature=", i, baseEntropy, newEntropy)
		if(infoGain > bsetInfoGain):
			bsetInfoGain = infoGain
			beatFeature = i
	return beatFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]

	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])

	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

	return myTree

def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	key = testVec[featIndex]
	valueOfFeat = secondDict[key]
	print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
	if isinstance(valueOfFeat,dict):
		classLabel = classify(valueOfFeat, featLabels,testVec)
	else:
		classLabel = valueOfFeat
	return classLabel

def storeTree(inputTree, filename):
    import pickle
    # -------------- 第一种方法 start --------------
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    # -------------- 第一种方法 end --------------

    # -------------- 第二种方法 start --------------
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
    # -------------- 第二种方法 start --------------


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()
    # print myDat, labels

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))
    
    # 获得树的高度
    print(get_tree_height(myTree))

    # 画图可视化展现
    dtPlot.createPlot(myTree)
def get_tree_height(tree):
    """
     Desc:
        递归获得决策树的高度
    Args:
        tree
    Returns:
        树高
    """

    if not isinstance(tree, dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1


if __name__ == "__main__":
    fishTest()
