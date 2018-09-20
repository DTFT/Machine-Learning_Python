# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:17:58 2018

@author: DR20160002
"""
import numpy as np
import operator
import matplotlib.pyplot as plt


"""
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵returnMat和对应的类别classLabelVector
"""
def file2matrix(filename):
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLine = len(fr.readlines())
    # 生成对应的空矩阵
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
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


# 归一化
'''
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
'''

def autoNorm(dataSet):
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxValx = dataSet.max(0)
    # 极差
    ranges = maxValx - minVals
    # 构建归一化matrix 与 dataSet shape  一致
    normDataSet = np.zeros(np.shape(dataSet))
    # 取行数
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    # np.tile 复制矩阵minVals, shape (m,1)
    normDataSet = dataSet - np.tile((minVals), (m,1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / np.tile((ranges),(m,1))

    return normDataSet, ranges, minVals


'''
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


def datingClassTest():
    hoRtion = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRtion)
    print('numTestVecs',numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],
        normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("the total error rate is: %f" %(errorCount / float(numTestVecs)))
        print(errorCount)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats,iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,
                                 normMat, datingLabels, 3)
    print("You will probably like this person: "
          ,resultList[classifierResult - 1])





# fig = plt.figure()
# ax = fig.add_subplot(111)
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
# ax.scatter(datingDataMat[:,0], datingDataMat[:,2], 15.0*(np.array(datingLabels)),
#            15.0*(np.array(datingLabels)))


if __name__ == '__main__':
    # datingClassTest()
    classifyPerson()
