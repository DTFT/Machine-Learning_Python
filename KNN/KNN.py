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



"""
    Desc:
        kNN 的分类函数
    Args:
        inX -- 用于分类的输入向量/测试数据
        dataSet -- 训练数据集的 features
        labels -- 训练数据集的 labels
        k -- 选择最近邻的数目
    Returns:
        sortedClassCount[0][0] -- 输入向量的预测分类 labels

    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet 的第一个点的距离。
       第二行： 同一个点 到 dataSet 的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet 的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    # 欧氏距离，取平方
    sqDiffMat = diffMat**2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances**0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        # 找到该样本对应的标签
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]


def datingClassTest():
     # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
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

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def test1():
    """
    第一个例子演示
    """
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))


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
   # classifyPerson()
   test1()
