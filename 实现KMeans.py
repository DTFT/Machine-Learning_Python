
from numpy import *

# 从文本中构建矩阵，加载文本文件，然后处理
# 通用函数，用来解析以 tab 键分隔的 floats（浮点数），例如: 1.658985    4.285136
def loadDataSet(fileName): 
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():	
		curLine = line.strip().split('\t')
		fitLine = map(float, curLine)   # 映射所有的元素为 float（浮点数）类型
		dataMat.append(fitLine)
	return dataMat

# 计算两个向量的欧式距离（可根据场景选择）
def distEclud(vecA, VecB):
	return sqrt(sum(power(vecA - vecB, 2)))



# 为给定数据集构建一个包含 k 个随机质心的集合。
#随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。
#然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
	n = shape(dataSet)[1]  # 列的数量
	centroids = mat(zeros((k, n)) # 创建k个质心矩阵
	for j in range(n):      # 创建随机簇质心，并且在每一维的边界内
		minJ = min(dataSet[:,j]) #返回j列最小值
		rangeJ = float(max(dataSet[:,j]) - minJ)  # 范围 = 最大值 - 最小值
		centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) # 随机生成质心
	return centroids
		
