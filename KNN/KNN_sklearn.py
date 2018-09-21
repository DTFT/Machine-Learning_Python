import numpy as np
from sklearn import neighbors,datasets

# 选取最近的几个点作比较
n_neighbors = 3
#导入数据
iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

h = .02  # 网格步长
#分类器分类界面颜色
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#数据点颜色
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


for weights in ['uniform','distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)

    #绘制决策边界
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,0].min() - 1, X[:,0].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),)
