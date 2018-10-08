sklearn：multiclass与multilabel，one-vs-rest与one-vs-one

针对多类问题的分类中，具体讲有两种，即multiclass classification和multilabel classification。multiclass是指分类任务中包含不止一个类别时，每条数据仅仅对应其中一个类别，不会对应多个类别。multilabel是指分类任务中不止一个分类时，每条数据可能对应不止一个类别标签，例如一条新闻，可以被划分到多个板块。
 无论是multiclass，还是multilabel，做分类时都有两种策略，一个是one-vs-the-rest(one-vs-all)，一个是one-vs-one。

在one-vs-all策略中，假设有n个类别，那么就会建立n个二项分类器，每个分类器针对其中一个类别和剩余类别进行分类。进行预测时，利用这n个二项分类器进行分类，得到数据属于当前类的概率，选择其中概率最大的一个类别作为最终的预测结果。

在one-vs-one策略中，同样假设有n个类别，则会针对两两类别建立二项分类器，得到k=n*(n-1)/2个分类器。对新数据进行分类时，依次使用这k个分类器进行分类，每次分类相当于一次投票，分类结果是哪个就相当于对哪个类投了一票。在使用全部k个分类器进行分类后，相当于进行了k次投票，选择得票最多的那个类作为最终分类结果。

在scikit-learn框架中，分别有sklearn.multiclass.OneVsRestClassifier和sklearn.multiclass.OneVsOneClassifier完成两种策略，使用过程中要指明使用的二项分类器是什么。另外在进行mutillabel分类时，训练数据的类别标签Y应该是一个矩阵，第[i,j]个元素指明了第j个类别标签是否出现在第i个样本数据中。例如，np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])，这样的一条数据，指明针对第一条样本数据，类别标签是第0个类，第二条数据，类别标签是第1，第2个类，第三条数据，没有类别标签。有时训练数据中，类别标签Y可能不是这样的可是，而是类似[[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]这样的格式，每条数据指明了每条样本数据对应的类标号。这就需要将Y转换成矩阵的形式，sklearn.preprocessing.MultiLabelBinarizer提供了这个功能。



以Logistic为例子

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
iris = datasets.load_iris()
# print iris
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''sc.scale_标准差, sc.mean_平均值, sc.var_方差'''

lr = LogisticRegression(C=10000.0, random_state=0)
lr.fit(X_train_std, y_train)
print '系数：',lr.coef_,lr.intercept_
# 预测
#print lr.intercept_
print 'phiz', 1.0/(1+np.e**(-(np.dot(lr.coef_, X_test_std[0])+lr.intercept_)))
print 'decision', lr.decision_function(X_test_std[0])
print 'phiz',1.0/(1+np.e**(-lr.decision_function(X_test_std[0])))
y_pred = lr.predict(X_test_std)
print '测试集',X_test_std[0]
print '预测值', y_pred[0]
print '预测概率',lr.predict_proba(X_test_std[0])
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

输出

```python
系数： [[-9.38725178 -8.62196104]
 [ 2.54760621 -2.34616582]
 [ 9.8260878   6.51345035]] [-12.237882    -0.89485178  -9.10527128]
phiz [  1.41131020e-14   6.71679171e-02   9.99537323e-01]
decision [[-31.89167281  -2.6310295    7.67801924]]
phiz [[  1.41131020e-14   6.71679171e-02   9.99537323e-01]]
测试集 [ 0.70793846  1.50872803]
预测值 2
预测概率 [[  1.32305547e-14   6.29676452e-02   9.37032355e-01]]
Misclassified samples: 1
Accuracy: 0.98
```

使用decision_function函数算出的概率[  1.41131020e-14   6.71679171e-02   9.99537323e-01]与我们使用逻辑函数算出的概率一样的但是与算出的预测[  1.32305547e-14   6.29676452e-02   9.37032355e-01]]是不符的，sklearn是怎么算出来的呢，看下面

```python
a = [  1.41131020e-14,   6.71679171e-02,   9.99537323e-01]
a = np.array(a)
print a/a.sum()
```

输出

```
[  1.32305547e-14   6.29676452e-02   9.37032355e-01]
```

那么再看看系数三组值，sklearn默认是用one-vs-rest方法
 看下面代码,只要把目标值调整一下，剩下两类

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
iris = datasets.load_iris()
# print iris
X = iris.data[:, [2, 3]]
y = iris.target
for i in range(len(y)):
    if y[i] == 1:
        y[i] = 0
    if y[i] == 2:
        y[i] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''sc.scale_标准差, sc.mean_平均值, sc.var_方差'''

lr = LogisticRegression(C=10000.0, random_state=0)
lr.fit(X_train_std, y_train)
print '系数：',lr.coef_,lr.intercept_
# 预测
#print lr.intercept_
print 'phiz', 1.0/(1+np.e**(-(np.dot(lr.coef_, X_test_std[0])+lr.intercept_)))
print 'decision', lr.decision_function(X_test_std[0])
print 'phiz',1.0/(1+np.e**(-lr.decision_function(X_test_std[0])))
y_pred = lr.predict(X_test_std)
print '测试集',X_test_std[0]
print '预测值', y_pred[0]
print '预测概率',lr.predict_proba(X_test_std[0])
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

```
系数： [[ 9.8260878   6.51345035]] [-9.10527128]
phiz [ 0.99953732]
decision [ 7.67801924]
phiz [ 0.99953732]
测试集 [ 0.70793846  1.50872803]
预测值 1
预测概率 [[  4.62676696e-04   9.99537323e-01]]
Misclassified samples: 1
Accuracy: 0.98
```

 

 

 

 

 

 

 

 

 

 