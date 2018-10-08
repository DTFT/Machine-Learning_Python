# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 09:34:16 2018

@author: DR20160002
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    data_arr = []
    label_arr = []
    f = open('TestSet.txt','r')
    for line in f.readlines():
        line_arr = line.strip().split()
        data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr, label_arr

def sigmoid(x):
    return 1.0 /(1 + np.exp(-x))

def grad_ascent(data_arr, class_lables):

    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_lables).transpose()

    m, n = np.shape(data_mat)
    alpha = 0.001
    max_cycles = 500

    weights = np.ones((n,1))
    for k in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        weights = weights + alpha * data_mat.transpose() * error

    return np.array(weights)

def plot_best_fit(weights):

    data_mat, label_mat = load_data_set()
    data_arr =np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []

    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

def stoc_grad_ascent0(data_mat, class_labels):
	m, n = np.shape(data_mat)
	alpha = 0.01
	weights = np.ones(n)
	for i in range(m):
		h = sigmoid(sum(data_mat[i] * weights))
		error = class_labels[i] - h
		weights = weights + alpha * error * data_mat[i]
	return weights

def stoc_grad_ascent1(data_mat, class_labels, num_iter = 150):
	m, n = np.shape(data_mat)
	weights = np.ones(n)
	for j in range(num_iter):
		data_index = list(range(m))
		for i in range(m):
			alpha = 4 / (1.0 + i + j) + 0.01
			rand_index = int(np.random.uniform(0, len(data_index)))
			h = sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
			error = class_labels[data_index[rand_index]] - h
			weights = weights + alpha * error * data_mat[data_index[rand_index]]
			del(data_index[rand_index])

		return weights

def test():
    data_arr, class_labels = load_data_set()
    # weights = grad_ascent(data_arr, class_labels)
    weights = stoc_grad_ascent0(np.array(data_arr),class_labels)
    print('weights: ',weights)
    plot_best_fit(weights)

if __name__ == '__main__':
    test()
