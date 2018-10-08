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

# 随机梯度上升
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
    weights = stoc_grad_ascent1(np.array(data_arr),class_labels)
    print('weights: ',weights)
    plot_best_fit(weights)


# ________________从疝气病症预测病马的死亡率____________

def classify_vector(in_x, weights):
	prob = sigmoid(np.sum(in_x * weights))
	if prob > 0.5:
		return 1
	return 0

def colic_test():
	f_train = open('HorseColicTraining.txt','r')
	f_test = open('HorseColicTest.txt','r')
	train_set = []
	train_labels = []
	for line in f_train.readlines():
		curr_line = line.strip().split('\t')
		if len(curr_line) == 1:
			continue
		line_arr = []
		for i in range(21):
			line_arr.append(float(curr_line[i]))
		train_set.append(line_arr)
		train_labels.append(float(curr_line[21]))

	train_weights = stoc_grad_ascent1(np.array(train_set), train_labels, 500)
	error_count = 0
	num_test_vec = 0.0

	for line in f_test.readlines():
		num_test_vec += 1
		curr_line =line.strip().split('\t')
		if len(curr_line) == 1:
			continue
		line_arr = []
		for i in range(21):
			line_arr.append(float(curr_line[i]))
		if int(classify_vector(np.array(line_arr),train_weights)) != int(curr_line[21]) :
			error_count += 1
	error_rate = error_count / num_test_vec
	print('the error rate is {}'.format(error_rate))
	return error_rate

def multi_test():
	num_tests = 10
	error_sum = 0
	for i in range(num_tests):
		error_sum += colic_test()
	print('after {} iteration the average error rate is {}'.format(num_tests, error_sum / num_tests))








if __name__ == '__main__':
    # test()
    # colic_test()
    multi_test()
