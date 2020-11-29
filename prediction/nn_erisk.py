# # -*- coding: utf-8 -*-

import os
import sys
import numpy
import pandas
import time
import csv
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import activations

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from math import *

import keras.backend as K
import tensorflow as tf
import sklearn.preprocessing as preprocessing

def F1(pred, truth):
	return (f1_score(pred, truth, pos_label = 1) + f1_score(pred, truth, pos_label = 0)) / 2

def nrelu(x):
	return activations.relu(x, alpha = -1.0, max_value = 0.0, threshold = 0.0)
	# return activations.relu(x)

def LoadData(file_name):
	import sklearn.preprocessing as preprocessing
	scaler = preprocessing.MinMaxScaler()
	
	csvFile = file_name
	dataset = []
	with open(csvFile, 'r') as ft:
		reader = csv.reader(ft)
		next(reader)
		for row in reader:
			line = [row[0]]
			line.extend([float(k) for k in row[1:18]])
			line.append(float(row[18]))
			dataset.append(line)
	dataset = numpy.array(dataset)

	original_X = dataset[:,1:18]
	# original_Y = numpy.hstack((
	# 	dataset[:,25:26],	# anxious
	# 	dataset[:,26:27],	# depress
	# 	dataset[:,28:29],	# inferiority
	# 	dataset[:,29:30],	# sensitive
	# 	dataset[:,30:31],	# social phobia
	# 	dataset[:,35:36],	# obsession
	# 	dataset[:,45:46],	# total
	# 	))
	original_Y = dataset[:,18]
	original_Y = original_Y.astype(numpy.float64)
	samples = dataset[:,0]

	encoder = LabelEncoder()

	X1 = original_X[:,:7]
	X1 = X1.astype(numpy.float64)
	X1 = scaler.fit_transform(X1)

	X2 = original_X[:,7:17]
	X2 = X2.astype(numpy.float64)
	X2 = scaler.fit_transform(X2)

	X = numpy.hstack((
		X1, # metaphor-wise
		# X2, # senti-wise
		))

	return samples, X, original_Y

def nn_model(features):

	in_put = keras.layers.Input(shape = (features.shape[1],))
	out = Dense(200)(in_put)
	out_pos = Activation('relu')(out)
	out_neg = Activation(nrelu)(out)
	out = concatenate([out_pos, out_neg], axis = -1)
	out = Dropout(0.4)(out)
	out = Dense(100)(out)
	out_pos = Activation('relu')(out)
	out_neg = Activation(nrelu)(out)
	out = concatenate([out_pos, out_neg], axis = -1)
	out = Dense(50)(out)
	out_pos = Activation('relu')(out)
	out_neg = Activation(nrelu)(out)
	out = concatenate([out_pos, out_neg], axis = -1)
	out = Dense(2, activation = 'softmax')(out)
	model = keras.models.Model(in_put, out)
	#model.compile(loss = 'mse', optimizer = 'adam,adadelta', metrics = [f1_score,'accuracy'])
	model.compile(loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])
	return model

if __name__ == '__main__':

	# samples, features, label_reg = LoadData_b('features_binary_v2.csv')
	samples, features_train, label_train = LoadData('all_features_train.csv')
	print(features_train.shape[1])
	
	best_result = [0, 0]
	best_param = [32, 100]
	# skf = StratifiedKFold(n_splits = 10)
	# data_spilit = list(skf.split(features_train, label_train))
	# for e in [50, 75, 100, 125, 150, 175, 200]:
	# 	for bs in [8, 16, 24, 32, 40, 48, 56, 64]:
	# 		pred = []
	# 		test_l = []
	# 		for train, test in data_spilit:
	# 			X_train = features_train[train,:]
	# 			Y_train = np_utils.to_categorical(label_train[train], 2)
	# 			X_test = features_train[test,:]
	# 			Y_test = np_utils.to_categorical(label_train[test], 2)
	# 			clf = nn_model(features_train)
	# 			# clf = LogisticRegression(penalty = 'l2', C = best_c, class_weight = 'balanced', solver = 'lbfgs', max_iter = 1000)
	# 			# clf = GaussianNB()
	# 			clf.fit(X_train, Y_train, batch_size = bs, epochs = e, verbose = 0)
	# 			pred_re = clf.predict(X_test).tolist()				
	# 			K.clear_session()
	# 			tf.reset_default_graph()
	# 			Y_test = Y_test.tolist()
	# 			for resu in range(len(Y_test)):
	# 				if pred_re[resu][0] > 0.5:
	# 					pred.append(0)
	# 				else:
	# 					pred.append(1)
	# 				if Y_test[resu][0] > 0.5:
	# 					test_l.append(0)
	# 				else:
	# 					test_l.append(1)
	# 			if round(F1(pred, test_l), 4) > best_result[1]:
	# 				best_result[0] = round(accuracy_score(pred, test_l), 4)
	# 				best_result[1] = round(F1(pred, test_l), 4)
	# 				best_param = [bs, e]
	
	samples, features_test, label_test = LoadData('all_features_test.csv')
	label_train = np_utils.to_categorical(label_train, 2)
	label_test = np_utils.to_categorical(label_test, 2)
	acc = []
	f1 = []
	for i in range(10):
		print(i)
		pred = []
		test_l = []
		clf = nn_model(features_train)
		clf.fit(features_train, label_train, batch_size = best_param[0], epochs = best_param[1], verbose = 0)
		pred_re = clf.predict(features_test).tolist()				
		K.clear_session()
		tf.reset_default_graph()
		Y_test = label_test.tolist()
		for resu in range(len(Y_test)):
			if pred_re[resu][0] > 0.5:
				pred.append(0)
			else:
				pred.append(1)
			if Y_test[resu][0] > 0.5:
				test_l.append(0)
			else:
				test_l.append(1)
		acc.append(accuracy_score(pred, test_l))
		f1.append(F1(pred, test_l))
		# print('accuracy:', round(accuracy_score(pred, test_l), 4))
		# print('f1:', round(F1(pred, test_l), 4))

		with open('clf_result_erisk.csv', 'w', newline = '') as fpr:
			writer_pr = csv.writer(fpr)
			writer_pr.writerow(['pred', 'truth'])
			for k in range(len(pred)):
				writer_pr.writerow([pred[k], test_l[k]])
	print('accuracy:', round(numpy.mean(acc), 4))
	print('f1:', round(numpy.mean(f1), 4))
	print('acc:', acc)
	print('f1:', f1)