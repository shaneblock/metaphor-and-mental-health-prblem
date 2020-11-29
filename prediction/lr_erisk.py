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
		X2, # senti-wise
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

	samples, features, label_reg = LoadData('all_features_train.csv')

	pred = []
	test_l = []
	skf = StratifiedKFold(n_splits = 10)
	
	max_f1 = 0
	best_c = 1
	data_spilit = list(skf.split(features, label_reg))
	for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
		loc_pred = []
		loc_test = []
		
		for train, test in data_spilit:
			X_train = features[train,:]
			Y_train = label_reg[train]
			X_test = features[test,:]
			Y_test = label_reg[test]
			# clf = SVC(kernel = 'linear', C = c, class_weight = 'balanced')
			clf = LogisticRegression(penalty = 'l2', C = c, class_weight = 'balanced', solver = 'lbfgs', max_iter = 1000)
			# clf = GaussianNB()
			clf.fit(X_train, Y_train)
			loc_pred.extend(clf.predict(X_test).tolist())
			loc_test.extend(Y_test.tolist())
			clf = None
		if F1(loc_pred, loc_test) > max_f1:
			max_f1 = F1(loc_pred, loc_test)
			best_c = c
	samples, features_test, label_test = LoadData('all_features_test.csv')

	clf = LogisticRegression(penalty = 'l2', C = best_c, class_weight = 'balanced', solver = 'lbfgs', max_iter = 1000)
	clf.fit(features, label_reg)
	pred.extend(clf.predict(features_test).tolist())
	test_l.extend(label_test.tolist())
	print('accuracy:', round(accuracy_score(pred, test_l), 4))
	print('f1_score:', round(F1(pred, test_l), 4))

	with open('clf_result_erisk.csv', 'w', newline = '') as fpr:
		writer_pr = csv.writer(fpr)
		writer_pr.writerow(['pred', 'truth'])
		for k in range(len(pred)):
			writer_pr.writerow([pred[k], test_l[k]])