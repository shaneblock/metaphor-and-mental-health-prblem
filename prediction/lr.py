# # -*- coding: utf-8 -*-

import os
import sys
import numpy
import pandas
import time
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import regularizers

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
import sklearn.preprocessing as preprocessing

def F1(pred, truth):
	return (f1_score(pred, truth, pos_label = 1) + f1_score(pred, truth, pos_label = 0)) / 2

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
			line.extend([float(k) for k in row[1:23]])
			line.extend([float(k) for k in row[23:]])
			dataset.append(line)
	dataset = numpy.array(dataset)
	print(dataset.shape)

	original_X = dataset[:,1:23]
	# original_Y = numpy.hstack((
	# 	dataset[:,25:26],	# anxious
	# 	dataset[:,26:27],	# depress
	# 	dataset[:,28:29],	# inferiority
	# 	dataset[:,29:30],	# sensitive
	# 	dataset[:,30:31],	# social phobia
	# 	dataset[:,35:36],	# obsession
	# 	dataset[:,45:46],	# total
	# 	))
	original_Y = dataset[:,23:]
	original_Y = original_Y.astype(numpy.float64)
	samples = dataset[:,0]

	encoder = LabelEncoder()

	X1 = original_X[:,:5]
	X1 = X1.astype(numpy.float64)
	X1 = scaler.fit_transform(X1)

	X2 = original_X[:,5:12]
	X2 = X2.astype(numpy.float64)
	X2 = scaler.fit_transform(X2)

	X3 = original_X[:,12:22]
	X3 = X3.astype(numpy.float64)
	X3 = scaler.fit_transform(X3)

	X = numpy.hstack((
		# X1, # article-wise
		X2, # metaphor-wise
		X3, # senti-wise
		))
	print(original_Y.shape)

	return samples, X, original_Y

if __name__ == '__main__':

	samples, features, label_reg = LoadData('features_binary_v2.csv')
	
	ft = open('clf_result.csv', 'w', newline = '')
	writer = csv.writer(ft)
	writer.writerow(['labels', 'acc', 'f1'])
	for l in range(7):
		pred = []
		test_l = []
		print(l)
		skf = StratifiedKFold(n_splits = 10)
		
		max_f1 = 0
		best_c = 1
		data_spilit = list(skf.split(features, label_reg[:,l]))
		for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			loc_pred = []
			loc_test = []
			
			for train, test in data_spilit:
				X_train = features[train,:]
				Y_train = label_reg[:,l][train]
				X_test = features[test,:]
				Y_test = label_reg[:,l][test]
				# clf = SVC(kernel = 'linear', C = c, class_weight = 'balanced')
				# clf = LogisticRegression(penalty = 'l2', C = c, class_weight = 'balanced', solver = 'lbfgs', max_iter = 1000)
				# clf = GaussianNB()
				clf.fit(X_train, Y_train)
				loc_pred.extend(clf.predict(X_test).tolist())
				loc_test.extend(Y_test.tolist())
				clf = None
			if F1(loc_pred, loc_test) > max_f1:
				max_f1 = F1(loc_pred, loc_test)
				best_c = c
		for train, test in data_spilit:
			X_train = features[train,:]
			Y_train = label_reg[:,l][train]
			X_test = features[test,:]
			Y_test = label_reg[:,l][test]
			# clf = SVC(kernel = 'linear', C = best_c, class_weight = 'balanced')
			clf = LogisticRegression(penalty = 'l2', C = best_c, class_weight = 'balanced', solver = 'lbfgs', max_iter = 1000)
			# clf = GaussianNB()
			clf.fit(X_train, Y_train)
			pred.extend(clf.predict(X_test).tolist())
			test_l.extend(Y_test.tolist())
		temp = [l, round(accuracy_score(pred, test_l), 4), round(F1(pred, test_l), 4)]
		writer.writerow(temp)
		with open('clf_result_' + str(l) + '.csv', 'w', newline = '') as fpr:
			writer_pr = csv.writer(fpr)
			writer_pr.writerow(['pred', 'truth'])
			for k in range(len(pred)):
				writer_pr.writerow([pred[k], test_l[k]])