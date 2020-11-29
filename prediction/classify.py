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
from sklearn import linear_model
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
	original_Y = numpy.hstack((
		dataset[:,25:26],	# anxious
		dataset[:,26:27],	# depress
		dataset[:,28:29],	# inferiority
		dataset[:,29:30],	# sensitive
		dataset[:,30:31],	# social phobia
		dataset[:,35:36],	# obsession
		dataset[:,45:46],	# total
		))
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
		X1, # article-wise
		X2, # metaphor-wise
		X3, # senti-wise
		))
	print(original_Y.shape)

	return samples, X, original_Y

if __name__ == '__main__':

	samples, features, label_reg = LoadData('all_features_binary.csv')
	
	ft = open('clf_result.csv', 'w', newline = '')
	writer = csv.writer(ft)
	writer.writerow(['labels', 'acc', 'f1'])
	for l in range(7):
		pred = []
		test = []
		print(l)

		# for i in range(10):
		# 	X_train, X_test, y_train, y_test = train_test_split(features, label_reg, test_size = 0.1, random_state = i * 3)
		# 	y_train = y_train[:,l]
		# 	y_test = y_test[:,l]
		# 	# clf = SVR(kernel = 'linear', C = 100)
		# 	# clf = linear_model.BayesianRidge()
		# 	# clf = linear_model.Lasso(alpha = 0.1)
		# 	clf = linear_model.Ridge(alpha = 1, copy_X = True)
		# 	clf.fit(X_train, y_train)
		# 	pred = clf.predict(X_test)
		# 	with open('pred_reg_' + str(i) + '.csv', 'w', newline = '') as ftw:
		# 		writerf = csv.writer(ftw)
		# 		writerf.writerow(['pid', 'pred', 'truth'])
		# 		for j in range(pred.shape[0]):
		# 			tep = [j]
		# 			tep.append(pred[j])
		# 			tep.append(y_test[j])
		# 			writerf.writerow(tep)
			
		# 	mse.append(mean_squared_error(pred, y_test))
		# 	r2.append(r2_score(pred, y_test))
		# temp = [l, numpy.mean(mse), numpy.mean(r2)]
		# # print(l)
		# # print('mse:', mse)
		# # print("mse: %.2f(%.2f)" % (numpy.mean(mse), sqrt(numpy.var(mse))))
		# # print('r2_score:', r2)
		# # print("r2_score: %.2f(%.2f)" % (numpy.mean(r2), sqrt(numpy.var(r2))))
		# writer.writerow(temp)
		max_f1 = 0
		best_c = 1
		for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			loc_pred = []
			loc_test = []
			for i in range(samples.shape[0]):
				X_train = numpy.vstack((features[:i,:], features[i + 1:,:]))
				Y_train = numpy.hstack((label_reg[:,l][:i], label_reg[:,l][i + 1:]))
				X_test = features[i,:].reshape(1, -1)
				Y_test = label_reg[:,l][i]
				clf = SVC(kernel = 'linear', C = c, class_weight = 'balanced')
				# clf = GaussianNB()
				clf.fit(X_train, Y_train)
				loc_pred.append(clf.predict(X_test))
				loc_test.append(Y_test)
				clf = None
			if F1(loc_pred, loc_test) > max_f1:
				max_f1 = F1(loc_pred, loc_test)
				best_c = c
		for i in range(samples.shape[0]):
			X_train = numpy.vstack((features[:i,:], features[i + 1:,:]))
			Y_train = numpy.hstack((label_reg[:,l][:i], label_reg[:,l][i + 1:]))
			X_test = features[i,:].reshape(-1, 1).T
			Y_test = label_reg[:,l][i]
			clf = SVC(kernel = 'linear', C = best_c, class_weight = 'balanced')
			# clf = GaussianNB()
			clf.fit(X_train, Y_train)
			pred.append(clf.predict(X_test))
			test.append(Y_test)
		temp = [l, round(accuracy_score(pred, test), 4), round(F1(pred, test), 4)]
		writer.writerow(temp)
		with open('clf_result_' + str(l) + '.csv', 'w', newline = '') as fpr:
			writer_pr = csv.writer(fpr)
			writer_pr.writerow(['pred', 'truth'])
			for k in range(len(pred)):
				writer_pr.writerow([pred[k][0], test[k]])