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
			line.extend([float(k) for k in row[1:14]])
			line.extend([float(k) for k in row[14:]])
			dataset.append(line)
	dataset = numpy.array(dataset)
	print(dataset.shape)

	original_X = dataset[:,1:14]
	original_Y = dataset[:,14:]
	original_Y = original_Y.astype(numpy.float64)
	samples = dataset[:,0]

	encoder = LabelEncoder()

	X = scaler.fit_transform(original_X)

	return samples, X, original_Y

if __name__ == '__main__':

	samples, features, label_reg = LoadData('liwc_features.csv')
	
	ft = open('liwc_clf_result.csv', 'w', newline = '')
	writer = csv.writer(ft)
	writer.writerow(['labels', 'acc', 'f1'])
	for l in range(7):
		pred = []
		test = []
		print(l)

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