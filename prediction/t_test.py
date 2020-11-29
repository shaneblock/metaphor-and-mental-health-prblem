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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from math import *

from scipy import stats

import keras.backend as K
import sklearn.preprocessing as preprocessing

def F1(pred, truth):
	return (f1_score(pred, truth, pos_label = 1) + f1_score(pred, truth, pos_label = 0)) / 2

def LoadData(file_name):
	import sklearn.preprocessing as preprocessing
	scaler = preprocessing.MinMaxScaler()
	scaler = preprocessing.StandardScaler()
	
	csvFile = file_name
	dataset = []
	head = []
	with open(csvFile, 'r') as ft:
		reader = csv.reader(ft)
		head = next(reader)
		for row in reader:
			line = [row[0]]
			line.extend([float(k) for k in row[1:23]])
			line.extend([float(k) for k in row[23:]])
			dataset.append(line)
	dataset = numpy.array(dataset)

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
	X1 = numpy.sum(X1, axis = 1).reshape(-1, 1)

	X2 = original_X[:,5:12]
	X2 = X2.astype(numpy.float64)
	X2 = scaler.fit_transform(X2)
	X2 = numpy.sum(X2, axis = 1).reshape(-1, 1)

	X3 = original_X[:,12:22]
	X3 = X3.astype(numpy.float64)
	X3 = scaler.fit_transform(X3)
	X3 = numpy.sum(X3, axis = 1).reshape(-1, 1)

	X = numpy.hstack((
		# X1, # article-wise
		X2, # metaphor-wise
		X3, # senti-wise
		))
	X = numpy.sum(X, axis = 1)

	return samples, X, original_Y, head

if __name__ == '__main__':

	samples, features, label_reg, head = LoadData('features_binary_v2.csv')
	features_name = head[1:23]
	labels_name = head[23:]
	t_metrix = []
	p_metrix = []
	
	for l in range(label_reg.shape[1]):
		pos_index = numpy.where(label_reg[:,l] == 1)
		neg_index = numpy.where(label_reg[:,l] == 0)
		temp_t = [labels_name[l]]
		temp_p = []
		t_test = stats.ttest_ind(features[pos_index], features[neg_index])
		# for x in range(features.shape[1]):
		# 	fv = features[:,x]
		# 	t_test = stats.ttest_ind(fv[pos_index], fv[neg_index])
		# 	temp_t.append(round(t_test[0], 4))
		# 	temp_p.append(round(t_test[1], 4))
		# t_metrix.append(temp_t)
		# p_metrix.append(temp_p)
		print(t_test)
	# with open('p_metrix.csv', 'w', newline = '') as ft:
	# 	writer = csv.writer(ft)
	# 	fr = [' ']
	# 	fr.extend(features_name)
	# 	writer.writerow(fr)
	# 	for row in p_metrix:
	# 		writer.writerow(row)
	# with open('t_metrix.csv', 'w', newline = '') as ft:
	# 	writer = csv.writer(ft)
	# 	fr = [' ']
	# 	fr.extend(features_name)
	# 	writer.writerow(fr)
	# 	for row in t_metrix:
	# 		writer.writerow(row)