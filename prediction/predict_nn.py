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
from math import *

import keras.backend as K
import sklearn.preprocessing as preprocessing

def visalization(history):

	import matplotlib.pyplot as plt
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc = 'upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc = 'upper left')
	plt.show()

def pred_class(pred):

	pred_c = []
	for l in pred:
		temp = [0] * len(l)
		temp[numpy.argmax(l)] = 1
		pred_c.append(temp)
	return pred_c

class Classifier():
	l = 2
	def __init__(self, l):
		self.seed = 70
		self.l = l
		numpy.random.seed(self.seed)

	def Run(self, X_train = None, Y_train = None, X_test = None, Y_test = None, cross_validate = True):
		def BuildModel(in_unit = 32, input_dim = 100, output_dim = 1):
			model = Sequential()
			model.add(Dense(in_unit, input_dim = input_dim, activation = None))
			model.add(Dropout(0.2))
			model.add(Dense(in_unit, activation = 'relu', kernel_regularizer = regularizers.l2(1)))
			model.add(Dropout(0.2))
			model.add(Dense(output_dim, activation = 'linear'))
			# model.compile(loss = 'mse', optimizer = 'adam', metrics = [f1_score,'accuracy'])
			if self.l == 1:
				model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
			else:
				model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
			return model

		if cross_validate:
			best_para = [0, 0]
			max_acc = 0
			for e in [40, 60, 80]:
				for batch in [100, 300, 500]:
					print(e, batch)

					estimator = KerasClassifier(build_fn = BuildModel, in_unit = 512, input_dim = X_train.shape[1], output_dim = l, epochs = e, batch_size = batch, verbose = 0)
					kfold = KFold(n_splits = 5, shuffle = True, random_state = self.seed)
					results = cross_val_score(estimator, X_train, Y_train, cv = kfold)
					if results.mean() > max_acc:
						best_para[0] = e
						best_para[1] = batch
						max_acc = results.mean()

			print("Baseline: %.2f%%" % (max_acc * 100))
			print("best_para: epochs = %d, batch_size = %d" % (best_para[0], best_para[1]))
		else:
			bs = 16
			model = BuildModel(16, X_train.shape[1], self.l)
			if self.l == 1:
				es = EarlyStopping(monitor = 'val_loss', patience = 5)
			else:
				es = EarlyStopping(monitor = 'val_acc', patience = 5)
			history = model.fit(X_train, Y_train, validation_split = 0.1, callbacks = [es], shuffle = True, batch_size = bs, epochs = 200, verbose = 0)

			# visalization(history)
			score = model.evaluate(X_test, Y_test, batch_size = bs, verbose = 0)
			pred = model.predict(X_test, batch_size = bs, verbose = 0)

			return score[1], pred
			# print("accuracy: %.2f%%" % (score[2] * 100))

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
			line.extend(row[23:25])
			line.extend([float(k) for k in row[25:]])
			dataset.append(line)
	dataset = numpy.array(dataset)

	original_X = dataset[:,1:25]
	original_Y = dataset[:,25:]
	samples = dataset[:0]

	encoder = LabelEncoder()

	X1 = original_X[:,:5]
	X1 = scaler.fit_transform(X1)

	X2 = original_X[:,5:12]
	X2 = scaler.fit_transform(X2)

	X3 = original_X[:,12:22]
	X3 = scaler.fit_transform(X3)

	# gender
	tmp22 = original_X[:,22].astype(str)
	encoder.fit(tmp22)
	tenc22 = encoder.transform(tmp22)
	enc22 = np_utils.to_categorical(tenc22)

	# location
	tmp23 = original_X[:,23].astype(str)
	encoder.fit(tmp23)
	tenc23 = encoder.transform(tmp23)
	enc23 = np_utils.to_categorical(tenc23)

	X_author = numpy.hstack((
		enc22, enc23,
		))

	# pca = PCA(n_components = 2)
	# X_author = pca.fit_transform(X_author)
	# X_author = scaler.fit_transform(X_author)

	X = numpy.hstack((
		X1, # article-wise
		X2, # metaphor-wise
		X3, # senti-wise
		X_author, # author-wise
		))

	encoder.fit(original_Y[:,-1])
	enc_Y = encoder.transform(original_Y[:,-1])

	dummy_Y = np_utils.to_categorical(enc_Y)

	return samples, X, original_Y, dummy_Y

if __name__ == '__main__':

	samples, features, label_reg, label_class = LoadData('all_features.csv')
	
	# accuracy = []
	# F1 = []

	# for i in range(10):
	# 	X_train, X_test, y_train, y_test = train_test_split(features, label_class, test_size = 0.2, random_state = i * 3, shuffle = True, stratify = label_class)
	# 	clf = Classifier(4)
	# 	# clf.Run(features, label)
	# 	predict_class = []
	# 	y_class = []
	# 	acc, pred = clf.Run(X_train = X_train, Y_train = y_train, X_test = X_test, Y_test = y_test, cross_validate = False)
	# 	with open('pred_' + str(i) + '.csv', 'w', newline = '') as ft:
	# 		writer = csv.writer(ft)
	# 		writer.writerow(['pid', 'pred', 'truth'])
	# 		for j in range(pred.shape[0]):
	# 			temp = [j]
	# 			temp.append(numpy.argmax(pred[j]))
	# 			temp.append(numpy.argmax(y_test[j]))
	# 			predict_class.append(numpy.argmax(pred[j]))
	# 			y_class.append(numpy.argmax(y_test[j]))
	# 			writer.writerow(temp)
	# 	accuracy.append(accuracy_score(y_class, predict_class))
	# 	F1.append(f1_score(predict_class, y_class, average = 'macro'))
	# print('accuracy:', accuracy)
	# print("accuracy: %.2f%%(%.2f%%)" % (numpy.mean(accuracy) * 100, sqrt(numpy.var(accuracy))))
	# print('f1_score:', F1)
	# print("f1_score: %.2f%%(%.2f%%)" % (numpy.mean(F1) * 100, sqrt(numpy.var(F1))))
	
	for l in range(45):
		mse = []
		r2 = []

		for i in range(10):
			X_train, X_test, y_train, y_test = train_test_split(features, label_reg, test_size = 0.2, random_state = i * 3)
			y_train = y_train[:,l]
			y_test = y_test[:,l]
			clf = Classifier(1)
			# clf.Run(features, label)
			predict_class = []
			y_class = []
			acc, pred = clf.Run(X_train = X_train, Y_train = y_train, X_test = X_test, Y_test = y_test, cross_validate = False)
			with open('pred_reg_' + str(i) + '.csv', 'w', newline = '') as ft:
				writer = csv.writer(ft)
				writer.writerow(['pid', 'pred', 'truth'])
				for j in range(pred.shape[0]):
					temp = [j]
					temp.append(pred[j][0])
					temp.append(y_test[j])
					predict_class.append(pred[j][0])
					y_class.append(float(y_test[j]))
					writer.writerow(temp)
			mse.append(mean_squared_error(predict_class, y_class))
			r2.append(r2_score(predict_class, y_class))
		print(l)
		print('mse:', mse)
		print("mse: %.2f(%.2f)" % (numpy.mean(mse), sqrt(numpy.var(mse))))
		print('r2_score:', r2)
		print("r2_score: %.2f(%.2f)" % (numpy.mean(r2), sqrt(numpy.var(r2))))