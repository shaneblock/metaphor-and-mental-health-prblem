# # -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import csv
import json
import random
from math import *
import time
import re
import xlrd
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.linear_model import LinearRegression

def main():

	file = 'all_features_binary.csv'
	encoder = LabelEncoder()
	coe_metrix = []
	reg_matrix = []

	labels = []
	features = []
	features_name = []
	labels_name = []
	with open(file, 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		features_name = fr[1:23]
		labels_name = fr[23:]
		for row in reader:
			features.append([float(e) for e in row[1:23]])
			labels.append([float(e) for e in row[23:]])

	scaler = MinMaxScaler()
	features = np.array(features)
	features = scaler.fit_transform(features)
	labels = np.array(labels)

	for x in range(features.shape[1]):
		temp = []
		for y in range(labels.shape[1]):
			temp.append(round(stats.pointbiserialr([float(e) for e in labels[:,y].tolist()], [float(e) for e in features[:,x].tolist()])[0], 4))
		coe_metrix.append(temp)
	with open('coe_metrix_binary.csv', 'wb') as ft:
		writer = csv.writer(ft)
		head = ['attr name']
		head.extend(labels_name)
		writer.writerow(head)
		for i in range(len(features_name)):
			row = [features_name[i]]
			row.extend(coe_metrix[i])
			writer.writerow(row)

	# sd_x = []
	# sd_y = []
	# for y in range(labels.shape[1]):
	# 	sd_y.append(sqrt(np.var(labels[:,y])))
	# for x in range(features.shape[1]):
	# 	sd_x.append(sqrt(np.var(features[:,x])))

	# for y in range(labels.shape[1]):
	# 	clf = LinearRegression(copy_X = True)
	# 	clf.fit(features, labels[:,y])
	# 	reg_coefficient = clf.coef_.tolist()
	# 	temp = []
	# 	for x in range(features.shape[1]):
	# 		temp.append(round(reg_coefficient[x] * (sd_x[x] / sd_y[y]), 4))
	# 	reg_matrix.append(temp)

	# with open('reg_coefficient.csv', 'wb') as ft:
	# 	writer = csv.writer(ft)
	# 	head = ['attr name']
	# 	head.extend(labels_name)
	# 	writer.writerow(head)
	# 	for i in range(len(features_name)):
	# 		row = [features_name[i]]
	# 		row.extend(reg_matrix[i])
	# 		writer.writerow(row)
	# cov = np.cov(data, rowvar = False)
	# # cor = np.corrcoef(data, rowvar = False)
	# with open('cov_of_features.csv', 'wb') as ft:
	# 	writer = csv.writer(ft)
	# 	writer.writerow(head)
	# 	i = 1
	# 	for row in list(cov):
	# 		temp = [head[i]]
	# 		temp.extend(row)
	# 		writer.writerow(temp)
	# 		i = i + 1
	

if __name__ == '__main__':
	main()