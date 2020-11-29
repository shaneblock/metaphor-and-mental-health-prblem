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
from scipy import stats

def main():

	file = 'all_features_binary_v2.csv'
	encoder = LabelEncoder()
	coe_metrix = []

	labels = []
	features = []
	features_name = []
	labels_name = []
	with open(file, 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()[1:]
		features_name = fr[:17]
		labels_name = fr[17:]
		for row in reader:
			features.append(row[1:18])
			labels.append(row[18:])
	
	features = np.array(features)
	labels = np.array(labels)

	for x in range(features.shape[1]):
		temp = []
		for y in range(labels.shape[1]):
			temp.append(abs(round(stats.pearsonr([float(e) for e in features[:,x].tolist()], [float(e) for e in labels[:,y].tolist()])[0], 4)))
		coe_metrix.append(temp)
	with open('all_coe_metrix_v2.csv', 'wb') as ft:
		writer = csv.writer(ft)
		head = ['attr name']
		head.extend(labels_name)
		writer.writerow(head)
		for i in range(len(features_name)):
			row = [features_name[i]]
			row.extend(coe_metrix[i])
			writer.writerow(row)
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