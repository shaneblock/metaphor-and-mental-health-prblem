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

	file = 'all_features.csv'
	encoder = LabelEncoder()
	coe_metrix = []

	labels = []
	features = []
	features_name = []
	labels_name = []
	with open(file, 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		features_name = fr[:23]
		labels_name = fr[48:]
		for row in reader:
			features.append(row[:23])
			labels.append(row[48:])
	with open('all_features_binary.csv', 'wb') as ft:
		writer = csv.writer(ft)
		head = features_name
		head.extend(labels_name)
		writer.writerow(head)
		for i in range(len(features)):
			row = features[i]
			row.extend([float(float(k) > 0) for k in labels[i][:-1]])
			if float(labels[i][-1]) == 0:
				row.append(0)
			else:
				row.append(1)
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