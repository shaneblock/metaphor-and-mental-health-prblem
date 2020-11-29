# -*- encoding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import numpy as np
import time
from math import *
import random
import os
import csv
import json
import re
import xlrd

def isfloat(i):
	try:
		float(i)
		return True
	except ValueError:
		return False

if __name__ == '__main__':

	features = {}
	labels = {}
	head = ['student id']

	data = xlrd.open_workbook("LIWC2015_Results.xlsx")

	table = data.sheets()[0]
	head.extend(table.row_values(0)[1:])
	for i in range(1, table.nrows):
		row = table.row_values(i)
		features[row[0].split('.')[0]] = row[1:]
		print row[0].split('.')[0]

	with open('E:\pythonworkspace\metaphor_mental\prediction\\all_features_binary_v2.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend([fr[i] for i in [25, 26, 28, 29, 30, 35, 45]])
		for row in reader:
			labels[str(row[0]).split('.')[0]] = [row[i] for i in [25, 26, 28, 29, 30, 35, 45]]

	with open('liwc_features_v2.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for (key, value) in labels.items():
			temp = [key]
			if features.has_key(key):
				temp.extend(features[key])
				temp.extend(value)
				writer.writerow(temp)