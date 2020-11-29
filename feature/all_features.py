# -*- encoding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import numpy as np
import time
from gensim.models import Word2Vec
import gensim.models as gmodels
from math import *
import random
import os
import csv
import json
import re
import xlrd

if __name__ == '__main__':

	all_student = {}
	all_features = {}
	head = ['student id']

	with open('features_of_student.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			all_student[float(row[0])] = ''

	with open('features_of_article.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend(fr[1:])
		for row in reader:
			if all_student.has_key(float(row[0])):
				all_features[float(row[0])] = row[1:]

	with open('features_of_metaphor.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend(fr[1:])
		for row in reader:
			if all_student.has_key(float(row[0])):
				all_features[float(row[0])].extend(row[1:])

	with open('features_of_sentiment.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend(fr[1:])
		for row in reader:
			if all_student.has_key(float(row[0])):
				all_features[float(row[0])].extend(row[1:])

	with open('features_of_student.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend(fr[1:])
		for row in reader:
			if all_student.has_key(float(row[0])) and all_features.has_key(float(row[0])):
				all_features[float(row[0])].extend(row[1:])

	with open('all_features.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for (key, value) in all_features.items():
			temp = [key]
			temp.extend(value)
			writer.writerow(temp)