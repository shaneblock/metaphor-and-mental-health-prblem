# -*- encoding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import numpy as np
import time
import csv
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
import os
from math import *

if __name__ == '__main__':

	selected_student = {}
	with open('all_selected_student.csv') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			selected_student[row[0]] = ''

	head = ['student_id']
	all_sample = {}
	with open('all_features_binary_v2.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend(fr[1:23])
		head.extend([fr[25], fr[26], fr[28], fr[29], fr[30], fr[35], fr[45]])
		for row in reader:
			temp = row[1:23]
			temp.extend([row[25], row[26], row[28], row[29], row[30], row[35], row[45]])
			all_sample[row[0]] = temp
	with open('features_binary_v2.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for (key, value) in all_sample.items():
			key = str(key).split('.')[0]
			if selected_student.has_key(key):
				temp = [key]
				temp.extend(value)
				writer.writerow(temp)

	head = ['student_id']
	all_sample = {}
	with open('liwc_features_v2.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend(fr[1:])
		for row in reader:
			temp = row[1:]
			all_sample[row[0]] = temp
	with open('selected_liwc_features_v2.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for (key, value) in all_sample.items():
			if selected_student.has_key(key):
				temp = [key]
				temp.extend(value)
				writer.writerow(temp)