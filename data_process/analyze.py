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

	all_features = {}
	head = ['student_id']
	with open('all_features.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head.extend([fr[50], fr[51], fr[53], fr[54], fr[55], fr[60], fr[70]])
		for row in reader:
			all_features[row[0]] = [float(row[50]), float(row[51]), float(row[53]), float(row[54]), float(row[55]), float(row[60]), float(row[70])]

	sumation = []
	for (key, value) in all_features.items():
		sumation.append(sum(value))
	sd = sqrt(np.var(sumation))
	sd = sd * 0.2
	print sd

	all_student = []
	head.append('sum')
	head.append('tag')
	with open('selected.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for (key, value) in all_features.items():
			temp = [key]
			temp.extend(value)
			temp.append(sum(value))
			if sum(value) > -sd and sum(value) < sd:
				temp.append(0)
			else:
				temp.append(1)
				all_student.append(key)
			writer.writerow(temp)

	with open('all_selected_student.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(['student_id'])
		for s in all_student:
			writer.writerow([s])