# -*- encoding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pickle
import numpy as np
import time
from gensim.models import Word2Vec
import gensim.models as gmodels
import csv
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from math import *
import os

def dirlist(path, allfile):

	filelist =  os.listdir(path)  

	for filename in filelist:  
		filepath = os.path.join(path, filename)  
		if os.path.isdir(filepath):  
			dirlist(filepath, allfile)  
		else:  
			allfile.append(filepath)  
	return allfile

if __name__ == '__main__':
	
	path = 'F:\wikipedia-dump\\'
	path1 = 'E:\pythonworkspace\metaphor_mental\metaphor\\article_metaphor\\'
	path2 = 'E:\pythonworkspace\metaphor_mental\sentence_sentiment\\'

	head = ['student id', 'total words metaphor', 'token metaphor rate', 'm/s', 'pos sentics meta', 'neg sentics meta', 'sentics metaphor rate', 'avg metaphor senti']
	all_student = {}
	metaphor_features = {}
	tokens = {}
	student_metaphor_features = {}
	arti_feature = {}

	with open('all_student.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			if all_student.has_key(row[0]):
				all_student[row[0]].append(row[1])
			else:
				all_student[row[0]] = [row[1]]

	with open('E:\pythonworkspace\metaphor_mental\\features_of_article.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			arti_feature[row[0]] = float(row[1])

	for (key, value) in all_student.items():
		for student in value:
			metaphor = {}
			with open(path1 + key + '\\' + student + '_metaphor.csv', 'rb') as ft:
				reader = csv.reader(ft)
				reader.next()
				m = 0
				for row in reader:
					m = m + 1
					if metaphor.has_key(row[0]):
						metaphor[row[0]] = metaphor[row[0]] + float(row[2])
					else:
						metaphor[row[0]] = float(row[2])
				student_metaphor_features[student] = [sum(metaphor.values()), sum(metaphor.values()) * 1.0 / m, sum(metaphor.values()) * 1.0 / arti_feature[student]]

			with open(path2 + key + '\\' + student + '_ClassID.csv', 'rb') as ft:
				reader = csv.reader(ft)
				reader.next()
				pos = 0
				neg = 0
				total = 0
				for row in reader:
					if metaphor.has_key(row[0]):
						if float(row[4]) > 0:
							pos = pos + metaphor[row[0]]
						if float(row[4]) < 0:
							neg = neg + metaphor[row[0]]
						total = total + metaphor[row[0]] * float(row[4])
				student_metaphor_features[student].extend([pos, neg, (pos + neg) * 1.0 / sum(metaphor.values()), total * 1.0 / sum(metaphor.values())])

	with open('features_of_metaphor.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for (key, value) in student_metaphor_features.items():
			temp = [key]
			temp.extend(value)
			writer.writerow(temp)