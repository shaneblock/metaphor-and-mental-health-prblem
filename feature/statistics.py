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
from nltk.tokenize import WordPunctTokenizer

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

	path = 'E:\pythonworkspace\metaphor_mental\\article_sentence\\'
	all_student = {}
	statis = {}

	with open('all_student.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			if all_student.has_key(row[0]):
				all_student[row[0]].append(row[1])
			else:
				all_student[row[0]] = [row[1]]

	for (key, value) in all_student.items():
		for student in value:
			with open(path + key + '\\' + student + '.csv', 'rb') as ft:
				reader = csv.reader(ft)
				count = 0
				words = 0
				self_count = 0
				for row in reader:
					count = count + 1
					row[0] = row[0].strip()
					words_list = WordPunctTokenizer().tokenize(row[0])
					
					for w in words_list:
						if w.lower() in ['me', 'i', 'my']:
							self_count = self_count + 1
					words = words + len(words_list)
				statis[student] = [count, words, self_count]

	with open('features_of_article.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(['student id', 'sentence', 'word', 'w/s', 'self word', 'self word rate'])
		for (key, value) in statis.items():
			temp = [key, value[0], value[1], value[1] * 1.0 / value[0], value[2], value[2] * 1.0 / value[1]]
			writer.writerow(temp)