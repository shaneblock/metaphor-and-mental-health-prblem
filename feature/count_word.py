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
import nltk

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
	classes = ['1803', '1807', '1811', '1815', 'w1802', 'w1805']
	english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '``', "''"]
	all_student = {}

	for c in classes:
		allfile = dirlist(path + c + '\\', [])
		for s in allfile:
			count = 0
			with open(s, 'rb') as ft:
				reader = csv.reader(ft)
				for row in reader:
					row[0] = row[0].strip()
					words = nltk.word_tokenize(row[0])
					for w in words:
						if w not in english_punctuations:
							count = count + 1
			all_student[s.split('\\')[-1][:-4].split('-')[0]] = count

	with open('all_student_words_count.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(['student id', 'word count'])
		for (key, value) in all_student.items():
			temp = [key, value]
			writer.writerow(temp)
	print 'min:', min(all_student.values())
	print 'max:', max(all_student.values())
	print 'mean:', np.mean(all_student.values())
	print 'sumation:', sum(all_student.values())