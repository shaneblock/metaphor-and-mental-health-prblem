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
	all_student = {}

	for c in classes:
		all_student[c] = []
		allfile = dirlist(path + c + '\\', [])
		for s in allfile:
			all_student[c].append(s.split('\\')[-1][:-4].split('-')[0])

	with open('all_studen.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(['class', 'student id'])
		for (key, value) in all_student.items():
			for s in value:
				temp = [key, s]
				writer.writerow(temp)