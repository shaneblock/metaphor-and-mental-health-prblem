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
import os
from senticnet.senticnet import SenticNet

def is_float(d):

	try:
		float(d)
		return True
	except:
		return False

if __name__ == '__main__':

	head = []
	feature = []

	with open('The features of sentiment.csv', 'rb') as ft:
		reader = csv.reader(ft)
		fr = reader.next()
		head = ['student id']
		head.extend(fr[5:])
		for row in reader:
			temp = []
			temp.append(row[0].split('-')[0])
			for d in row[1:]:
				if is_float(d):
					temp.append(d)
			feature.append(temp)

	with open('features_of_word_sentiment.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(head)
		for row in feature:
			writer.writerow(row)