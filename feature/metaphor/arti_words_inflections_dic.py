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
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from math import *
import csv
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

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wn.ADJ
	elif treebank_tag.startswith('V'):
		return wn.VERB
	elif treebank_tag.startswith('N'):
		return wn.NOUN
	elif treebank_tag.startswith('R'):
		return wn.ADV
	else:
		return None

def lemmatize_sentence(sentence):
	res = []
	lemmatizer = WordNetLemmatizer()
	text = nltk.word_tokenize(sentence)
	for word, pos in nltk.pos_tag(text):
		wordnet_pos = get_wordnet_pos(pos) or wn.NOUN
		res.append(lemmatizer.lemmatize(word, pos = wordnet_pos))

	return res, text

if __name__ == '__main__':

	path = 'E:\\2539\download\documentation\sentences\\'
	words_inflections = {}

	allfile = dirlist(path, [])

	for file in allfile:

		with open(file, 'rb') as ft:
			reader = csv.reader(ft)
			reader.next()
			for row in reader:
				if row:
					stem, raw = lemmatize_sentence(row[1].decode('utf-8', 'ignore').lower())
					for i in range(len(stem)):
						if stem[i] != raw[i]:
							if words_inflections.has_key(stem[i]):
								if raw[i] not in words_inflections[stem[i]]:
									words_inflections[stem[i]].append(raw[i])
							else:
								words_inflections[stem[i]] = [raw[i]]

	with open('arti_words_inflections_dic.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(['word', 'inflections'])
		for (key, value) in words_inflections.items():
			temp = [key]
			temp.extend(value)
			writer.writerow(temp)