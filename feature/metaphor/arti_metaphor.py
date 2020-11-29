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

def get_wordnet_pos(w):

	try:
		treebank_tag = nltk.pos_tag(w)[0][1]
	except IndexError:
		return None

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

def cos_sim(x, y):
	return sum(map(lambda a, b:a * b, x, y)) / (sqrt(sum([a**2 for a in x])) * sqrt(sum([a**2 for a in y])))

if __name__ == '__main__':
	
	path = 'F:\wikipedia-dump\\'
	path1 = 'E:\pythonworkspace\metaphor_mental\metaphor\\article_metaphor\\'
	path2 = 'E:\pythonworkspace\metaphor_mental\metaphor\word_corpus_fit\\'
	in_vector = {}
	all_student = {}
	lemmatizer = WordNetLemmatizer()
	head = []

	with open(path + 'CBOW_in_vector.pkl','rb') as file:
		in_vector = pickle.load(file)

	with open('all_student.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			if all_student.has_key(row[0]):
				all_student[row[0]].append(row[1])
			else:
				all_student[row[0]] = [row[1]]

	for (k, value) in all_student.items():
		
		for student in value:
			words = {}
			out_f = 'E:\pythonworkspace\metaphor_mental\metaphor\\article_metaphor\\' + k + '\\' + student + '_metaphor.csv'
			out_f_words = 'E:\pythonworkspace\metaphor_mental\metaphor\\article_words_metaphor\\' + k + '\\' + student + '_metaphor_words.csv'

			with open(path2 + k + '\\' + student + '_fit.csv', 'rb') as ft:
				reader = csv.reader(ft)
				head = [reader.next()[0]]
				for row in reader:
					words[(row[0], row[1])] = row[2]

			threshold = 0.4
			for (key, value) in words.items():
				word = lemmatizer.lemmatize(value.lower(), get_wordnet_pos(value) or wn.NOUN)
				word_t = lemmatizer.lemmatize(key[1].lower(), get_wordnet_pos(key[1]) or wn.NOUN)
				if in_vector.has_key(word_t) and in_vector.has_key(word):
					words[key] = cos_sim(in_vector[word_t].tolist(), in_vector[word].tolist())
					if words[key] > threshold:
						words[key] = [words[key], 0]
					else:
						words[key] = [words[key], 1]
				else:
					words[key] = [1, 0]

			with open(out_f, 'wb') as ft:
				writer = csv.writer(ft)
				head.extend(['similarity', 'predict result'])
				writer.writerow(head)
				for (key, value) in words.items():
					temp = [key[0]]
					temp.extend(value)
					writer.writerow(temp)

			with open(out_f_words, 'wb') as ft:
				writer = csv.writer(ft)
				writer.writerow(['s-unit id', 'target word'])
				for (key, value) in words.items():
					if value[1] == 1:
						writer.writerow(key)