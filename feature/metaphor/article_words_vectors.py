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
	for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
		wordnet_pos = get_wordnet_pos(pos) or wn.NOUN
		res.append([lemmatizer.lemmatize(word, pos = wordnet_pos), wordnet_pos])

	return res

def cos_sim(x, y):
	return sum(map(lambda a, b:a * b, x, y)) / (sqrt(sum([a**2 for a in x])) * sqrt(sum([a**2 for a in y])))

if __name__ == '__main__':
	
	path = 'F:\wikipedia-dump\\'
	path1 = 'E:\pythonworkspace\metaphor_mental\metaphor\word_corpus_extend\\'
	in_vector = {}
	out_vector = {}
	all_student = {}
	head = []

	with open(path + 'CBOW_in_vector.pkl','rb') as file:
		in_vector = pickle.load(file)
	with open(path + 'CBOW_out_vector.pkl','rb') as file:
		out_vector = pickle.load(file)

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
			context_vector = {}
			best_fit_word = {}
			context = {}
			candidates = {}
			out_f = 'E:\pythonworkspace\metaphor_mental\metaphor\word_corpus_fit\\' + k + '\\' + student + '_fit.csv'
			with open(path1 + k + '\\' + student + '_extend.csv', 'rb') as ft:
				reader = csv.reader(ft)
				head = reader.next()[:2]
				for row in reader:
					context[(row[0], row[1])] = row[3].split('#')
					candidates[(row[0], row[1])] = row[2].split('#')

			for (key, value) in context.items():
				context_vector[key] = None
				count = 0
				for word in value:
					if in_vector.has_key(word):
						if count == 0:
							context_vector[key] = in_vector[word]
							count = 1
						else:
							context_vector[key] = in_vector[word] + context_vector[key]
							count = count + 1
				if count != 0:
					context_vector[key] = (context_vector[key] / count).tolist()
				if count == 0:
					context_vector[key] = list(np.random.uniform(-1, 1, size = 100))

			for (key, value) in candidates.items():
				max_cos = -1
				best_fit = None
				for word in value:
					if out_vector.has_key(word):
						out = [float(a) for a in out_vector[word].tolist()]
						cos = cos_sim(out, context_vector[key])
						if cos > max_cos:
							max_cos = cos
							best_fit = word
				best_fit_word[key] = best_fit

			with open(out_f, 'wb') as ft:
				writer = csv.writer(ft)
				head.append('Best Fit Word')
				writer.writerow(head)
				for (key, value) in context.items():
					temp = [key[0], key[1]]
					temp.append(best_fit_word[key])
					writer.writerow(temp)