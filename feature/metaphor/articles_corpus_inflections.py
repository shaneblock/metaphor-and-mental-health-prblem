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

if __name__ == '__main__':
	
	path = 'E:\pythonworkspace\metaphor_mental\metaphor\word_corpus\\'
	head = []
	words_inflections = {}
	all_student = {}
	lemmatizer = WordNetLemmatizer()

	with open('articles_words_inflections.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			if len(row) > 1:
				words_inflections[row[0]] = list(set(row[1:]))

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
			word_corpus = []
			out = 'E:\pythonworkspace\metaphor_mental\metaphor\word_corpus_extend\\' + key + '\\' + student + '_extend.csv'
		
			with open(path + key + '\\' + student + '_word_corpus.csv', 'rb') as ft:
				reader = csv.reader(ft)
				head = reader.next()
				for row in reader:
					if len(row) == 4:
						temp = row[:]
						text = row[2].split('#')
						candidates = text[:]
						for w in text:
							w = lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w) or wn.NOUN)
							if words_inflections.has_key(w):
								candidates.extend(words_inflections[w])

						candidates = list(set(candidates))
						temp[2] = '#'.join(candidates)
						word_corpus.append(temp)

			with open(out, 'wb') as ft:
				writer = csv.writer(ft)
				writer.writerow(head)
				for row in word_corpus:
					writer.writerow(row)