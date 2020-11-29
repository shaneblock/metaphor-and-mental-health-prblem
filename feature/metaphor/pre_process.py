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
		res.append([lemmatizer.lemmatize(word.lower(), pos = wordnet_pos), wordnet_pos])

	return res

if __name__ == '__main__':

	all_student = {}
	path = 'E:\pythonworkspace\metaphor_mental\\article_sentence\\'

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
			out = 'E:\pythonworkspace\metaphor_mental\metaphor\\target_wrod\\' + key + '\\' + student + '.csv'
			outfile = open(out, 'wb')
			writer = csv.writer(outfile)
			writer.writerow(['sentence id', 'sentence', 'VERB', 'ADJ', 'ADV'])
			with open(path + key + '\\' + student + '.csv', 'rb') as ft:
				reader = csv.reader(ft)
				i = 0
				for row in reader:
					temp = [i, row[0].decode('utf-8', 'ignore').lower().strip(), [], [], []]
					for word_pos in lemmatize_sentence(row[0].decode('utf-8', 'ignore').lower()):
						if word_pos[1] == wn.VERB:
							temp[2].append(word_pos[0])
						if word_pos[1] == wn.ADJ:
							temp[3].append(word_pos[0])
						if word_pos[1] == wn.ADV:
							temp[4].append(word_pos[0])
					temp[2] = '#'.join(temp[2])
					temp[3] = '#'.join(temp[3])
					temp[4] = '#'.join(temp[4])
					writer.writerow(temp)
					i = i + 1
				outfile.close()