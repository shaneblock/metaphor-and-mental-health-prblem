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
from nltk.stem import WordNetLemmatizer

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

def get_inflections(w):

	inflections = []
	MORPHOLOGICAL_SUBSTITUTIONS = {
		wn.NOUN: [
			('y', 'ies'),
			('man', 'men'),
			('sh', 'shes'),
			('ch', 'ches'),
			('z', 'zes'),
			('x', 'xes'),
			('f', 'ves'),
			('s', 'es'),
			('', 's'),
		],
		wn.VERB: [
			('y', 'ies'),
			('e', 'es'),
			('e', 'ed'),
			('y', 'ied'),
			('e', 'ing'),
			('', 's'),
			('', 'es'),
			('', 'ed'), 
			('', 'ing'),
		],
		wn.ADJ: [('', 'er'), ('', 'est'), ('e', 'er'), ('e', 'est')],
		wn.ADV: [('', 'er'), ('', 'est'), ('e', 'er'), ('e', 'est')],
	}
	pos = get_wordnet_pos(w)
	if MORPHOLOGICAL_SUBSTITUTIONS.has_key(pos):
		substitutions = MORPHOLOGICAL_SUBSTITUTIONS[pos]
		for tup in substitutions:
			if w.endswith(tup[0]):
				if len(tup[0]) > 0:
					inflections.append(w[:-len(tup[0])] + tup[1])
				else:
					inflections.append(w + tup[1])
		return inflections
	else:
		return None

if __name__ == '__main__':
	
	path = 'F:\wikipedia-dump\\'
	word_corpus = {}
	lemmatizer = WordNetLemmatizer()
	in_vector = {}
	visited = {}
	all_student = {}

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

	path = 'E:\pythonworkspace\metaphor_mental\metaphor\word_corpus\\'
	words_inflections = {}

	for (key, value) in all_student.items():
		for student in value:
		
			with open(path + key + '\\' + student + '_word_corpus.csv', 'rb') as ft:
				reader = csv.reader(ft)
				for row in reader:
					if len(row) == 4:
						words = row[2].split('#')
						for w in words:
							if not visited.has_key(w):
								w = lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w) or wn.NOUN)
								if in_vector.has_key(w):
									inflections = get_inflections(w)
									if inflections:
										inflections = [inf for inf in inflections if in_vector.has_key(inf)]
										word_corpus[w] = inflections
							visited[w] = ''

	with open('articles_words_inflections.csv', 'wb') as ft:
		writer = csv.writer(ft)
		writer.writerow(['Word', 'Inflection'])
		for key, value in word_corpus.items():
			temp = [key]
			temp.extend(value)
			writer.writerow(temp)
