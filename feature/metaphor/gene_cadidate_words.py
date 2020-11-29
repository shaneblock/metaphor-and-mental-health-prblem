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

if __name__ == '__main__':
	
	head = ['s-unit id', 'target word', 'candidates', 'context']
	in_vector = {}
	all_student = {}

	english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '``', "''"]
	# stops = set(stopwords.words("english"))
	stops = ['the', 'a', 'an', 'this', 'that', 'these', 'those', "'s", "n't", "'m", 'i', 'you', 'me', 'he', 'them']
	pos = ['MD', 'PRP', 'SYM', 'TO', 'UH', 'IN']

	with open('F:\wikipedia-dump\\' + 'CBOW_in_vector.pkl','rb') as file:
		in_vector = pickle.load(file)

	with open('all_student.csv', 'rb') as ft:
		reader = csv.reader(ft)
		reader.next()
		for row in reader:
			if all_student.has_key(row[0]):
				all_student[row[0]].append(row[1])
			else:
				all_student[row[0]] = [row[1]]

	path = 'E:\pythonworkspace\metaphor_mental\metaphor\\target_wrod\\'
	words_inflections = {}

	for (key, value) in all_student.items():
		for student in value:
			out = 'E:\pythonworkspace\metaphor_mental\metaphor\\word_corpus\\' + key + '\\' + student + '_word_corpus.csv'
			word_corpus = []

			with open(path + key + '\\' + student + '.csv', 'rb') as ft:
				reader = csv.reader(ft)
				reader.next()
				for row in reader:
					if row:
						text = nltk.word_tokenize(row[1].decode('utf-8', 'ignore'))
						text = [word for word in text if word not in english_punctuations]
						target_words = []
						for item in row[2:]:
							if len(item) > 0:
								target_words.extend(item.decode('utf-8', 'ignore').lower().split('#'))
						target_words = list(set(target_words))
						target_words = [w for w in target_words if w not in stops]

						# text = [word for word in text if word not in stops]
						# word_pos = nltk.pos_tag(text)
						# text = [tag[0] for tag in word_pos if tag[1] not in pos]
						for tw in target_words:
							temp = [row[0], tw]
							candidates = []
							context = text[:]
							try:
								context.remove(tw)
							except ValueError:
								continue
							syn_hyp = []
							for syn_set in wn.synsets(tw):
								syn_hyp.extend(syn_set.lemma_names())
								for hyp_set in syn_set.hypernyms():
									syn_hyp.extend(hyp_set.lemma_names())
							syn_hyp = list(set(syn_hyp))

							candidates.extend(syn_hyp)
							candidates = list(set(candidates))
							temp.append('#'.join(candidates))
							temp.append('#'.join(context))
							word_corpus.append(temp)

			with open(out, 'wb') as ft:
				writer = csv.writer(ft)
				writer.writerow(head)
				for row in word_corpus:
					writer.writerow(row)
