#!/usr/bin/env python3

import re
import csv
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import gensim

nltk.data.path.append("../Lab2/nltk_data")

prepared_corpus = []

skip_generation = False

ans = input("Do you want to load already prepared corpus? (Y/N) ").lower()

if ans == 'y':
    try:
        with open("prepared_corpus.txt", 'r') as file:
            for line in file:
                prepared_corpus.append(line.strip())
        skip_generation = True
    except FileNotFoundError:
        print("Prepared corpus not found, generating it.")

if skip_generation == False:
    file = open("news.csv", 'r')
    next(file)  # To skip headers row
    csv = csv.DictReader(file, delimiter=',', quotechar='"', fieldnames=['text', 'label'])
    texts = {i: v for i, v in enumerate(csv)}
    labels = [texts[i]['label'] for i in range(len(texts))]

    wpt = WordPunctTokenizer()

    # Preparing corpus
    corpus = [texts[i]['text'] for i in range(len(texts)) if len(texts[i]['text']) != 0]


    def preprocess_sentence(sentence):
        sentence = re.sub(r'[^a-zA-Z\s]|http\S+', '', sentence, re.I | re.A)
        sentence = sentence.lower()
        sentence = sentence.strip()
        tokens = wpt.tokenize(sentence)
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        sentence = ' '.join(filtered_tokens)
        return sentence

    for sentence in corpus:
        prepared_corpus.append(preprocess_sentence(sentence))

    with open("prepared_corpus.txt", 'w') as file2:
        for sentence in prepared_corpus:
            file2.write(sentence + '\n')

# print("Підготовлений корпус: ")
# print(prepared_corpus)

# Creating BOW (Bag Of Words)

cv = CountVectorizer(binary=False, min_df=0., max_df=1.)
bow_corpus = cv.fit_transform(prepared_corpus)

# --------
# Creating id2word dictionary

single_array = [sentence.split() for sentence in prepared_corpus]

bigram = gensim.models.Phrases(single_array, min_count=10, threshold=20, delimiter='_')
bigram_model = gensim.models.phrases.Phraser(bigram)

print(bigram_model)

norm_corpus_bigrams = [bigram_model[doc] for doc in prepared_corpus]
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)

print(dictionary)

array = pd.DataFrame(dictionary)

print(array)

# --------
# Task 1. Latent Dirichlet Allocation

#lda_model = gensim.models.LdaModel(
#    corpus = bow_corpus,
#    id2word = 
#)

# --------
