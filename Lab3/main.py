#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import nltk
from nltk.tokenize import WordPunctTokenizer
from gensim.models import word2vec

nltk.data.path.append("../Lab2/nltk_data")

file = open("doc11.txt", 'r').read().split('\n')
file = [sentence for sentence in file if len(sentence) != 0]

corpus = [re.sub(r'[^a-zA-Z\s]', '', sentence, re.I | re.A).lower() for sentence in file]

print("Підготовлений корпус: ")
print(corpus)

# Task 1
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(corpus)
vocab = cv.get_feature_names_out()

array = pd.DataFrame(cv_matrix.toarray(), columns=vocab)

print(array['mariner'])
# --------
# Task 2
# Отримуємо TD-IDF матрицю
tt = TfidfTransformer(norm='l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix)

array2 = pd.DataFrame(tt_matrix.toarray(), columns=vocab)
print("TD-IDF: ")
print(array2)
# Отримуємо матрицю подібності
similarity_matrix = cosine_similarity(tt_matrix)

array3 = pd.DataFrame(similarity_matrix)
print("\n")
print("Similarity matrix: ")
print(array3)
print('\n')

# Creating links
links = linkage(similarity_matrix, 'ward')

# Creating plot
print("Dendrogram: ")
plt.figure(figsize=(8, 3))
plt.title('Dendrogram')
plt.xlabel('Documents')
plt.ylabel('Length')
dendrogram(links)
plt.show()

# From dendrogram we can choose max distance = 1.5
max_dist = 1.5

cluster_labels = fcluster(links, max_dist, criterion='distance')

array4 = pd.DataFrame(corpus, columns=["Sentences"])
array5 = pd.DataFrame(cluster_labels, columns=["Cluster Labels ID"])
array6 = pd.concat([array4, array5], axis=1)
print(array6)
print("\n\n")
# --------
# Task 3
wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(sentence) for sentence in corpus]
vector_size = 100
window = 30
min_count = 1
sample = 1e-3
w2v_model = word2vec.Word2Vec(tokenized_corpus,
                              vector_size=vector_size,
                              window=window,
                              min_count=min_count,
                              sample=sample)
# Similar words
sim_words = {search_term: [
                            item[0]
                            for item
                            in w2v_model.wv.most_similar(
                                                         [search_term],
                                                         topn=5
                                                        )
                            ]
              for search_term in ["mobile", "athens"]}

print(sim_words)
# --------
