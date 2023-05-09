#!/usr/bin/env python3

import re
import csv
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

nltk.data.path.append("../Lab2/nltk_data")

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


prepared_corpus = []

for sentence in corpus:
    prepared_corpus.append(preprocess_sentence(sentence))

print("Підготовлений корпус: ")
print(prepared_corpus)

# Task 1 - Split data into train and test
train_corpus, test_corpus, train_label_nums, test_label_nums, \
    = train_test_split(prepared_corpus, labels,
                       test_size=0.3, random_state=0)
# --------
# Task 2 - Bag of Words
cv = CountVectorizer(binary=False, min_df=0., max_df=1.)
cv_train_matrix = cv.fit_transform(train_corpus)
cv_test_matrix = cv.transform(test_corpus)
vocab = cv.get_feature_names_out()

array = pd.DataFrame(cv_train_matrix.toarray(), columns=vocab)

print(array)
# --------
# Task 3 - NaiveBayesClassifier
mnb = MultinomialNB(alpha=1)
mnb.fit(cv_train_matrix, train_label_nums)
# --------
# Task 4 - RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
rfc.fit(cv_train_matrix, train_label_nums)
# --------
# Task 5 - Compare accuracy
mnb_prediction = mnb.predict(cv_test_matrix)
mnb_accuracy = accuracy_score(test_label_nums, mnb_prediction)
print("Accuracy of MultinomialNB: " + str(mnb_accuracy))

rfc_prediction = rfc.predict(cv_test_matrix)
rfc_accuracy = accuracy_score(test_label_nums, rfc_prediction)
print("Accuracy of RandomForestClassifier: " + str(rfc_accuracy))
# --------
# Task 6 - GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    # 'min_samples_leaf': [1, 2, 4],
    # 'bootstrap': [True, False],
    'warm_start': [True]
}


grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(cv_train_matrix, train_label_nums)

best_params = grid_search.best_params_
print("Best hyperparameters for Random Forest Classifier:", best_params)

best_rfc_classifier = grid_search.best_estimator_
best_rfc_predictions = best_rfc_classifier.predict(cv_test_matrix)
best_rfc_accuracy = accuracy_score(test_label_nums, best_rfc_predictions)
print("Accuracy of RandomForestClassifier with best hyperparameters:", best_rfc_accuracy)
# --------
