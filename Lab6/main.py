#!/usr/bin/env python3

import re
import csv
import pandas as pd
import random

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

from textblob import TextBlob

nltk.data.path.append("../Lab2/nltk_data")

prepared_corpus = []
labels = []

skip_generation = False

ans = input("Do you want to load already prepared corpus? (Y/N) ").lower()

if ans == 'y':
    try:
        with open("prepared_corpus.txt", 'r') as file:
            for line in file:
                prepared_corpus.append(line.strip())
        with open("prepared_labels.txt", 'r') as file:
            for line in file:
                labels.append(line.strip())
        skip_generation = True
    except FileNotFoundError:
        print("Prepared corpus not found, generating it.")

if skip_generation is False:
    file = open("twitter1.csv", 'r')
    csv = csv.DictReader(file, delimiter=',', quotechar='"', fieldnames=['number','topic','sentiment','content'])
    texts = {i: v for i, v in enumerate(csv)}

    wpt = WordPunctTokenizer()

    # Preparing corpus
    corpus = [texts[i]['content'] for i in range(len(texts))]


    def preprocess_sentence(sentence):
        sentence = re.sub(r'[^a-zA-Z\s]+', '', sentence, re.I | re.A)  # Links are still present in array
        sentence = sentence.lower()
        sentence = sentence.strip()
        tokens = wpt.tokenize(sentence)
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        sentence = ' '.join(filtered_tokens)
        return sentence


    i = 0
    for sentence in corpus:
        sentence_processed = preprocess_sentence(sentence)
        if len(sentence_processed) != 0:
            labels.append(texts[i]['sentiment'])
            prepared_corpus.append(sentence_processed)
        i = i+1

    with open("prepared_corpus.txt", 'w') as file2:
        for sentence in prepared_corpus:
            file2.write(sentence + '\n')

    with open("prepared_labels.txt", 'w') as file2:
        for label in labels:
            file2.write(label + '\n')

print("Підготовлений корпус: ")
print(prepared_corpus[len(prepared_corpus)-50:])
print(len(prepared_corpus))

# Limiting corpus (for faster results)

prepared_corpus = prepared_corpus[:len(prepared_corpus)]

# Task 1 - Split data into train and test
train_corpus, test_corpus, train_sentiments, test_sentiments, \
    = train_test_split(prepared_corpus, labels,
                       test_size=0.4, random_state=0)
# --------
# Bag of Words
cv = CountVectorizer(binary=False, min_df=0., max_df=1.)
cv_train_matrix = cv.fit_transform(train_corpus)
cv_test_matrix = cv.transform(test_corpus)
vocab = cv.get_feature_names_out()

array = pd.DataFrame(cv_train_matrix.toarray(), columns=vocab)

print(array)
# --------
# NaiveBayesClassifier
mnb = MultinomialNB(alpha=1)
mnb.fit(cv_train_matrix, train_sentiments)
# --------
# Predict sentiments
mnb_prediction = mnb.predict(cv_test_matrix)
mnb_prediction2 = mnb.predict(cv_train_matrix)
mnb_accuracy = accuracy_score(test_sentiments, mnb_prediction)
mnb_accuracy2 = accuracy_score(train_sentiments, mnb_prediction2)
print("Accuracy of MultinomialNB: " + str(mnb_accuracy))
print("Accuracy of MultinomialNB on train data: " + str(mnb_accuracy2))
# --------
# Confusion matrix
table_labels = ["Positive", "Negative", "Neutral", "Irrelevant"]
matrix1 = confusion_matrix(test_sentiments, mnb_prediction, labels=table_labels)
array = pd.DataFrame(matrix1, columns=table_labels, index=table_labels)

print(array, end="\n\n")

print("Accuracy of Positive: " + str(matrix1[0][0] / matrix1[0].sum()))
print("Accuracy of Negative: " + str(matrix1[1][1] / matrix1[1].sum()))
print("Accuracy of Neutral: " + str(matrix1[2][2] / matrix1[2].sum()))
print("Accuracy of Irrelevant: " + str(matrix1[3][3] / matrix1[3].sum()))
print()
# --------
# TextBlob

sentiments_polarity = [TextBlob(sentence).sentiment.polarity for sentence in test_corpus]
predicted_sentiments = ['Positive' if score >= 0.1 
                        else "Negative" if score <= -0.1 
                        else "Neutral"
                        for score in sentiments_polarity]



table_labels = ["Positive", "Negative", "Neutral"]
sentiments_without_irrelevant = [label if label != "Irrelevant" else "Neutral" for label in test_sentiments]

blob_accuracy = accuracy_score(sentiments_without_irrelevant, predicted_sentiments)
print("Accuracy of TextBlob: " + str(blob_accuracy))

matrix2 = confusion_matrix(sentiments_without_irrelevant, predicted_sentiments, labels=table_labels)
array2 = pd.DataFrame(matrix2, columns=table_labels, index=table_labels)
print(array2)
print("Accuracy of Positive: " + str(matrix2[0][0] / matrix2[0].sum()))
print("Accuracy of Negative: " + str(matrix2[1][1] / matrix2[1].sum()))
print("Accuracy of Neutral: " + str(matrix2[2][2] / matrix2[2].sum()))
print()

# Compare 3 records
rec = random.sample(range(1, len(test_sentiments)+1), 3)

print("Compare MultinomialNB:")
for record in rec:
    print("Post: " + test_corpus[record])
    print("Actual sentiment: " + test_sentiments[record])
    print("Predicted: " + mnb_prediction[record])
    print("----------")

print()

print("Compare TextBlob:")
for record in rec:
    print("Post: " + test_corpus[record])
    print("Actual sentiment: " + sentiments_without_irrelevant[record])
    print("Predicted: " + predicted_sentiments[record])
    print("----------")

