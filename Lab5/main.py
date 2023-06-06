#!/usr/bin/env python3

import re
import csv
import numpy as np

from matplotlib import pyplot as plt

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords, gutenberg
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

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

if skip_generation is False:
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
# print(len(prepared_corpus))

# Limiting corpus (for faster results)

prepared_corpus = prepared_corpus[:len(prepared_corpus)]

# --------
# Creating id2word dictionary

norm_papers = [sentence.split() for sentence in prepared_corpus]

bigram = gensim.models.Phrases(norm_papers, min_count=10, threshold=20, delimiter='_')
bigram_model = gensim.models.phrases.Phraser(bigram)

print(bigram_model)

norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)

print(dictionary)

# --------
# Creating BOW (Bag Of Words)

bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]

# --------
# Task 1. Latent Dirichlet Allocation
if 1:
    print("LDA Gensim:")
    Total_topics = 10
    lda_model = gensim.models.LdaModel(
        corpus = bow_corpus,       # +
        id2word = dictionary,      # +
        chunksize=1416,            # +
        alpha = 'auto',            # +
        eta = 'auto',              # +
        random_state = 0,          # +
        iterations=500,            # +
        num_topics = Total_topics, # +
        passes = 20,               # +
        eval_every=None            # +
    )
    
    for topic_id, topic in lda_model.print_topics(num_topics=10, num_words=20):
        print('Topic #'+str(topic_id+1)+':')
        print(topic)
    
    print("Coherence: ", end='')
    topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
    avg_coherence_score = np.mean([item[1] for item in topics_coherences])
    print(avg_coherence_score)
    ## --------
    ## Analysing coherence
    cv_coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model, corpus=bow_corpus, texts=norm_corpus_bigrams,
        dictionary=dictionary, coherence='c_v')
    avg_coherence_cv = cv_coherence_model_lda.get_coherence()
    print("Avg. Coherence Score (CV): " + str(avg_coherence_cv))
    
    umass_coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model, corpus=bow_corpus, texts=norm_corpus_bigrams,
        dictionary=dictionary, coherence='u_mass')
    avg_coherence_umass = umass_coherence_model_lda.get_coherence()
    print("Avg. Coherence Score (U-Mass): " + str(avg_coherence_umass))
    
    perplexity = lda_model.log_perplexity(bow_corpus)
    print("Model Perplexity: " + str(perplexity))
# --------
# Now compare it to other Total_topics numbers
x = []
y1 = []
y2 = []
y3 = []
ans = input("Do you want to load already prepared corpus? (Y/N) ").lower()
skip_generation = False
if ans == 'y':
    try:
        with open("prepared_graph.txt", 'r') as file:
            line = []
            for l in file:
                line.append(l.strip())

            i = 0
            while line[i] != '---':
                y1.append(float(line[i]))
                i = i + 1

            i = i + 1
            while line[i] != '---':
                y2.append(float(line[i]))
                i = i + 1

            i = i + 1
            while line[i] != '---':
                y3.append(float(line[i]))
                i = i + 1

            for v in range(1, len(y1)+1):
                x.append(v)

        skip_generation = True
    except FileNotFoundError:
        print("Prepared corpus not found, generating it.")

print(x)
print(y1)
print(y2)
print(y3)

if skip_generation is False:
    for i in range(1, 20 +1) :
        print("i = " + str(i) + ":")
        x.append(i)
        lda_model = gensim.models.LdaModel(
            corpus = bow_corpus,       # +
            id2word = dictionary,      # +
            chunksize=1416,            # +
            alpha = 'auto',            # +
            eta = 'auto',              # +
            random_state = 0,          # +
            iterations=500,            # +
            num_topics = i,            # +
            passes = 20,               # +
            eval_every=None            # +
        )
        
        print("Coherence: ", end='')
        topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
        avg_coherence_score = np.mean([item[1] for item in topics_coherences])
        print(avg_coherence_score)
        y1.append(avg_coherence_score)
        ## --------
        ## Analysing coherence
        cv_coherence_model_lda = gensim.models.CoherenceModel(
            model=lda_model, corpus=bow_corpus, texts=norm_corpus_bigrams,
            dictionary=dictionary, coherence='c_v')
        avg_coherence_cv = cv_coherence_model_lda.get_coherence()
        print("Avg. Coherence Score (CV): " + str(avg_coherence_cv))
        y2.append(avg_coherence_cv)
    
        umass_coherence_model_lda = gensim.models.CoherenceModel(
            model=lda_model, corpus=bow_corpus, texts=norm_corpus_bigrams,
            dictionary=dictionary, coherence='u_mass')
        avg_coherence_umass = umass_coherence_model_lda.get_coherence()
        print("Avg. Coherence Score (U-Mass): " + str(avg_coherence_umass))
        y3.append(avg_coherence_umass)
    
        perplexity = lda_model.log_perplexity(bow_corpus)
        print("Model Perplexity: " + str(perplexity))
            
    with open("prepared_graph.txt", 'w') as file2:
        for y in y1:
            file2.write(str(y) + '\n')
        file2.write("---\n")
        for y in y2:
            file2.write(str(y) + '\n')
        file2.write("---\n")
        for y in y3:
            file2.write(str(y) + '\n')
        file2.write("---\n")

fig, axs = plt.subplots(2)
axs[0].plot(x, y2, marker = 'o')
axs[1].plot(x, y3, marker = 'o')
plt.show()

# --------
# Task 2. NLTK Gutenberg Trigrams
print("Gutenberg trigrams:")

gutenberg_sents = gutenberg.sents(fileids="chesterton-thursday.txt")
gutenberg_sents = [' '.join(gutenberg_sents[i]) for i in range(len(gutenberg_sents))]

prepared_corpus = []
wpt = WordPunctTokenizer()

def preprocess_sentence(sentence):
    sentence = re.sub(r'[^a-zA-Z\s]+', '', sentence, re.I | re.A)
    sentence = sentence.lower()
    sentence = sentence.strip()
    tokens = wpt.tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    sentence = filtered_tokens
    return sentence

for sentence in gutenberg_sents:
    prepared_corpus.append(preprocess_sentence(sentence))

finder = TrigramCollocationFinder.from_documents(prepared_corpus)
trigram_measures = TrigramAssocMeasures()

print("Top 10 raw frequency: ", end='')
print(finder.nbest(trigram_measures.raw_freq, 10))

print("Top 10 PMI: ", end='')
print(finder.nbest(trigram_measures.pmi, 10))

print("Top 10 Likelihood Ratio: ", end='')
print(finder.nbest(trigram_measures.likelihood_ratio, 10))

# --------
