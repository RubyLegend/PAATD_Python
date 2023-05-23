#!/usr/bin/env python3

import re
import csv
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords, gutenberg
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF

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
# print(len(prepared_corpus))

# Limiting corpus (for faster results)

prepared_corpus = prepared_corpus[:len(prepared_corpus)//4]

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
# Comparing to other models
# LSI Gensim
print("LSI Gensim:")
lsi_bow = gensim.models.LsiModel(
    bow_corpus,
    id2word = dictionary,
    num_topics = Total_topics,
    onepass = True,
    chunksize = 1416,
    power_iters = 500
)

for topic_id, topic in lsi_bow.print_topics(num_topics=10, num_words=20):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)
    
## --------
## Analysing coherence
cv_coherence_model_lda = gensim.models.CoherenceModel(
    model=lsi_bow, corpus=bow_corpus, texts=norm_corpus_bigrams,
    dictionary=dictionary, coherence='c_v')
avg_coherence_cv = cv_coherence_model_lda.get_coherence()
print("Avg. Coherence Score (CV): " + str(avg_coherence_cv))

umass_coherence_model_lda = gensim.models.CoherenceModel(
    model=lsi_bow, corpus=bow_corpus, texts=norm_corpus_bigrams,
    dictionary=dictionary, coherence='u_mass')
avg_coherence_umass = umass_coherence_model_lda.get_coherence()
print("Avg. Coherence Score (U-Mass): " + str(avg_coherence_umass))

# perplexity = lsi_bow.log_perplexity(bow_corpus)
print("Model Perplexity: N/A")
# --------
# LSI sklearn
print("LSI sklearn:")
# BOW sklearn
cv = CountVectorizer(min_df=20, max_df=0.6, ngram_range=(1,2),
token_pattern=None, tokenizer=lambda doc: doc,
preprocessor=lambda doc: doc)
cv_features = cv.fit_transform(norm_papers)
vocabulary = np.array(cv.get_feature_names_out())

lsi_model = TruncatedSVD(
    n_components = Total_topics,
    n_iter = 500,
    random_state = 0
)
document_topics = lsi_model.fit_transform(cv_features)

topic_terms = lsi_model.components_
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms),axis=1)[:, :10]
topic_keyterms = vocabulary[topic_key_term_idxs]
topics = [', '.join(topic) for topic in topic_keyterms]
pd.set_option('display.max_colwidth', None)
topics_df = pd.DataFrame(
    topics, 
    columns = ['Terms per Topic'],
    index=['Topic'+str(t) for t in range(1, Total_topics+1)]
)
print(topics_df)

# --------
# LDA sklearn
print("LDA sklearn:")
lda_model2 = LatentDirichletAllocation(
    n_components = Total_topics,
    max_iter = 500,
    max_doc_update_iter = 50,
    learning_method = 'online',
    batch_size = 1416,
    learning_offset = 50.,
    random_state = 0,
    n_jobs = -1 # All cpu power
)
document_topics2 = lda_model2.fit_transform(cv_features)

topic_terms = lda_model2.components_
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms),axis=1)[:, :10]
topic_keyterms = vocabulary[topic_key_term_idxs]
topics = [', '.join(topic) for topic in topic_keyterms]
pd.set_option('display.max_colwidth', None)
topics_df = pd.DataFrame(
    topics, 
    columns = ['Terms per Topic'],
    index=['Topic'+str(t) for t in range(1, Total_topics+1)]
)
print(topics_df)

print("LDA sklearn perplexity: " + str(lda_model2.perplexity(cv_features)))
print("LDA sklearn score: " + str(lda_model2.score(cv_features)))
# --------
# NMF sklearn
print("NMF sklearn:")
nmf_model = NMF(
    n_components = Total_topics,
    solver = 'cd',
    max_iter = 500,
    random_state = 0,
    # alpha_W = .1,
    l1_ratio = .85,
    init = "nndsvd"
)
document_topics3 = nmf_model.fit_transform(cv_features)

topic_terms = nmf_model.components_
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms),axis=1)[:, :10]
topic_keyterms = vocabulary[topic_key_term_idxs]
topics = [', '.join(topic) for topic in topic_keyterms]
pd.set_option('display.max_colwidth', None)
topics_df = pd.DataFrame(
    topics, 
    columns = ['Terms per Topic'],
    index=['Topic'+str(t) for t in range(1, Total_topics+1)]
)
print(topics_df)

# --------
# Task 2. NLTK Gutenberg Trigrams
print("Gutenberg trigrams:")

gutenberg_sents = gutenberg.sents(fileids="chesterton-thursday.txt")

finder = TrigramCollocationFinder.from_documents(gutenberg_sents)
trigram_measures = TrigramAssocMeasures()

print("Top 10 raw frequency: ", end='')
print(finder.nbest(trigram_measures.raw_freq, 10))

print("Top 10 PMI: ", end='')
print(finder.nbest(trigram_measures.pmi, 10))

print("Top 10 Likelihood Ratio: ", end='')
print(finder.nbest(trigram_measures.likelihood_ratio, 10))

# --------
