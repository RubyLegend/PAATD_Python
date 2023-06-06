#!/usr/bin/env python3 

import spacy
from spacy.matcher import Matcher

text = ""
with open("text3.txt", "r") as file:
    for line in file:
        # Normalize some phone numbers
        line = line.replace(')', ' )').replace('(', '( ').replace('-', ' - ')
        text = text + line.strip()

text2 = ""
with open("lab7-3.txt", "r") as file:
    for line in file:
        text2 = text2 + line.strip()

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
doc2 = nlp(text2)
print([token.text for token in doc])
print([token.text for token in doc2])

# Task 1. Match phone numbers with Matcher
matcher = Matcher(nlp.vocab)
pattern_phone = [
                    [
                    {'TEXT': '(', 'OP': '+'}, 
                    {'TEXT': {'REGEX': '^\\d{2,3}$'}},
                    {'TEXT': ')', 'OP': '+'},
                    {'TEXT': '-', 'OP': '?'},
                    {'TEXT': {'REGEX': '^\\d{2,3}$'}},
                    {'TEXT': '-', 'OP': '?'},
                    {'TEXT': {'REGEX': '^\\d{2,3}$'}}
                    ],
                    [
                    # {'TEXT': '(', 'OP': '?'}, 
                    {'TEXT': {'REGEX': '^\\d{2,3}'}},
                    # {'TEXT': ')', 'OP': '?'},
                    {'TEXT': '-', 'OP': '?'},
                    {'TEXT': {'REGEX': '^\\d{2,3}$'}},
                    {'TEXT': '-', 'OP': '?'},
                    {'TEXT': {'REGEX': '^\\d{2,3}$'}}
                    ]
                 ]
matcher.add('phoneNum', pattern_phone)
matches = matcher(doc)
for _, start, end in matches:
    m_span = doc[start:end]
    print(start, end, '\t', m_span.text)
# -----------
# Task 2.
# Find and display stop-words, which are present
print("Stop words:")
for token in doc2:
    if token.is_stop:
        print(token.text + ", ", end='')

print()

# Find and display all nouns, which are present
print("Nouns:")
for token in doc2:
    if token.tag_ == 'NN':
        print(token, end=', ')

print()

# Find and display all numbers and organizations, which are present
print("Numbers and organizations:")
ner_tagged = [(word.text, word.ent_type_) for word in doc2]
for tag in ner_tagged:
    if tag[1] == 'ORG' or tag[1] == 'CARDINAL':
        print(tag)
