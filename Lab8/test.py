#!/usr/bin/env python3

import spacy

nlp = spacy.load("trained_model")

text = (
    "I want to open new account. "
    + "Thank you for helping me to restore my account. "
    + "Can I create new savings account?"
    + "How to restore my password? "
    + "I cannot access my account"
)

doc = nlp(text)
print(doc.cats)
for ent in doc.ents:
    print(ent.text, ent.label_)
