#!/usr/bin/env python3

import json
import random

import spacy
from spacy.training import Example

model = "trained_model"

# load the model
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.load("en_core_web_md")
    print("Loaded en_core_web_md model")

with open("own_set.json", "r") as file:
    examples = json.loads(file.read())

n_iter = 20

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.create_optimizer()
    for itn in range(n_iter):
        random.shuffle(examples)
        losses = {}
        for example in examples:
            text = example["utterance"]
            doc = nlp.make_doc(text)
            annotations = {"entities": []}
            for ent in example["entities"]:
                annotations["entities"].append(
                    (
                        ent["start"],
                        ent["end"],
                        example["intent"],
                    )
                )
            # print(annotations)
            ex = Example.from_dict(doc, annotations)
            nlp.update([ex], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)

nlp.to_disk("trained_model")

text = (
    "I want to open new account. "
    + "Thank you for helping me to restore my account. "
    + "Can I create new savings account?"
    + "How to restore my password? "
    + "I cannot access my account"
)

doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
