#!/usr/bin/env python3

import json
import random

import spacy
from spacy.training import Example
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from sklearn.model_selection import train_test_split

config = {"threshold": 0.5, "model": DEFAULT_SINGLE_TEXTCAT_MODEL}

model = None  # "trained_model2"

# load the model
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.load("en_core_web_md")
    print("Loaded en_core_web_md model")

# # set up the pipeline
# if "ner" not in nlp.pipe_names:
#     ner = nlp.create_pipe("ner")
#     nlp.add_pipe(ner, last=True)
# else:
#     ner = nlp.get_pipe("ner")

with open("banks.json", "r") as file:
    data = json.loads(file.read())

training_data = []
for dialogue in data:
    for turn in dialogue["turns"]:
        if turn["speaker"] == "USER":
            utterance = turn["utterance"]
            intent = None

            for frame in turn["frames"]:
                for action in frame["actions"]:
                    if action["act"] == "INFORM_INTENT":
                        intent = action["values"][0]
                        break

            if intent is not None:
                training_data.append((utterance, intent))

train_data, valid_data = train_test_split(
    training_data, test_size=0.4, random_state=1, shuffle=True
)
# train_data = training_data[:split]
# valid_data = training_data[split:]

n_iter = 10

if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", config=config)
    for _, intent in training_data:
        textcat.add_label(intent)

    train_examples = [
        Example.from_dict(nlp.make_doc(text), {"cats": {label: 1}})
        for text, label in train_data
    ]
    textcat.initialize(lambda: train_examples, nlp=nlp)
else:
    textcat = nlp.get_pipe("textcat")

with nlp.select_pipes(enable="textcat"):
    optimizer = nlp.resume_training()
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for text, label in train_data:
            # if label == "ReserveHotel":
            #     print(text, " | ", label)
            doc = nlp.make_doc(text)
            ex = Example.from_dict(doc, {"cats": {label: 1}})
            nlp.update([ex], sgd=optimizer, losses=losses)
        print(losses)

nlp.to_disk("trained_model2")

# Оцінка моделі на перевірочних даних
eval_results = {}

for text, label in valid_data:
    doc = nlp(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    print(doc, predicted_label)
    if predicted_label not in eval_results:
        eval_results[predicted_label] = {"tp": 0, "fp": 0, "fn": 0}
    if label not in eval_results:
        eval_results[label] = {"tp": 0, "fp": 0, "fn": 0}
    if predicted_label == label:
        eval_results[predicted_label]["tp"] += 1
    else:
        eval_results[predicted_label]["fp"] += 1
        eval_results[label]["fn"] += 1

# Виведення результатів оцінки
for label, metrics in eval_results.items():
    try:
        precision = metrics["tp"] / (metrics["tp"] + metrics["fp"])
    except ZeroDivisionError:
        precision = 0

    try:
        recall = metrics["tp"] / (metrics["tp"] + metrics["fn"])
    except ZeroDivisionError:
        recall = 0

    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    print(f"Label: {label}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")
