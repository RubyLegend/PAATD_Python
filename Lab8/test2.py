#!/usr/bin/env python3
import json

import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load("trained_model2")

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
