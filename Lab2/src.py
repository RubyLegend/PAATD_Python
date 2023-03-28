#!/usr/bin/env python3

from sty import fg
import colorsys

import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import brown

# nltk.download('all', download_dir="./nltk_data")
nltk.data.path.append("./nltk_data")


def h_to_color(h):
    (r, g, b) = colorsys.hsv_to_rgb(h, 1, 1)
    (r, g, b) = (int(r * 255), int(g * 255), int(b * 255))
    return fg(r, g, b)


if __name__ == "__main__":
    with open("text1.txt") as file:
        text = file.read()
        print(text, end="\n---------------------\n")

        print("Кількість речень в тексті: " + str(len(sent_tokenize(text))))

        print(sent_tokenize(text)[-1])

        pos_tags = dict()
        for item in nltk.pos_tag(word_tokenize(text)):
            try:
                pos_tags[item[1]] += 1
            except KeyError:
                pos_tags[item[1]] = 1

        pos_tags_len = len(pos_tags)
        list_pos_tags = list(pos_tags)

        print("Color guide:")
        for tag_index in range(len(list_pos_tags)):
            print(
                h_to_color(tag_index / pos_tags_len) + list_pos_tags[tag_index] + fg.rs,
                end="\t",
            )

        print("\n")
        for word, tag in nltk.pos_tag(word_tokenize(text)):
            index = list_pos_tags.index(tag)
            print(h_to_color(index / pos_tags_len) + word + fg.rs + " ", end="")

        print("\n")

        print("Частини мови: " + str(nltk.pos_tag(word_tokenize(text))), end="\n\n")

        freqDist = nltk.FreqDist(word_tokenize(text))
        print("Частота зустрічі слів: " + str(freqDist.most_common(10)), end="\n\n")

        print("---------------------\n")
        print(
            "Кількість слів в категорії science_fiction: "
            + str(len(brown.words(categories="science_fiction"))),
            end="\n\n",
        )

        # print(brown.fileids(categories='science_fiction'))
        sents = brown.tagged_words(fileids="cm02")
        # print(sents)
        regex = re.compile(r"^V.*")
        # print(len(' '.join([''.join(sent_tokens) for sent_tokens, tags in sents])))
        # print(len(' '.join([''.join(sent_tokens) for sent_tokens, tags in sents if not regex.search(tags)])))
        print("Другий текст без дієслів: ")
        print(
            " ".join(
                [
                    "".join(sent_tokens)
                    for sent_tokens, tags in sents
                    if not regex.search(tags)
                ]
            )
        )
