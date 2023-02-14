#!/usr/bin/env python3
file_text = ""

with open("text3.txt", "r", encoding="utf-8") as file:
    file_text = file.read()

if __name__ == "__main__":
    text_splice = file_text[0:20]
    print("Splice: " + text_splice)
    print("Splice length: " + str(len(text_splice)))
    print("Number of a's in text: " + str(text_splice.count('a')))
    find_res = text_splice.find('q')
    print("Position of 'q' in text: " + str(find_res if find_res != -1 else "Not found"))
    print("Upper case: " + text_splice.upper())
