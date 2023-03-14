#!/usr/bin/env python3
import re

file_text = ""

with open("text3.txt", "r", encoding="utf-8") as file:
    file_text = file.read()


def task_1():
    print("Task 1")
    print("----------------------------------")
    text_splice = file_text[0:20]
    print("Splice: " + text_splice)
    print("Splice length: " + str(len(text_splice)))
    print("Number of a's in text: " + str(text_splice.count("a")))
    find_res = text_splice.find("q")
    print(
        "Position of 'q' in text: " + str(find_res if find_res != -1 else "Not found")
    )
    print("Upper case: " + text_splice.upper())
    print("Replaced 'Slowly' with 'Fastly': " + text_splice.replace("Slowly", "Fastly"))
    print("Split string by symbol ' ': ", end="")
    print(text_splice.split(" "))
    print("Join previously splitted strings: " + "".join(text_splice.split(" ")))


def task_2():
    global file_text
    print("----------------------------------")
    print("Task 2")
    print("Variant 11")
    print("----------------------------------")
    print("Whole file content:")
    print(file_text)
    print("Preparing regex...")
    print("----------------------------------")
    n = re.compile(
        r"(?<=(?<=^)|(?<=[^\w(]))\+?\(?\d{2,3}\)?(?:\-\d{1,3}){2,3}(?=(?<=$)|(?=[^\w-]))"
    )
    print("Blurring phone numbers...")
    start = 0
    end = len(file_text)

    while start != end:
        res = n.search(file_text, start, end)

        if res is None:
            break

        startPos = res.span()[0]
        endPos = res.span()[1]
        numCount = 0

        while startPos != endPos:
            if file_text[startPos].isnumeric():
                if numCount == 2:
                    file_text = file_text[:startPos] + "X" + file_text[startPos + 1:]
                else:
                    numCount = numCount + 1
            startPos = startPos + 1

        start = endPos

    print("Done.")
    print("----------------------------------")
    print(file_text)


if __name__ == "__main__":
    task_1()
    task_2()
