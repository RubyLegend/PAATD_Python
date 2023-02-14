#!/usr/bin/env python3
file_text = ""

with open("text3.txt", "r", encoding="utf-8") as file:
    file_text = file.read()

if __name__ == "__main__":
    print("Content:")
    print(file_text)
    print("----------")
    print("File length: ", end="")
    print(len(file_text))
