import os
import re


def main():
    if os.path.exists("./totab.out"):
        fp = open("./totab.out", "r")
        lines = fp.readlines()
        for line in lines:
            row = line.split(" ")
            headers = row[0].split("aa")
            for header in headers:
                print(header)


if __name__ == '__main__':
    main()
