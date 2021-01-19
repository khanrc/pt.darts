import os
import re


def main():
    if os.path.exists("./totab.out"):
        fp = open("./totab.out", "r")
        lines = fp.readlines()
        for line in lines:
            row = line.split(" ")
            print(row[0])
            dotIdx = row[0].index(".out")
            row[0] = row[0][:dotIdx]
            headers = row[0].split("aa")
            digits = re.search(r'\d', headers[0])
            headers[0] = headers[0][digits.start():]
            for header in headers:
                print(header)


if __name__ == '__main__':
    main()
