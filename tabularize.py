import os
import re
import matplotlib.pyplot as plt

def main():
    if os.path.exists("./totab.out"):
        fp = open("./totab.out", "r")
        lines = fp.readlines()
        x = []
        y = []
        z = []
        c = []
        for line in lines:
            row = line.split(" ")
            dotIdx = row[0].index(".out")
            row[0] = row[0][:dotIdx]
            headers = row[0].split("aa")
            digits = re.search(r'\d', headers[0])
            headers[0] = headers[0][digits.start():]
            print(headers, row[-1])
            x.append(headers[0])
            y.append(headers[1])
            z.append(headers[2])
            c.append(row[-1])

        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);

if __name__ == '__main__':
    main()
