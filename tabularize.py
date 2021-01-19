import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
            # print(headers, row[-1])
            x.append(float(headers[0]))
            y.append(float(headers[1]))
            z.append(float(headers[2]))
            print(cm.hot(float(row[-1][:-2])))
            c.append(cm.hot(float(row[-1][:-2])))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(x, y, z, c=c, cmap='viridis', linewidth=0.5)
        plt.savefig("cube.png")

if __name__ == '__main__':
    main()
