import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def main():
    if os.path.exists("./totab.out"):
        fp = open("./totab.out", "r")
        lines = fp.readlines()
        x = []
        y = []
        z = []
        c = []
        cmap = cm.inferno
        norm = Normalize(vmin=-20, vmax=10)

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
            print(cmap(offset_color(row[-1][:-2])))
            c.append(cmap(offset_color(row[-1][:-2])))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(x, y, z, c=c, cmap='viridis', linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xlabel('subset size')
        ax.set_yticks(list(y))
        ax.set_ylabel('hardness')
        ax.set_zticks(list(z))
        ax.set_zlabel('mastery')
        fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)


        plt.savefig("cube.png")


def offset_color(num):
    return ((float(num)-80)*5)/100


if __name__ == '__main__':
    main()
