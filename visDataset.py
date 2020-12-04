import os


def main():
    path_name = "/home2/lgfm95/nas/darts/accuraciesFashion.out"
    if os.path.isfile(path_name):
        with open(path_name, "r") as f:
            data = f.readlines()
            accs = []
            for datum in data:
                bar_idx = str(datum).index("|")
                suffix = datum[bar_idx:]
                spl = suffix.split(" ")
                tv = spl[0]
                acc = spl[-1]
                # acc = float(acc.replace("%\n", ""))
                accs.append([tv,acc])
            print(accs)


if __name__ == "__main__":
    main()
