import os


def main():
    path_name = "/home2/lgfm95/nas/darts/accuraciesFashion.out"
    if os.path.isfile(path_name):
        with open(path_name, "r") as f:
            data = f.readlines()
            accs = []
            for datum in data:
                idx = str(datum).index("@")
                suffix = datum[idx:]
                spl = suffix.split(" ")
                acc = int(spl[2].replace("\n", ""))
                accs.append(acc)
            print(accs)


if __name__ == "__main__":
    main()
