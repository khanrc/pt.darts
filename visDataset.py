import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    path_name = "/home2/lgfm95/nas/darts/accuraciesFashion.out"
    if os.path.isfile(path_name):
        with open(path_name, "r") as f:
            data = f.readlines()
            train_accs = []
            val_accs = []
            for datum in data:
                try:
                    bar_idx = str(datum).index("|")
                    suffix = datum[bar_idx:]
                    spl = suffix.split(" ")
                    tv = spl[1]
                    acc = spl[-1]
                    acc = float(acc.replace("%\n", ""))
                    if "Train:" in tv:
                        train_accs.append(acc)
                    elif "Valid:" in tv:
                        val_accs.append(acc)
                    else:
                        print("neither")
                except ValueError:
                    pass

            assert len(train_accs) == len(val_accs), "len of train and val accuracies should be same"
            x_axis = [i for i in range(len(train_accs))]
            fig, ax = plt.subplots()
            ax.plot(x_axis, train_accs, color="green")
            ax.plot(x_axis, val_accs, color="red")
            fig.savefig("test.png")
            print(train_accs, val_accs)


if __name__ == "__main__":
    main()
