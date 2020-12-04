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
            bars = []
            count = 0
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
                        count += 1
                    elif "Valid:" in tv:
                        val_accs.append(acc)
                    else:
                        print("neither")
                except ValueError:
                    bars.append(count)

            assert len(train_accs) == len(val_accs), "len of train and val accuracies should be same"
            x_axis = [i for i in range(len(train_accs))]
            fig, ax = plt.subplots()
            ax.plot(x_axis, train_accs, color="green")
            ax.plot(x_axis, val_accs, color="red")

            for xc in bars:
                plt.axvline(x=xc, color='k', linestyle='--')

            fig.savefig("test.png")
            print(train_accs, val_accs, bars)


if __name__ == "__main__":
    main()
