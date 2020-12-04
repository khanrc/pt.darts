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


from pathlib import Path


def processTxt():
    folder_name = "/home2/lgfm95/hem/perceptual/tempSave/mnistclusters/"
    badpoints_files = sorted(Path(folder_name).iterdir(), key=os.path.getmtime)
    print(badpoints_files)


if __name__ == "__main__":
    # main()
    processTxt()

# if not getting high val before a new bar, then mastery too low

# if val lightning not shallow, then perhaps mastery marginally too high as overfitting to previous dataset

# if train lightning shallow, then have already mastered. therefore should be switching out quickly.
# if not switching out quickly, hardness might be too high, as it is retaining too many images