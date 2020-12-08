import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

bars = []

parser = argparse.ArgumentParser()
parser.add_argument('--out_file', default="/home2/lgfm95/nas/darts/accuraciesMnist.out", type=str, help='create graph')
parser.add_argument('--cluster_folder', default="/home2/lgfm95/hem/perceptual/tempSave/mnistclusters/", type=str, help='process text files')
args = None

def main():
    path_name = args.out_file
    if os.path.isfile(path_name):
        with open(path_name, "r") as f:
            data = f.readlines()
            train_accs = []
            val_accs = []
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

            legend = ax.legend(loc="best", title="train/val", bbox_to_anchor=(1.13, 1), ncol=1, fontsize='x-small')
            ax.add_artist(legend)
            fig.savefig("test.png")
            print(train_accs, val_accs, bars)


from pathlib import Path


def processTxt():
    folder_name = args.cluster_folder
    temp_badpoints_files = sorted(Path(folder_name).iterdir(), key=os.path.getmtime)
    badpoints_files = []
    for file in temp_badpoints_files:
        if "badpoints" in str(file):
            badpoints_files.append(file)

    sets = []
    for file in badpoints_files:
        with open(file, "r") as f:
            data = f.readlines()
            my_set = set(data)
            if len(my_set) != len(data):
                print("num unique", len(my_set), len(data))
            sets.append(my_set)


    print("how many got changed on update:")
    for epoch in bars:
        try:
            set1 = sets[epoch-1]
            set2 = sets[epoch]
            difference = len(set1-set2) + len(set2-set1)
            total = len(set1) + len(set2)
            print("difference after update at {} is {} out of {}".format(epoch, difference, total))
        except IndexError:
            pass


    print("how many flipflopped on update:")
    for i in range(1, len(bars)):
        try:
            set1 = sets[bars[i-1]-1]
            set2 = sets[bars[i]-1]
            difference = len(set1-set2) + len(set2-set1)
            total = len(set1) + len(set2)
            print("difference after update at {} is {} out of {}".format(epoch, difference, total))
            print("that percentage is {}".format(difference/total))
        except IndexError as e:
            print(e)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
    processTxt()

# if not getting high val before a new bar, then mastery too low

# if val lightning not shallow, then perhaps mastery marginally too high as overfitting to previous dataset

# if train lightning shallow, then have already mastered. therefore should be switching out quickly.
# if not switching out quickly, hardness might be too high, as it is retaining too many images

# if flipflopping difference/total is low, we have a lot of flipflops, implying hardness is too high