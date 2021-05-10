import csv
import os


def load_csv(dataset, epoch):
    file_name = f"/home2/lgfm95/nas/darts/tempSave/curriculums/indices_{dataset}_{epoch}.csv"
    with open(file_name, "r") as fp:
        csvreader = csv.reader(fp, delimeter=' ')
        return [idx for idx in csvreader]


def load_all(dataset):
    epoch_dict = {}
    for file in os.listdir(f"/home2/lgfm95/nas/darts/tempSave/curriculums_{dataset}*.csv"):
        print(file)
        slashIdx = file.rfind("/")
        suffix = file[slashIdx:][:-4]
        print(suffix)
        elems = suffix.split("_")
        epoch = elems[2]
        print(epoch)
        epoch_dict[epoch] = load_csv(dataset, epoch)

    return epoch_dict


if __name__ == "__main__":
    load_all("mnist")
