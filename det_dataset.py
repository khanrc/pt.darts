import os
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as tf
import random
import torch

class Imagenet_Det:
    def __init__(self, root, transforms):
        # TODO train/test split
        self.annotation_path = os.path.join(root, "Annotations/DET/train/ILSVRC2014_train_0000") #TODO extend to other folders
        self.image_path = os.path.join(root, "Data/DET/train/ILSVRC2014_train_0000")

        self.transforms = transforms
        self.data = []
        self.fine = []

        self.classes = {}
        self.class_ids = 0
        self.classes_count = {}

        temp_count = 0

        for xml_path in sorted(os.listdir(self.annotation_path))[:20]:
            file_name = os.path.join(self.annotation_path, xml_path)
            # print(file_name)
            tree = ET.parse(file_name)
            root = tree.getroot()
            image_name = root.find("filename").text

            fine = []
            for object in root.iter("object"):
                class_id = object.find("name").text

                if class_id not in self.classes:
                    self.classes[class_id] = self.class_ids
                    self.classes_count[class_id] = 1
                    self.class_ids += 1
                else:
                    self.classes_count[class_id] = self.classes_count[class_id] + 1

                bbox = object.find("bndbox")
                x1, x2, y1, y2 = bbox.find("xmin").text, bbox.find("xmax").text, bbox.find("ymin").text, bbox.find("ymax").text
                fine.append([x1,y1,x2,y2])

            if len(fine) > 0:
                self.fine.append(fine)
                self.data.append(image_name)

            # temp_count += 1
            # if temp_count > 100:
            #     break

        for key, value in self.classes_count.items():
            print(self.classes[key], value)

        print(f"have only seen {len(self.classes.keys())} keys")

    def __len__(self):
        assert len(self.data) == len(self.fine)
        return len(self.data)

    def __getitem__(self, item):
        with Image.open(f"{self.image_path}/{self.data[item]}.JPEG") as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transforms is not None:
                image = self.transforms(image)
            # print("gt labels in imagenetDet", self.get_multihot(self.fine[item]))
            return image, self.fine[item]


def get_split(dataset):
    n_train = len(dataset)
    split = n_train * 0.8
    remainder = split % 8
    return int(split - remainder)


if __name__ == "__main__":
    normalize = tf.Normalize(
        mean=[0.13066051707548254],
        std=[0.30810780244715075])
    perc_transforms = tf.Compose([
        tf.RandomResizedCrop((224,224)),
        tf.ToTensor(),
        normalize,
    ])

    dset = Imagenet_Det(root="/hdd/PhD/data/imagenet2017detection", transforms=perc_transforms)
    split = get_split(dset)
    indices = list(range(len(dset)))
    random.seed(1337) # note must use same random seed as subloader (and thus process same images as subloader)
    random.shuffle(indices)
    # train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    split_loader = torch.utils.data.DataLoader(dset,
                                                    batch_size=1,
                                                    sampler=train_sampler,
                                                    num_workers=0,
                                                    pin_memory=False)

    idx_per_label = [None for i in range(184)]
    fine = []
    idx = [i for i in range(100)]
    print(idx)

    for q, (_, lab) in enumerate(split_loader):
        fine.append(lab)
        if q % 100 == 0:
            print(q)

    cur_labels = [fine[idx] for idx in idx]

    print(cur_labels[0])
    # for q, (im, label) in enumerate(split_loader):
    #     assert len(label) == 184
    #     raise AttributeError(type(label), len(label))
    #     for j in range(len(label)):  # 184
    #         lab = label[j]
    #         if lab == 1:
    #             if idx_per_label[j] is None:
    #                 idx_per_label[j] = [q]
    #             else:
    #                 idx_per_label[j].append(q)
    #     if q%100 == 0:
    #         print(q)
    # for i in range(184):
    #     print(i, len(idx_per_label[i]))