import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()
    resize_transform = None
    if dataset == 'cifar10':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        resize_transform = [transforms.Resize(64)]
        transf = [
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'imagenet' or dataset == 'imageobj' or dataset == "cocomask" or dataset == "pure_det":
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        resize_transform = [transforms.Resize((128,128))]
        transf = [
            transforms.Resize((224,224)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
        MEAN_lbl = 0.2807
        STD_lbl = 0.1765
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    elif dataset == 'planes':
        MEAN = [0.4791, 0.5107, 0.5351]
        STD = [0.1704, 0.1696, 0.1942]
        resize_transform = [transforms.Resize((128,128))]
        transf = [
            transforms.Resize((224,224)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    elif dataset == 'cityscapes':
        MEAN = [0.2807, 0.3216, 0.2829]
        STD = [0.1765, 0.1800, 0.1748]
        MEAN_lbl = 0.2807
        STD_lbl = 0.1765
        transf = [
            transforms.Resize((64, 64)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    if dataset == "cityscapes" or dataset == "cocomask":
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_lbl, STD_lbl)
        ]
        valid_transform = transforms.Compose(transf + normalize)
    else:
        if resize_transform is not None:
            valid_transform = transforms.Compose(resize_transform + normalize)
        else:
            valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform
