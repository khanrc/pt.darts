""" Utilities """
import sys
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
from torchvision import transforms
import numpy as np
import preproc
from config import SearchConfig, AugmentConfig
sys.path.insert(0, "/home2/lgfm95/hem/perceptual")
sys.path.insert(0, "C:\\Users\\Matt\\Documents\\PhD\\x11\\HEM\\perceptual")
sys.path.insert(0, "/hdd/PhD/hem/perceptual")
sys.path.insert(0, "/home/matt/Documents/hem/perceptual")
saved_name = sys.argv[sys.argv.index('--name')+1]
from dataloader_convert import DynamicDataset as DynamicObj
from dataloader_classification import DynamicDataset as DynamicClass
from subloader import SubDataset
from tiny_imagenet import Tiny_ImageNet
from coco_obj import COCODetLoader as Coco_Det
sys.argv.insert(1, saved_name)
sys.argv.insert(1, "--name")
from sklearn.metrics import average_precision_score as ap


def get_data(dataset, data_path, cutout_length, validation, search, bede, is_concat):
    if search:
        config = SearchConfig()
    else:
        config = AugmentConfig()
    """ Get torchvision dataset """
    dataset = dataset.lower()

    pretrain_resume = "/home2/lgfm95/hem/perceptual/good.pth.tar"
    grayscale = False
    is_detection = False
    convert_to_paths = False
    convert_to_lbl_paths = False
    isize = 64
    nz = 8
    aisize = 3
    DynamicDataset = DynamicClass
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        dynamic_name = "cifar10"
        n_classes = 10
        # nz = 32
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercCifar10Good.pth.tar"
    elif dataset == 'imagenet':
        dset_cls = dset.ImageNet
        dynamic_name = "imagenet"
        n_classes = 1000
        if config.ncc:
            auto_resume = "/home2/lgfm95/hem/perceptual/ganPercImagenetGood.pth.tar"
        else:
            auto_resume = "/hdd/PhD/hem/perceptual/ganPercImagenetGood.pth.tar"
        isize = 256
        convert_to_paths = True
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
        dynamic_name = "mnist"
        grayscale = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercMnistGood.pth.tar"
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
        dynamic_name = "fashion"
        grayscale = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercFashionGood.pth.tar"
    elif dataset == 'planes':
        # dset_cls = dset.FashionMNIST
        n_classes = 70
        dynamic_name = "planes"
        grayscale = False
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercPlaneGood.pth.tar"
    elif dataset == 'cityscapes':
        # dset_cls = dset.FashionMNIST
        n_classes = 30
        dynamic_name = "cityscapes"
        grayscale = False
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercCityscapesGood.pth.tar"
    elif dataset == "imageobj":
        n_classes = 200
        dynamic_name = "imageobj"
        grayscale = False
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercObjGood.pth.tar"
        is_detection = True
        convert_to_paths = True
    elif dataset == "cocomask":
        n_classes = 80
        dynamic_name = "cocomask"
        grayscale = False
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercMaskGood.pth.tar"
        is_detection = True
        convert_to_paths = True
        # convert_to_lbl_paths = True
    elif dataset == "coco_det":
        n_classes = 91
        dynamic_name = "coco_det"
        grayscale = False
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercCocoOld.pth.tar"
        is_detection = True

        # coco_train_path = '/home2/lgfm95/coco/'
        coco_train_path = '/home/matt/Documents/coco/'
        # coco_train_path = '/hdd/PhD/data/coco/'
        DynamicDataset = DynamicObj
    elif dataset == "pure_det":
        n_classes = 200
        dynamic_name = "pure_det"
        grayscale = False
        is_detection = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercObjGood.pth.tar"
    elif dataset == "tiny_image":
        n_classes = 200
        dynamic_name = "tiny_image"
        grayscale = False
        is_detection = True
        # auto_resume = "/hdd/PhD/hem/perceptual/ganPercTiny.pth.tar"
        auto_resume = "/home/matt/Documents/hem/perceptual/ganPercTiny.pth.tar"
    else:
        raise ValueError(dataset)

    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
    perc_transforms = transforms.Compose([
        transforms.Resize((isize, isize)),
        transforms.ToTensor(),
        normalize,
    ])
    if config.badpath:
        auto_resume = "badpath"

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    if config.dynamic:
        # print(perc_transforms)
        trn_data = DynamicDataset(
            perc_transforms=perc_transforms,
            pretrain_resume=pretrain_resume,
            image_transforms=trn_transform,
            val_transforms=val_transform,
            val=False,
            dataset_name=dynamic_name,
            auto_resume=auto_resume,
            hardness=config.hardness,
            isize=isize,
            nz=nz,
            aisize=aisize,
            grayscale=grayscale,
            isTsne=True,
            tree=config.isTree,
            subset_size=config.subset_size,
            is_csv=config.is_csv,
            is_detection=is_detection,
            is_concat=is_concat)
            # is_csv=False)
        input_size = len(trn_data)
        input_channels = 3 if len(trn_data.bands) == 3 else 1 # getbands() gives rgb if rgb, l if grayscale
    else:
        if config.vanilla:
            if dataset == 'imagenet' or dataset == "imageobj" or dataset == "pure_det":
                trn_data = SubDataset(transforms=trn_transform, val_transforms=val_transform, val=False,
                                      dataset_name=dynamic_name)
                input_size = len(trn_data)
                input_channels = 3 if len(trn_data.bands) == 3 else 1 # getbands() gives rgb if rgb, l if grayscale
            elif dataset == "coco_det":
                trn_data = Coco_Det(train_path=coco_train_path, transforms=val_transform)
                input_size = len(trn_data)
                input_channels = 3
            else:
                trn_data = dset_cls(root=data_path, train=True, download=False, transform=trn_transform)        # # assuming shape is NHW or NHWC
                shape = trn_data.data.shape
                input_channels = 3 if len(shape) == 4 else 1
                assert shape[1] == shape[2], "not expected shape = {}".format(shape)
                input_size = shape[1]
        else:
            if dataset == "coco_det":
                subset_size = 10000
                if search:
                    subset_size = config.subset_size
                trn_data = Coco_Det(train_path=coco_train_path, transforms=val_transform, max_size=subset_size)
                input_size = len(trn_data)
                input_channels = 3
            else:
                subset_size = 10000
                if search:
                    subset_size = config.subset_size
                trn_data = SubDataset(transforms=trn_transform, val_transforms=val_transform, val=False, dataset_name=dynamic_name, subset_size=subset_size)

                input_size = len(trn_data)
                input_channels = 3 if len(trn_data.bands) == 3 else 1 # getbands() gives rgb if rgb, l if grayscale


    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        # if config.dynamic:
        #     val_dataset = DynamicDataset(
        #         perc_transforms=perc_transforms,
        #         pretrain_resume=pretrain_resume,
        #         image_transforms=val_transform, val=True,
        #         dataset_name=dynamic_name,
        #         auto_resume=auto_resume,
        #         isize=isize,
        #         nz=nz,
        #         aisize=aisize,
        #         grayscale=grayscale,
        #         isTsne=True,
        #         tree=config.isTree)
        #     ret.append(val_dataset)
        # else:
        if dataset == 'coco_det':
            # ret.append(SubDataset(transforms=val_transform, val=True, dataset_name="coco_det", bede=bede))

            # train_path = '/home2/lgfm95/coco/'
            # train_path = '/home/matt/Documents/coco/'
            ret.append(Coco_Det(train_path=coco_train_path, transforms=val_transform))
        elif dataset == "tiny_image":
            print("using tiny imagenet")
            # train_path = '/home2/lgfm95/tiny-imagenet-200/'
            train_path = '/data/tiny-imagenet-200/'
            # train_path = '/hdd/PhD/data/tiny-imagenet-200/'

            ret.append(Tiny_ImageNet(root=train_path, transforms=val_transform))
        else:
            ret.append(dset_cls(root=data_path, train=False, download=False, transform=val_transform))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def accuracy_multilabel(output, target, topk=(1,)):
#     assert max(topk) == 1 # topk doesn't make sense for multilabel
#
#     sigmoid = torch.sigmoid(output)
#
#     avp = ap(target, sigmoid)

def accuracy_multilabel_new(output, target, topk=(1,)):
    assert max(topk) == 1 # topk doesn't make sense for multilabel

    sigmoid = torch.sigmoid(output)
    # sigmoid[output>0.5] = 1
    # sigmoid[output<=0.5] = 0
    sigmoid[sigmoid > 0.5] = 1
    sigmoid[sigmoid <= 0.5] = 0
    is_one_sigm = np.where(np.float32(sigmoid.detach().cpu())==1)
    is_one_lab = np.where(np.float32(target.detach().cpu())==1)

    intersection = len(np.intersect1d(is_one_sigm, is_one_lab))
    union = len(np.union1d(is_one_sigm, is_one_lab))
    return intersection/union, np.array(0)



def accuracy_multilabel(output, target, topk=(1,), thr=None):
    assert max(topk) == 1 # topk doesn't make sense for multilabel

    if thr is None:
        sigmoid = torch.sigmoid(output)
        # sigmoid[output>0.5] = 1
        # sigmoid[output<=0.5] = 0
        sigmoid[sigmoid>0.5] = 1
        sigmoid[sigmoid<=0.5] = 0
        avg = 0
        thresholds = np.arange(0.9,1,0.005)
        for a_thr in thresholds:
            avg += accuracy_multilabel(sigmoid, target, topk, a_thr)
        return np.array(avg / len(thresholds)), np.array(0)
    else:
        avg = 0
        samples = (output == target)
        batch_size = samples.size(1)
        for sample in samples:
            if sample.sum().float() / batch_size > thr:
                avg += 1
        ret = avg / samples.size(0)
        # print(f"ret at threshold {thr} is {ret}")
        return ret


# def accuracy_multilabel(output, target, topk=(1,)):
#     assert max(topk) == 1 # topk doesn't make sense for multilabel
#     batch_size = target.size(0)
#     num_labels = target.size(1)
#
#     sigmoid = torch.sigmoid(output)
#     sigmoid[output>0.5] = 1
#     sigmoid[output<=0.5] = 0
#
#     return [(sigmoid == target).sum().float() / (batch_size * num_labels), np.array(0)] # return 0 for top5
#     raise AttributeError(ret, sigmoid, (sigmoid==target).sum().float(), target)
#     return ret


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
