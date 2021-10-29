""" Config class for search/augment """
import argparse
import os
import genotypes as gt
from functools import partial
import torch
import sys

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    idx = sys.argv.index('--name')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--nodes', type=int, default=4, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')
        parser.add_argument('--dynamic', default=False, type=bool, help="learn dataset at same time")
        parser.add_argument('--ncc', default=False, type=bool, help="use ncc or no?")
        parser.add_argument('--isTree', default=True, type=bool, help="use kdtree or random sampling?")
        parser.add_argument('--init_train_epochs', default=20, type=int, help="how many epochs before we start to (consider to) adjust dataset")
        parser.add_argument('--subset_size', default=100, type=int, help="how many images in dataloader during epoch")
        parser.add_argument('--is_csv', default=False, type=bool, help="load initial idx from csv file?")
        parser.add_argument('--early_stopping', default=False, type=bool, help="are we doing early stopping?")
        parser.add_argument('--mastery', default=0.5, type=float, help="mastery threshold")
        parser.add_argument('--hardness', default=0.5, type=float, help="hardness threshold")
        parser.add_argument('--visualize', default=False, type=bool, help="are we visualizing?")
        parser.add_argument('--bede', default=False, type=bool, help="are we using bede server?")
        parser.add_argument('--resume', default=None, type=str, help="where to save pth file")
        parser.add_argument('--best_resume', default=None, type=str, help="where to save best pth file")
        parser.add_argument('--curriculum', default=False, type=bool, help="are we using curriculum?")
        parser.add_argument('--vanilla', default=False, type=bool, help="are we using vanilla data")
        parser.add_argument('--badpath', default=False, type=bool, help="are we using untrained autoencoder?")
        parser.add_argument('--nosave', default=False, type=bool, help="are we producing visualiation of updates?")

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')

        parser.add_argument('--genotype', required=True, help='Cell genotype')
        parser.add_argument('--dynamic', default=False, type=bool, help="learn dataset at same time")
        parser.add_argument('--ncc', default=True, type=bool, help="use ncc or no?")
        parser.add_argument('--isTree', default=True, type=bool, help="use kdtree or random sampling?")
        parser.add_argument('--init_train_epochs', default=20, type=int, help="how many epochs before we start to (consider to) adjust dataset")
        parser.add_argument('--subset_size', default=100, type=int, help="how many images in dataloader during epoch")
        parser.add_argument('--is_csv', default=False, type=bool, help="load initial idx from csv file?")
        parser.add_argument('--early_stopping', default=False, type=bool, help="are we doing early stopping?")
        parser.add_argument('--mastery', default=0.5, type=float, help="mastery threshold")
        parser.add_argument('--hardness', default=0.5, type=float, help="hardness threshold")
        parser.add_argument('--use_curriculum', default=False, type=bool, help="use learned curriculum")
        parser.add_argument('--final_mined', default=False, type=bool, help="use final mined curriculum only")
        parser.add_argument('--bede', default=False, type=bool, help="are we using bede server?")
        parser.add_argument('--vanilla', default=False, type=bool, help="are we using vanilla data")
        parser.add_argument('--badpath', default=False, type=bool, help="are we using untrained autoencoder?")

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('augments', self.name)
        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)
