""" Config class for search/augment """
import argparse
import os
import genotypes as gt


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
        parser = argparse.ArgumentParser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9)
        parser.add_argument('--w_weight_decay', type=float, default=3e-4)
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=96)
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=3e-4)
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20)
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')

        parser.add_argument('--genotype', required=True, help='Cell genotype')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('augments', self.name)
        self.genotype = gt.from_str(self.genotype)
