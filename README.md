# DARTS: Differentiable Architecture Search

Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). [[arxiv](https://arxiv.org/abs/1806.09055)]

## Requirements

- python 3
- pytorch >= 0.4
- graphviz
    - First install using `apt install` and then `pip install`.
- numpy
- tensorboardX

## Run example

Adjust the batch size if out of memory (OOM) occurs. It dependes on your gpu memory size and genotype.

- Search

```shell
python search.py --name cifar10 --dataset cifar10
```

- Augment

```shell
# genotype: from search
python augment.py --name cifar10 --dataset cifar10 --genotype genotype
```

- with docker

```shell
$ docker run --runtime=nvidia -it khanrc/pytorch-darts:0.1 bash

# you can run directly also
$ docker run --runtime=nvidia -it khanrc/pytorch-darts:0.1 python search.py --name cifar10 --dataset cifar10
```

## Results

The following results are obtained using the default arguments.

| Dataset | Final validation acc | Best validation acc |
| ------- | -------------------- | ------------------- |
| MNIST         | 99.75% | 99.80% |
| Fashion-MNIST | 99.20% | 99.31% | 
| CIFAR-10       | 97.17% | 97.23% |

97.17%, final validation accuracy in CIFAR-10, is the same number as the paper.

### Architecture progress

<p align="center">
<img src="assets/cifar10-normal.gif" alt="cifar10-progress-normal" width=45% />
<img src="assets/cifar10-reduce.gif" alt="cifar10-progress-reduce" width=45% />
<br/> CIFAR-10 
</p>

<p align="center">
<img src="assets/mnist-normal.gif" alt="mnist-progress-normal" width=45% />
<img src="assets/mnist-reduce.gif" alt="mnist-progress-reduce" width=45% />
<br/> MNIST 
</p>

<p align="center">
<img src="assets/fashionmnist-normal.gif" alt="fashionmnist-progress-normal" width=45% />
<img src="assets/fashionmnist-reduce.gif" alt="fashionmnist-progress-reduce" width=45% />
<br/> Fashion-MNIST 
</p>

### Plots

<p align="center">
<img src="assets/fashionmnist-search.png" alt="fashionmnist-search" width=80% />
</p>
<p align="center"> Search-training phase of Fashion-MNIST </p>

<p align="center">
<img src="assets/cifar10-val.png" alt="cifar10-val" width=48% />
<img src="assets/fashionmnist-val.png" alt="fashionmnist-val" width=48% />
</p>
<p align="center"> Augment-validation phase of CIFAR-10 and Fashion-MNIST </p>

## Reference

https://github.com/quark0/darts (official implementation)

### Main differences to reference code

- Supporting pytorch >= 0.4
- Code that is easy to read and commented.
- Implemenation of architect
    - Original implementation is very slow in pytorch >= 0.4.
- Tested on FashionMNIST / MNIST
- Tensorboard
- No RNN

and so on.
