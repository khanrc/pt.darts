# DARTS: Differentiable Architecture Search

Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). [[arxiv](https://arxiv.org/abs/1806.09055)]

## Requirements

- python 3
- pytorch >= 0.4
- graphviz
    - First install using `apt install` and then `pip install`.
    - or conda install may make it work.
- numpy
- tensorboardX

## Results

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
- Various dataset
- Tensorboard
- No RNN

and so on.
