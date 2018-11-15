# DARTS: Differentiable Architecture Search

- Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). [[arxiv](https://arxiv.org/abs/1806.09055)] [[code](https://github.com/quark0/darts)]

## Requirements

- python 3
- pytorch 0.4.1
- graphviz
    - apt-get install or conda install is required
- numpy
- tensorboardX

> To install graphviz on Ubuntu. First install using  sudo apt install graphviz and then sudo pip install graphviz, else it won't work.

## Results

| Dataset | Final validation acc | Best validation acc |
| ------- | -------------------- | ------------------- |
| MNIST         | 99.75% | 99.80% |
| Fashion-MNIST | 99.20% | 99.31% |
| CIFAR10       | 99.17% | 99.23% |

### Architecture progress

<p align="center">
<img src="assets/cifar10-normal.gif" alt="cifar10-progress-normal" width=45% />
<img src="assets/cifar10-reduce.gif" alt="cifar10-progress-reduce" width=45% />
</p>
<p align="center">
Figure: CIFAR10
</p>

<p align="center">
<img src="assets/mnist-normal.gif" alt="mnist-progress-normal" width=45% />
<img src="assets/mnist-reduce.gif" alt="mnist-progress-reduce" width=45% />
</p>
<p align="center">
Figure: MNIST 
</p>

<p align="center">
<img src="assets/fashionmnist-normal.gif" alt="fashionmnist-progress-normal" width=45% />
<img src="assets/fashionmnist-reduce.gif" alt="fashionmnist-progress-reduce" width=45% />
</p>
<p align="center">
Figure: Fashion-MNIST
</p>

