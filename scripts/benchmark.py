""" Benchmark deepcopy vs. load_state_dict()
"""
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from models.search_cnn import SearchCNNController
import copy
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--sanity", action="store_true", default=False, help="sanity check")
args = parser.parse_args()


net_crit = nn.CrossEntropyLoss().cuda()
src = SearchCNNController(3, 16, 10, 8, net_crit, device_ids=[0]).cuda()
tgt = SearchCNNController(3, 16, 10, 8, net_crit, device_ids=[0]).cuda()


### Settings ###
B = 64
if args.sanity:
    print("Sanity check ...")
    N = 1

    # fixed inputs
    gen_X, gen_y = torch.randn(B, 3, 32, 32).cuda(), torch.randint(10, [B], dtype=torch.long).cuda()

    def gen_inputs(B):
        return copy.deepcopy(gen_X), copy.deepcopy(gen_y)
else:
    N = 5
    print("Benchmark with N = {}".format(N))

    # random inputs
    def gen_inputs(b):
        return torch.randn(b, 3, 32, 32).cuda(), torch.randint(10, [b], dtype=torch.long).cuda()


def load_state_dict(src, tgt, B):
    w_optim = torch.optim.SGD(src.weights(), 0.1, momentum=0.9, weight_decay=0.003)

    st = time.time()
    tgt.load_state_dict(src.state_dict())
    lsd = time.time() - st

    X, y = gen_inputs(B)
    loss = tgt.loss(X, y)
    gradients = torch.autograd.grad(loss, tgt.weights())

    with torch.no_grad():
        for rw, w, g in zip(src.weights(), tgt.weights(), gradients):
            m = w_optim.state[rw].get('momentum_buffer', 0.) * 0.9
            w -= 0.1 * (m + g + 0.003*w)

    total = time.time() - st

    return lsd, total, tgt


def deepcopy(src, tgt, B):
    w_optim = torch.optim.SGD(src.weights(), 0.1, momentum=0.9, weight_decay=0.003)

    st = time.time()
    tgt = copy.deepcopy(src)
    cpy = time.time() - st

    X, y = gen_inputs(B)
    loss = tgt.loss(X, y)
    gradients = torch.autograd.grad(loss, tgt.weights())

    with torch.no_grad():
        for rw, w, g in zip(src.weights(), tgt.weights(), gradients):
            m = w_optim.state[rw].get('momentum_buffer', 0.) * 0.9
            w -= 0.1 * (m + g + 0.003*w)

    total = time.time() - st

    return cpy, total, tgt


def direct(src, tgt, B):
    w_optim = torch.optim.SGD(src.weights(), 0.1, momentum=0.9, weight_decay=0.003)

    st = time.time()
    cpy = time.time() - st

    X, y = gen_inputs(B)
    loss = src.loss(X, y)
    gradients = torch.autograd.grad(loss, src.weights())

    with torch.no_grad():
        for rw, w, g in zip(src.weights(), tgt.weights(), gradients):
            m = w_optim.state[rw].get('momentum_buffer', 0.) * 0.9
            w.copy_(rw - 0.1 * (m + g + 0.003*rw))

        # synchronize alphas
        for ra, a in zip(src.alphas(), tgt.alphas()):
            a.copy_(ra)

    total = time.time() - st

    return cpy, total, tgt


def benchmark(name, N, func, func_args, sanity=None):
    print("Benchmark {} ...".format(name))
    copy_time = 0.
    total_time = 0.
    for i in range(N):
        torch.cuda.empty_cache()
        cpy, tot, res = func(*func_args)
        copy_time += cpy
        total_time += tot
        if args.sanity and i == N-1:
            print(next(res.weights()).view(-1)[:10])
            if sanity:
                maxdiff = 0.
                for (fn, fw), sw in zip(sanity.named_parameters(), res.parameters()):
                    md = (fw - sw).abs().max().item()
                    assert md < 1e-5, "diff = {} on {}".format(md, fn)
                    if md > maxdiff:
                        maxdiff = md
                print("Sanity check pass with max diff = {}".format(maxdiff))

    print("{}: {:.1f}s / {:.1f}s\n".format(name, copy_time/N, total_time/N))

    return res


if args.sanity:
    print("Source weights:")
    print(next(src.weights()).view(-1)[:10])
    print("")

print("-"*80)
standard = None
standard = benchmark("load_state_dict", N, load_state_dict, [src, copy.deepcopy(tgt), B], standard)
benchmark("deepcopy", N, deepcopy, [src, copy.deepcopy(tgt), B], standard)
benchmark("direct", N, direct, [src, copy.deepcopy(tgt), B], standard)
print("-"*80)
