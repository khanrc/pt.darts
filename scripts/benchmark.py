""" Benchmark deepcopy vs. load_state_dict()
"""
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from models.search_cnn import SearchCNN
import copy
import time


net_crit = nn.CrossEntropyLoss().cuda()
src = SearchCNN(3, 16, 10, 8, net_crit).cuda()
tgt = SearchCNN(3, 16, 10, 8, net_crit).cuda()
N = 10

print("benchmarking load_state_dict ...")
st = time.time()
for i in range(N):
    tgt.load_state_dict(src.state_dict())
print("load_state_dict: {:.1f}s".format(time.time() - st))


print("benchmarking deepcopy ...")
st = time.time()
for i in range(N):
    tgt = copy.deepcopy(src)
print("deepcopy: {:.1f}s".format(time.time() - st))
