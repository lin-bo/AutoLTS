# coding: utf-8
# Author: Bo Lin
import torch
from torch import nn
from tqdm import tqdm
import argparse
import numpy as np
import time

from model import MoCo
from utils import MoCoDataset, initialization
from torch.utils.data import DataLoader

device = 'mps'
local = True
simple_shuffle = False
toy = True
batch_size = 32
check_path = './checkpoint/'

net = MoCo(dim=128, device=device, local=local, simple_shuffle=simple_shuffle, queue_size=12800).to(device)
dataset_train = MoCoDataset(purpose='training', local=local, toy=toy)
loader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size, drop_last=True)
criterion = nn.CrossEntropyLoss().to(device)

for q_img, k_img in loader_train:
    q_img, k_img = q_img.to(device), k_img.to(device)
    logits, labels = net(q_img, k_img)
    print(logits[0].sum())
    print(criterion(logits, labels))
    break

# checkpoint
checkpoint = torch.load(check_path + '8544579_6.pt', map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])

for q_img, k_img in loader_train:
    q_img, k_img = q_img.to(device), k_img.to(device)
    logits, labels = net(q_img, k_img)
    print(logits[0].sum())
    print(criterion(logits, labels))
    break

