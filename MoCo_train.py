# coding: utf-8
# Author: Bo Lin
import torch
from torch import nn
from tqdm import tqdm

from model import MoCo
from utils import MoCoDataset
from torch.utils.data import DataLoader


def train_one_epoch(loader_train, net, criterion, optimizer, epoch, device):
    # switch to train
    net.train()
    for imgs in tqdm(loader_train):
        # forward step
        img_q, img_k = imgs[0].to(device), imgs[1].to(device)
        logits, labels = net(img_q, img_k)
        loss = criterion(logits, labels)
        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(device='mps', n_epoch=10, lr=0.03):
    # initialization
    loader_train = DataLoader(MoCoDataset(purpose='validation', local=True), shuffle=True, batch_size=32)
    net = MoCo(dim=128).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(n_epoch):
        train_one_epoch(loader_train, net, criterion, optimizer, epoch, device)


if __name__ == '__main__':
    train(device='mps', n_epoch=1)


