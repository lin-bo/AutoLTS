# coding: utf-8
# Author: Bo Lin
import torch
from torch import nn
from tqdm import tqdm
import argparse
import numpy as np

from model import MoCo
from utils import MoCoDataset, initialization
from torch.utils.data import DataLoader


def train_one_epoch(loader_train, net, criterion, optimizer, epoch, device):
    # switch to train
    net.train()
    total_loss = 0.
    for imgs in tqdm(loader_train):
        # forward step
        img_q, img_k = imgs[0].to(device), imgs[1].to(device)
        logits, labels = net(img_q, img_k)
        loss = criterion(logits, labels)
        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def validate(loader_vali, net, criterion, device):
    net.eval()
    net.vali = True
    total_loss = 0.
    for imgs in tqdm(loader_vali):
        # forward step
        img_q, img_k = imgs[0].to(device), imgs[1].to(device)
        logits, labels = net(img_q, img_k)
        loss = criterion(logits, labels)
        total_loss += loss.item()
    net.vali = False
    return total_loss


def save_checkpoint(net, optimizer, epoch, loss_records, n_epoch, n_check, device, batch_size, lr, check_path, job_id):
    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_records': loss_records,
                'hyper-parameters': {'n_epoch': n_epoch, 'n_check': n_check, 'device': device, 'batch_size': batch_size, 'lr': lr}
                },
               check_path + f'{job_id}_{epoch}.pt')


def train(device='mps', n_epoch=10, n_check=3, lr=0.03, toy=False, batch_size=32, job_id=None, local=False):
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # create dataloaders
    dataset_train = MoCoDataset(purpose='training', local=local, toy=toy)
    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True)
    dataset_vali = MoCoDataset(purpose='validation', local=local, toy=toy)
    loader_vali = DataLoader(dataset_vali, shuffle=True, batch_size=batch_size, drop_last=False)
    n_train, n_vali = len(dataset_train), len(dataset_vali)
    # initialize the network and optimizer
    net = MoCo(dim=128, device=device, local=local).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    loss_records = []
    # load checkpoint if needed
    init_epoch, loss_records, net, optimizer = initialization(check_path, n_check, n_epoch, job_id, net, optimizer)
    print('start training ...')
    # here we go
    for epoch in range(init_epoch, n_epoch):
        loss_train = train_one_epoch(loader_train, net, criterion, optimizer, epoch, device)
        loss_vali = validate(loader_vali, net, criterion, device)
        loss_train, loss_vali = loss_train / n_train * 100, loss_vali / n_vali * 100 # normalize
        loss_records.append((loss_train, loss_vali))
        np.savetxt(check_path + f'{job_id}_loss.txt', loss_records, delimiter=',')
        if epoch + 1 % n_check == 0:
            save_checkpoint(net, optimizer, epoch, loss_records, n_epoch, n_check, device, batch_size, lr, check_path, job_id)
        print(f'epoch {epoch}, training loss: {loss_train:.2f}, validation loss: {loss_vali:.2f}')


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='device for training')
    parser.add_argument('--jobid', type=int, help='job id')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size')
    parser.add_argument('-ne', '--nepoch', type=int, help='the number of epoch')
    parser.add_argument('-nc', '--ncheck', type=int, help='checkpoint frequency')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--toy', action='store_true', help='use the toy example or not')
    parser.add_argument('--no-toy', dest='toy', action='store_false')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    args = parser.parse_args()
    # here we go
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, toy=args.toy, local=args.local)


