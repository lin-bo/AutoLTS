import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import argparse

from model import MoCoClf, MoCoClfV2, MoCoClfV2Fea
from utils import StreetviewDataset, initialization, cal_dim


def validation(net, vali_loader, device, side_fea):
    tot_cnt = 0
    corr_cnt = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.
    net.eval()
    with torch.no_grad():
        if not side_fea:
            for x, y in tqdm(vali_loader):
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                loss = criterion(outputs, y-1)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predicted += 1
                tot_cnt += predicted.shape[0]
                corr_cnt += (predicted == y).sum().item()
        else:
            for x, s, y in tqdm(vali_loader):
                x, s, y = x.to(device), s.to(device), y.to(device)
                outputs = net(x, s)
                loss = criterion(outputs, y-1)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predicted += 1
                tot_cnt += predicted.shape[0]
                corr_cnt += (predicted == y).sum().item()
    net.train()
    return total_loss, corr_cnt/tot_cnt * 100


def train_one_epoch(net, optimizer, train_loader, device, side_fea):
    net.train()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    total_loss = 0.
    tot_cnt = 0
    corr_cnt = 0
    if not side_fea:
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            outputs = net.forward(x)
            loss = criterion(outputs, y-1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, y_pred = torch.max(outputs, dim=1)
            y_pred += 1
            tot_cnt += len(y_pred)
            corr_cnt += (y_pred == y).sum().item()
    else:
        for x, s, y in tqdm(train_loader):
            x, s, y = x.to(device), s.to(device), y.to(device)
            # forward
            net.zero_grad()
            outputs = net.forward(x, s)
            _, y_pred = torch.max(outputs, dim=1)
            y_pred += 1
            loss = criterion(outputs, y-1)
            # backward
            loss.backward()
            optimizer.step()
            # log
            total_loss += loss.item()
            tot_cnt += len(y_pred)
            corr_cnt += (y_pred == y).sum().item()
    return total_loss, corr_cnt/tot_cnt * 100


def train(checkpoint=None, lr=0.0003, device='mps', batch_size=64, job_id=None,
          n_epoch=30, n_check=1, toy=False, local=False, version=1, side_fea=[]):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # initialize
    # if version == 1:
    #     net = MoCoClf(checkpoint_name=checkpoint, local=local).to(device)
    # else:
    #     net = MoCoClfV2(checkpoint_name=checkpoint, local=local).to(device)
    if side_fea:
        n_fea = cal_dim(side_fea)
        net = MoCoClfV2Fea(checkpoint_name=checkpoint, local=local, n_fea=n_fea).to(device)
    else:
        net = MoCoClfV2(checkpoint_name=checkpoint, local=local).to(device)
    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local, augmentation=True, biased_sampling=False, side_fea=side_fea),
                              batch_size=batch_size, shuffle=False)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local, augmentation=False, biased_sampling=False, side_fea=side_fea),
                             batch_size=batch_size, shuffle=False)
    # start training
    init_epoch, loss_records, net, optimizer, _ = initialization(check_path, n_check, n_epoch, job_id, net, optimizer)
    print(f'(Rs)Start training from epoch {init_epoch}')
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, device, side_fea)
        vali_loss, vali_acc = validation(net, vali_loader, device, side_fea)
        loss_records.append((train_loss, vali_loss))
        np.savetxt(check_path + f'{job_id}_loss.txt', loss_records, delimiter=',')
        if (epoch + 1) % n_check == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_records': loss_records,
                        'hyper-parameters': {'n_epoch': n_epoch, 'n_check': n_check, 'device': device, 'batch_size': batch_size, 'lr': lr}
                        },
                       check_path + f'{job_id}_{epoch}.pt')
        print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, '
              f'vali loss: {vali_loss:.4f}, vali accuracy: {vali_acc:.2f}%, '
              f'time: {time.time() - tick:.2f} sec')


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
    parser.add_argument('--checkpoint', type=str, help='checkpoint name {JobID}_{Epoch}')
    parser.add_argument('--version', type=int, default=1, help='MoCoClf version, choose from 1 and 2')
    parser.add_argument('--sidefea', nargs='+', type=str, help='side features that you want to consider, e.g. speed_limit, n_lanes')
    args = parser.parse_args()
    # here we go
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, toy=args.toy, version=args.version, side_fea=args.sidefea,
          local=args.local, batch_size=args.batchsize, job_id=args.jobid, checkpoint=args.checkpoint)
