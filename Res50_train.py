import numpy as np
import torch
from tqdm import tqdm
import argparse
from torch import nn
from torch.utils.data import DataLoader
import time

from model.naive import FeaFC
from model import Res50FC, Res50FCFea
from utils import initialization, StreetviewDataset, cal_dim


def validation(net, vali_loader, device, criterion, side_fea):
    tot_cnt = 0
    corr_cnt = 0
    total_loss = 0.
    net.eval()
    with torch.no_grad():
        if not side_fea:
            for x, y in tqdm(vali_loader):
                x, y = x.to(device), y.to(device)
                # forward
                outputs = net(x)
                _, y_pred = torch.max(outputs, 1)
                loss = criterion(outputs, y-1)
                # log
                total_loss += loss.item()
                tot_cnt += y_pred.shape[0]
                corr_cnt += (y_pred + 1 == y).sum().item()
        else:
            for x, s, y in tqdm(vali_loader):
                x, s, y = x.to(device), s.to(device).to(torch.float), y.to(device)
                # forward
                outputs = net(x, s)
                _, y_pred = torch.max(outputs, 1)
                loss = criterion(outputs, y-1)
                # log
                total_loss += loss.item()
                tot_cnt += y_pred.shape[0]
                corr_cnt += (y_pred + 1 == y).sum().item()
    net.train()
    return total_loss, corr_cnt/tot_cnt * 100


def train_one_epoch(net, optimizer, train_loader, criterion, device, side_fea):
    net.train()
    total_loss = 0.
    tot_cnt = 0
    corr_cnt = 0
    if not side_fea:
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            # forward
            net.zero_grad()
            outputs = net.forward(x)
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


def train(device='mps', n_epoch=10, n_check=5, local=True, batch_size=32, lr=0.0003,
          job_id=None, toy=False, frozen=False, aug=False, biased=False, side_fea=[]):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # load training data
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local, augmentation=aug, biased_sampling=biased, side_fea=side_fea),
                              batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local, augmentation=False, biased_sampling=False, side_fea=side_fea),
                             batch_size=batch_size, shuffle=True)
    # initialization
    if not side_fea:
        net = Res50FC(pretrained=True, frozen=frozen).to(device)
    else:
        n_fea = cal_dim(side_fea)
        net = Res50FCFea(pretrained=True, frozen=frozen, n_fea=n_fea).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    loss_records = []
    init_epoch, loss_records, net, optimizer, _ = initialization(check_path, n_check, n_epoch, job_id, net, optimizer)
    print(f'(Rs)Start training from epoch {init_epoch}')
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, criterion, device, side_fea)
        vali_loss, vali_acc = validation(net, vali_loader, device, criterion, side_fea)
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
        print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, vali loss: {vali_loss:.4f}, '
              f'vali accuracy: {vali_acc:.2f}%, time: {time.time() - tick:.2f} sec')


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
    parser.add_argument('--frozen', action='store_true', help='freeze the pretrained resent or not')
    parser.add_argument('--no-frozen', dest='frozen', action='store_false')
    parser.add_argument('--aug', action='store_true', help='apply data augmentation or not')
    parser.add_argument('--no-aug', dest='aug', action='store_false')
    parser.add_argument('--biased', action='store_true', help='apply data augmentation or not')
    parser.add_argument('--no-biased', dest='biased', action='store_false')
    parser.add_argument('--speed', action='store_true', help='apply data augmentation or not')
    parser.add_argument('--no-speed', dest='speed', action='store_false')
    parser.add_argument('--sidefea', nargs='+', type=str, help='side features that you want to consider, e.g. speed_limit, n_lanes')
    args = parser.parse_args()
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, local=args.local, aug=args.aug, side_fea=args.sidefea,
          batch_size=args.batchsize, lr=args.lr, job_id=args.jobid, toy=args.toy, frozen=args.frozen, biased=args.biased)
