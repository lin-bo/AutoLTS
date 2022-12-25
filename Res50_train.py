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

# set random seed
torch.manual_seed(0)


def validation(net, vali_loader, device, criterion, side_fea, label):
    tot_cnt = 0
    corr_cnt = 0
    total_loss = 0.
    epoch_cnt = 0
    net.eval()
    with torch.no_grad():
        if not side_fea:
            for x, y in tqdm(vali_loader):
                x, y = x.to(device), y.to(device)
                # forward
                outputs = net(x)
                loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
                # log
                total_loss += loss.item()
                tot_cnt += len(y)
                epoch_cnt += 1
                if label != 'speed_actual' and label != 'n_lanes':
                    _, y_pred = torch.max(outputs, 1)
                    if label == 'lts':
                        y_pred += 1
                    corr_cnt += (y_pred == y).sum().item()
        else:
            for x, s, y in tqdm(vali_loader):
                x, s, y = x.to(device), s.to(device).to(torch.float), y.to(device)
                # forward
                outputs = net(x, s)
                loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
                # log
                total_loss += loss.item()
                tot_cnt += len(y)
                epoch_cnt += 1
                if label != 'speed_actual' and label != 'n_lanes':
                    _, y_pred = torch.max(outputs, 1)
                    if label == 'lts':
                        y_pred += 1
                    corr_cnt += (y_pred == y).sum().item()
    net.train()
    if label != 'speed_actual' and label != 'n_lanes':
        return total_loss, corr_cnt/tot_cnt * 100
    else:
        return total_loss/epoch_cnt, 0


def train_one_epoch(net, optimizer, train_loader, criterion, device, side_fea, label):
    net.train()
    total_loss = 0.
    epoch_cnt = 0
    tot_cnt = 0
    corr_cnt = 0
    if not side_fea:
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            # forward
            net.zero_grad()
            outputs = net.forward(x)
            loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
            # backward
            loss.backward()
            optimizer.step()
            # log
            total_loss += loss.item()
            tot_cnt += len(y)
            epoch_cnt += 1
            if label != 'speed_actual' and label != 'n_lanes':
                _, y_pred = torch.max(outputs, dim=1)
                if label == 'lts':
                    y_pred += 1
                corr_cnt += (y_pred == y).sum().item()
    else:
        for x, s, y in tqdm(train_loader):
            x, s, y = x.to(device), s.to(device), y.to(device)
            # forward
            net.zero_grad()
            outputs = net.forward(x, s)
            loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
            # backward
            loss.backward()
            optimizer.step()
            # log
            total_loss += loss.item()
            tot_cnt += len(y)
            epoch_cnt += 1
            if label != 'speed_actual' and label != 'n_lanes':
                _, y_pred = torch.max(outputs, dim=1)
                if label == 'lts':
                    y_pred += 1
                corr_cnt += (y_pred == y).sum().item()
    if label != 'speed_actual' and label != 'n_lanes':
        return total_loss, corr_cnt/tot_cnt * 100
    else:
        return total_loss/epoch_cnt, 0


def train(device='mps', n_epoch=10, n_check=5, local=True, batch_size=32, lr=0.0003, transform=False,
          job_id=None, toy=False, frozen=False, aug=False, biased=False, side_fea=[], label='lts', start_point=None):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # load training data
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local, augmentation=aug, label=label,
                                                biased_sampling=biased, side_fea=side_fea, transform=transform),
                              batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local, augmentation=False, label=label,
                                               biased_sampling=False, side_fea=side_fea, transform=transform),
                             batch_size=batch_size, shuffle=True)
    # initialization
    l2d = {'lts': 4, 'speed_actual': 1, 'cyc_infras': 2, 'n_lanes': 1, 'n_lanes_onehot': 9, 'road_type': 9}
    if not side_fea:
        net = Res50FC(pretrained=True, frozen=frozen, out_dim=l2d[label]).to(device)
    else:
        n_fea = cal_dim(side_fea)
        net = Res50FCFea(pretrained=True, frozen=frozen, n_fea=n_fea, out_dim=l2d[label]).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    msefeas = {'speed_actual', 'n_lanes'}
    criterion = nn.MSELoss(reduction='mean') if label in msefeas else nn.CrossEntropyLoss(reduction='sum')
    criterion = criterion.to(device)
    init_epoch, loss_records, net, optimizer, _ = initialization(check_path, n_check, n_epoch, job_id, net, optimizer, start_point)
    print(f'(Rs)Start training from epoch {init_epoch}')
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, criterion, device, side_fea, label)
        vali_loss, vali_acc = validation(net, vali_loader, device, criterion, side_fea, label)
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
        if label != 'speed_actual' and label != 'n_lanes':
            print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, vali loss: {vali_loss:.4f}, '
                  f'vali accuracy: {vali_acc:.2f}%, time: {time.time() - tick:.2f} sec')
        else:
            print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, vali loss: {vali_loss:.4f}, time: {time.time() - tick:.2f} sec')


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
    parser.add_argument('--label', type=str, default='lts', help='label to predict, choose from lts and speed_actual')
    parser.add_argument('--transform', default=False, action='store_true', help='apply data target log transformation or not')
    parser.add_argument('--no-transform', dest='transform', action='store_false')
    parser.add_argument('--start_point', default=None, type=str, help='starting point, must be saved in ./checkpoint/')
    args = parser.parse_args()
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, local=args.local, aug=args.aug, side_fea=args.sidefea,
          label=args.label, batch_size=args.batchsize, lr=args.lr, job_id=args.jobid, toy=args.toy, frozen=args.frozen,
          biased=args.biased, transform=args.transform, start_point=args.start_point)
