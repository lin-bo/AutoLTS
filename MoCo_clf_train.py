import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import argparse

from model import MoCoClf, MoCoClfV2, MoCoClfV3, MoCoClfV2Fea, MoCoClfV3Fea
from utils import StreetviewDataset, initialization, cal_dim

# set random seed
torch.manual_seed(0)


def validation(net, vali_loader, device, side_fea, criterion, label):
    tot_cnt = 0
    corr_cnt = 0
    total_loss = 0.
    epoch_cnt = 0
    net.eval()
    with torch.no_grad():
        if not side_fea:
            for x, y in tqdm(vali_loader):
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
                total_loss += loss.item()
                tot_cnt += y.shape[0]
                epoch_cnt += 1
                if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
                    _, predicted = torch.max(outputs, 1)
                    if label == 'lts':
                        predicted += 1
                    corr_cnt += (predicted == y).sum().item()
        else:
            for x, s, y in tqdm(vali_loader):
                x, s, y = x.to(device), s.to(device), y.to(device)
                outputs = net(x, s)
                loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
                total_loss += loss.item()
                tot_cnt += y.shape[0]
                epoch_cnt += 1
                if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
                    _, predicted = torch.max(outputs, 1)
                    if label == 'lts':
                        predicted += 1
                    corr_cnt += (predicted == y).sum().item()
    net.train()
    if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
        return total_loss, corr_cnt/tot_cnt * 100
    else:
        return total_loss/epoch_cnt, 0


def train_one_epoch(net, optimizer, train_loader, device, side_fea, criterion, label):
    net.train()
    total_loss = 0.
    tot_cnt = 0
    epoch_cnt = 0
    corr_cnt = 0
    if not side_fea:
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            outputs = net.forward(x)
            loss = criterion(outputs, y-1) if label == 'lts' else criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tot_cnt += len(y)
            epoch_cnt += 1
            if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
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
            if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
                _, y_pred = torch.max(outputs, dim=1)
                if label == 'lts':
                    y_pred += 1
                corr_cnt += (y_pred == y).sum().item()
    if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
        return total_loss, corr_cnt/tot_cnt * 100
    else:
        return total_loss/epoch_cnt, 0


def train(checkpoint=None, lr=0.0003, device='mps', batch_size=64, job_id=None, transform=False, biased=False, enc_frozen=True,
          n_epoch=30, n_check=1, toy=False, local=False, side_fea=[], label='lts', start_point=None, MoCoV=2, loc=None):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # initialize
    # if version == 1:
    #     net = MoCoClf(checkpoint_name=checkpoint, local=local).to(device)
    # else:
    #     net = MoCoClfV2(checkpoint_name=checkpoint, local=local).to(device)
    l2d = {'lts': 4,
           'oneway': 1, 'oneway_onehot': 2,
           'parking': 1, 'parking_onehot': 2,
           'volume': 1, 'volume_onehot': 2,
           'speed_actual': 1, 'speed_actual_onehot': 4,
           'cyc_infras': 2, 'cyc_infras_onehot': 4,
           'n_lanes': 1, 'n_lanes_onehot': 5,
           'road_type': 9, 'road_type_onehot': 4}
    if side_fea:
        n_fea = cal_dim(side_fea)
        if MoCoV == 2:
            net = MoCoClfV2Fea(checkpoint_name=checkpoint, local=local, n_fea=n_fea, out_dim=l2d[label]).to(device)
        elif MoCoV == 3:
            net = MoCoClfV3Fea(checkpoint_name=checkpoint, local=local, n_fea=n_fea, out_dim=l2d[label], enc_frozen=enc_frozen).to(device)
    else:
        if MoCoV == 2:
            net = MoCoClfV2(checkpoint_name=checkpoint, local=local, out_dim=l2d[label]).to(device)
        elif MoCoV == 3:
            net = MoCoClfV3(checkpoint_name=checkpoint, local=local, out_dim=l2d[label], enc_frozen=enc_frozen).to(device)
        else:
            raise ValueError('MoCo Version not found')
    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
    # l2c = {'lts': nn.CrossEntropyLoss(reduction='mean'), 'speed_actual': nn.MSELoss(reduction='mean')}
    # criterion = l2c[label].to(device)
    msefeas = {'speed_actual', 'n_lanes'}
    criterion = nn.MSELoss(reduction='mean') if label in msefeas else nn.CrossEntropyLoss(reduction='sum')
    criterion = criterion.to(device)
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local, augmentation=True, loc=loc,
                                                biased_sampling=biased, side_fea=side_fea, label=label, transform=transform),
                              batch_size=batch_size, shuffle=False)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local, augmentation=False, loc=loc,
                                               biased_sampling=False, side_fea=side_fea, label=label, transform=transform),
                             batch_size=batch_size, shuffle=False)
    # start training
    init_epoch, loss_records, net, optimizer, _ = initialization(check_path, n_check, n_epoch, job_id, net, optimizer, start_point)
    print(f'(Rs)Start training from epoch {init_epoch}')
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, device, side_fea, criterion, label)
        vali_loss, vali_acc = validation(net, vali_loader, device, side_fea, criterion, label)
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
        if label == 'lts' or label == 'road_type' or label[-7:] == '_onehot':
            print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, '
                  f'vali loss: {vali_loss:.4f}, vali accuracy: {vali_acc:.2f}%, '
                  f'time: {time.time() - tick:.2f} sec')
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
    parser.add_argument('--biased', action='store_true', default=False, help='apply data augmentation or not')
    parser.add_argument('--no-biased', dest='biased', action='store_false')
    parser.add_argument('--checkpoint', type=str, help='checkpoint name {JobID}_{Epoch}')
    parser.add_argument('--sidefea', nargs='+', type=str, help='side features that you want to consider, e.g. speed_limit, n_lanes')
    parser.add_argument('--label', type=str, default='lts', help='label to predict, choose from lts and speed_actual')
    parser.add_argument('--transform', default=False, action='store_true', help='apply data target log transformation or not')
    parser.add_argument('--no-transform', dest='transform', action='store_false')
    parser.add_argument('--start_point', default=None, type=str, help='starting point, must be saved in ./checkpoint/')
    parser.add_argument('--MoCoVersion', default=2, type=int, help='Version of MoCo, choose from 2 and 3, default 2')
    parser.add_argument('--frozen', default=True, action='store_true', help='apply data target log transformation or not')
    parser.add_argument('--no-frozen', dest='frozen', action='store_false')
    parser.add_argument('--location', default=None, type=str, help='the location of the test network, choose from None, scarborough, york, and etobicoke')
    args = parser.parse_args()
    # here we go
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, toy=args.toy, side_fea=args.sidefea, enc_frozen=args.frozen,
          local=args.local, batch_size=args.batchsize, job_id=args.jobid, checkpoint=args.checkpoint, label=args.label,
          transform=args.transform, start_point=args.start_point, biased=args.biased, lr=args.lr, MoCoV=args.MoCoVersion, loc=args.location)
