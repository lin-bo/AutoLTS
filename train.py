import numpy as np
import torch
from tqdm import tqdm
import argparse
from torch import nn
from torch.utils.data import DataLoader

from model import Res50FC
from utils import initialization, StreetviewDataset


def validation(net, vali_loader, device, criterion):
    tot_cnt = 0
    corr_cnt = 0
    total_loss = 0.
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(vali_loader):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y-1)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            predicted += 1
            tot_cnt += predicted.shape[0]
            corr_cnt += (predicted == y).sum().item()
    net.train()
    return total_loss, corr_cnt/tot_cnt * 100


def train_one_epoch(net, optimizer, train_loader, device, criterion):
    net.train()
    total_loss = 0.
    tot_cnt = 0
    corr_cnt = 0
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
    return total_loss, corr_cnt/tot_cnt * 100


def train(device='mps', n_epoch=10, n_check=5, local=True, batch_size=32, lr=0.0003,
          job_id=None, toy=False, frozen=False, aug=False, biased=False, label='lts'):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # load training data
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local, augmentation=aug, biased_sampling=biased, label=label),
                              batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local, label=label), batch_size=batch_size, shuffle=True)
    # initialization
    net = Res50FC(pretrained=True, frozen=frozen).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    l2c = {'lts': nn.CrossEntropyLoss(reduction='sum'),
           'speed_actual': nn.MSELoss(reduction='sum')}
    criterion = l2c[label]
    init_epoch, loss_records, net, optimizer, _ = initialization(check_path, n_check, n_epoch, job_id, net, optimizer)
    print('start training ...')
    for epoch in range(init_epoch, n_epoch):
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, device, criterion)
        vali_loss, vali_acc = validation(net, vali_loader, device, criterion)
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
        if label == 'lts':
            print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, vali loss: {vali_loss:.4f}, vali accuracy: {vali_acc:.2f}%')
        else:
            print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, vali loss: {vali_loss:.4f}')


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
    parser.add_argument('--label', type=str, default='lts', help='label to predict, choose from lts and speed_actual')
    args = parser.parse_args()
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, local=args.local, aug=args.aug, label=args.label,
          batch_size=args.batchsize, lr=args.lr, job_id=args.jobid, toy=args.toy, frozen=args.frozen, biased=args.biased)
