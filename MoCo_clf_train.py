import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import argparse

from model import MoCoClf
from utils import StreetviewDataset, initialization
from validate import validation


def train_one_epoch(net, optimizer, train_loader, device):
    net.train()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
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


def train(checkpoint=None, lr=0.0003, device='mps', batch_size=64, job_id=None,
          n_epoch=30, n_check=1, toy=False, local=False, aug=True, biased=False):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # initialize
    net = MoCoClf(checkpoint_name=checkpoint, local=local).to(device)
    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(parameters, lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local, augmentation=aug, biased_sampling=biased), batch_size=batch_size, shuffle=False)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local, augmentation=False, biased_sampling=False), batch_size=batch_size, shuffle=False)
    # start training
    init_epoch, loss_records, net, optimizer, _ = initialization(check_path, n_check, n_epoch, job_id, net, optimizer)
    print(f'(Rs)Start training from epoch {init_epoch}')
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, device)
        vali_loss, vali_acc = validation(net, vali_loader, device)
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
    args = parser.parse_args()
    # here we go
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, toy=args.toy,
          local=args.local, batch_size=args.batchsize, job_id=args.jobid, checkpoint=args.checkpoint)

    # train(checkpoint='8618521_49', n_epoch=20, n_check=31, local=True, toy=True, aug=False, biased=False, batch_size=64,
    #       lr=0.0003)

