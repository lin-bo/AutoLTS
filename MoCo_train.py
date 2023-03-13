# coding: utf-8
# Author: Bo Lin
import torch
from torch import nn
from tqdm import tqdm
import argparse
import numpy as np
import time

from model import MoCo, LabelMoCo, OrdLabelMoCo
from utils import MoCoDataset, LabelMoCoDataset, LabelMoCoLoss, OrdLabelMoCoLoss, initialization
from torch.utils.data import DataLoader

# set random seed
torch.manual_seed(0)


def train_one_epoch(loader_train, net, criterion, optimizer, device, aware=False):
    # switch to train
    net.train()
    total_loss = 0.
    for dt in tqdm(loader_train):
        # forward step
        if aware:
            img_q, img_k, label = dt[0].to(device), dt[1].to(device), dt[2].to(device)
            logits, targets = net(img_q, img_k, label)
        else:
            img_q, img_k = dt[0].to(device), dt[1].to(device)
            logits, targets = net(img_q, img_k)
        loss = criterion(logits, targets)
        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def validate(loader_vali, net, criterion, device, aware=False):
    net.eval()
    net.vali = True
    total_loss = 0.
    for dt in tqdm(loader_vali):
        # forward step
        if aware:
            img_q, img_k, label = dt[0].to(device), dt[1].to(device), dt[2].to(device)
            logits, targets = net(img_q, img_k, label)
        else:
            img_q, img_k = dt[0].to(device), dt[1].to(device)
            logits, targets = net(img_q, img_k)
        loss = criterion(logits, targets)
        total_loss += loss.item()
    net.vali = False
    return total_loss


def save_checkpoint(net, optimizer, epoch, loss_records, n_epoch, n_check, device,
                    batch_size, lr, check_path, job_id, output_records):
    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_records': loss_records,
                'output_records': output_records,
                'hyper-parameters': {'n_epoch': n_epoch, 'n_check': n_check, 'device': device, 'batch_size': batch_size, 'lr': lr}
                },
               check_path + f'{job_id}_{epoch}.pt')


def train(device='mps', n_epoch=10, n_check=3, lr=0.03, toy=False, batch_size=32, awaretype='clf', alpha=2, aug_method='SimCLR', loc=None, temperature=0.007,
          job_id=None, local=False, simple_shuffle=False, aware=False, memsize=6400, weight_func='exp', start_point=None, hlinc=True, label='lts'):
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    output_records = []
    # initialize the network, data loader, and loss function
    if aware and awaretype == 'clf':
        net = LabelMoCo(dim=128, device=device, local=local, simple_shuffle=simple_shuffle, queue_size=memsize, temperature=temperature).to(device)
        dataset_train = LabelMoCoDataset(purpose='training', local=local, toy=toy, aug_method=aug_method, loc=loc, label=label)
        dataset_vali = LabelMoCoDataset(purpose='validation', local=local, toy=toy, aug_method=aug_method, loc=loc, label=label)
        criterion = LabelMoCoLoss().to(device)
    elif aware and awaretype == 'ord':
        net = OrdLabelMoCo(dim=128, device=device, local=local, simple_shuffle=simple_shuffle, queue_size=memsize,
                           alpha=alpha, weight_func=weight_func, inc_hl_dist=hlinc, temperature=temperature).to(device)
        dataset_train = LabelMoCoDataset(purpose='training', local=local, toy=toy, aug_method=aug_method, loc=loc, label=label)
        dataset_vali = LabelMoCoDataset(purpose='validation', local=local, toy=toy, aug_method=aug_method, loc=loc, label=label)
        criterion = OrdLabelMoCoLoss().to(device)
    elif aware and awaretype == 'reg':
        raise ValueError('reg MoCo loss has not been implemented yet')
    else:
        net = MoCo(dim=128, device=device, local=local, simple_shuffle=simple_shuffle, queue_size=memsize, temperature=temperature).to(device)
        dataset_train = MoCoDataset(purpose='training', local=local, toy=toy, loc=loc)
        dataset_vali = MoCoDataset(purpose='validation', local=local, toy=toy, loc=loc)
        criterion = nn.CrossEntropyLoss().to(device)
    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True)
    loader_vali = DataLoader(dataset_vali, shuffle=False, batch_size=int(batch_size//2), drop_last=True)
    n_train = len(dataset_train)
    n_vali = len(dataset_vali)
    # initialize optimizer and loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # load checkpoint if needed
    init_epoch, loss_records, net, optimizer, output_records = initialization(check_path, n_check, n_epoch, job_id, net, optimizer, start_point)
    # here we go
    msg = f'------------------------------------\n(re)Start training from epoch {init_epoch}\n------------------------------------'
    print(msg)
    output_records.append(msg)
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        loss_train = train_one_epoch(loader_train, net, criterion, optimizer, device, aware)
        loss_train = loss_train / n_train * 100  # normalize
        if epoch % n_check == 0:
            loss_vali = validate(loader_vali, net, criterion, device, aware)
            loss_vali = loss_vali / n_vali * 100
        else:
            loss_vali = loss_records[-1][1]
        loss_records.append((loss_train, loss_vali))
        msg = f'epoch {epoch}, training loss: {loss_train}, vali loss: {loss_vali}, time: {time.time() - tick:.2f} sec'
        output_records.append(msg)
        print(msg)
        if (epoch + 1) % n_check == 0:
            np.savetxt(check_path + f'{job_id}_loss.txt', loss_records, delimiter=',')
            save_checkpoint(net, optimizer, epoch, loss_records, n_epoch, n_check,
                            device, batch_size, lr, check_path, job_id, output_records)


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='device for training')
    parser.add_argument('--jobid', type=int, help='job id')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size')
    parser.add_argument('-ne', '--nepoch', type=int, help='the number of epoch')
    parser.add_argument('-nc', '--ncheck', type=int, help='checkpoint frequency')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--memsize', type=int, help='size of the memory bank')
    parser.add_argument('--toy', action='store_true', help='use the toy example or not')
    parser.add_argument('--no-toy', dest='toy', action='store_false')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.add_argument('--simple', action='store_true', help='whether or not to apply simple shuffle for the keys')
    parser.add_argument('--no-simple', dest='simple', action='store_false')
    parser.add_argument('--aware', action='store_true', help='whether or not to use the label aware MoCo')
    parser.add_argument('--no-aware', dest='aware', action='store_false')
    parser.add_argument('--awaretype', default='clf', type=str, help='type of the loss function, choose from clf, reg, and ord')
    parser.add_argument('--alpha', default=2, type=int, help='penalty factor for ord moco loss')
    parser.add_argument('-wf', '--weight_function', default='exp', type=str, help='weighting function, choose from exp and rec')
    parser.add_argument('--start_point', default=None, type=str, help='starting point, must be saved in ./checkpoint/')
    parser.add_argument('--aug_method', type=str, default='SimCLR', help='augmentation method, choose from SimCLR and Auto')
    parser.add_argument('--hlinc', action='store_true', default=True, help='is the training on a local device or not')
    parser.add_argument('--no-hlinc', dest='hlinc', action='store_false')
    parser.add_argument('--location', default=None, type=str, help='the location of the test network, choose from None, scarborough, york, and etobicoke')
    parser.add_argument('--label', type=str, default='lts', help='label to predict, choose from lts and lts_wo_volume')
    parser.add_argument('--temperature', type=float, default=0.007, help='the temperature (tau) used in contrastive learning loss')
    args = parser.parse_args()
    # here we go
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, toy=args.toy, aware=args.aware, awaretype=args.awaretype, label=args.label,
          start_point=args.start_point, weight_func=args.weight_function, alpha=args.alpha, local=args.local, batch_size=args.batchsize,
          job_id=args.jobid, simple_shuffle=args.simple, memsize=args.memsize, aug_method=args.aug_method, hlinc=args.hlinc, loc=args.location,
          temperature=args.temperature)


