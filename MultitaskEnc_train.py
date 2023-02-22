import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import argparse

from model import MultiTaskEncoder
from utils import MultitaskEncDataset, MultitaskLoss, assess_batch_pred, initialization, save_checkpoint

# set random seed
torch.manual_seed(0)


def validate(net, vali_loader, device, criterion, target_features):
    net.eval()
    net.vali = True
    total_loss = 0.
    cnts = torch.zeros(len(target_features)).to(device)
    dt_cnt = 0
    for img_q, img_k, lts, trues in tqdm(vali_loader):
        img_q, img_k, lts, trues = img_q.to(device), img_k.to(device), lts.to(device), [t.to(device) for t in trues]
        # forward
        logits, targets, preds = net(img_q, img_k, lts)
        loss = criterion(logits, targets, preds, trues)
        # backward
        total_loss += loss.item()
        # calculate accuracy
        with torch.no_grad():
            cnts += assess_batch_pred(preds, trues, target_features, device)
            dt_cnt += len(lts)
    net.vali = False
    return total_loss, cnts/dt_cnt


def train_one_epoch(net, train_loader, device, criterion, optimizer, target_features):
    net.train()
    total_loss = 0.
    cnts = torch.zeros(len(target_features)).to(device)
    dt_cnt = 0
    for img_q, img_k, lts, trues in tqdm(train_loader):
        img_q, img_k, lts, trues = img_q.to(device), img_k.to(device), lts.to(device), [t.to(device) for t in trues]
        # forward
        logits, targets, preds = net(img_q, img_k, lts)
        loss = criterion(logits, targets, preds, trues)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # calculate accuracy
        with torch.no_grad():
            cnts += assess_batch_pred(preds, trues, target_features, device)
            dt_cnt += len(lts)
    return total_loss, cnts/dt_cnt


def train(toy=True, local=True, batch_size=32, lr=0.003, device='mps', job_id=None, weights=None, n_check=100, n_epoch=1,
          start_point=None, aug_method='SimCLR', target_features=None, memsize=12800, alpha=2, MoCo_type = 'sup', weight_func='exp'):
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # initialize the model
    net = MultiTaskEncoder(dim=128, queue_size=memsize, n_chunk=4, target_features=target_features, device=device,
                           MoCo_type=MoCo_type, weight_func=weight_func, simple_shuffle=False, alpha=alpha).to(device)
    # initialize optimizer and loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = MultitaskLoss(contras='SupMoCo', weights=weights, target_features=target_features)
    # load dataset
    train_loader = DataLoader(MultitaskEncDataset(purpose='training', local=local, toy=toy, aug_method=aug_method, target_features=target_features),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    vali_loader = DataLoader(MultitaskEncDataset(purpose='validation', local=local, toy=toy, aug_method=aug_method, target_features=target_features),
                             batch_size=batch_size, shuffle=False, drop_last=False)
    n_train, n_vali  = len(train_loader), len(vali_loader)
    # initialize (read checkpoint)
    init_epoch, loss_records, net, optimizer, output_records = initialization(check_path, n_check, n_epoch, job_id, net, optimizer, start_point)
    # here we go
    msg = f'------------------------------------\n(re)Start training from epoch {init_epoch}\n------------------------------------'
    print(msg)
    output_records.append(msg)
    for epoch in range(init_epoch, n_epoch):
        tick = time.time()
        loss_train, metrics_train = train_one_epoch(net, train_loader, device, criterion, optimizer, target_features)
        loss_train = loss_train / n_train  # normalize
        if epoch % n_check == 0:
            loss_vali, metrics_vali = validate(net, vali_loader, device, criterion, target_features)
            loss_vali = loss_vali / n_vali
        else:
            loss_vali = loss_records[-1][1]
        loss_records.append((loss_train, loss_vali))
        # print record
        msg = f'epoch {epoch} -- train loss: {loss_train:.2f}, vali loss: {loss_vali:.2f} '
        for idx, fea in enumerate(target_features):
            if fea[-7:] == '_onehot' or fea == 'oneway':
                msg += f'|| **{fea[:-7]}** train: {metrics_train[idx] * 100:.2f}%, vali: {metrics_vali[idx] * 100:.2f}% '
            else:
                msg += f'|| **{fea}** train: {metrics_train[idx]:.4f}, vali: {metrics_vali[idx]:.4f} '
        msg += f'|| time: {time.time() - tick:.2f} sec'
        output_records.append(msg)
        print(msg)
        # save checkpoint
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
    parser.add_argument('--memsize', type=int, default=12800, help='size of the memory bank')
    parser.add_argument('--toy', action='store_true', help='use the toy example or not')
    parser.add_argument('--no-toy', dest='toy', action='store_false')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.add_argument('--MoCo_type', default='sup', type=str, help='type of the loss function, choose from sup and ord')
    parser.add_argument('--alpha', default=2, type=int, help='penalty factor for ord moco loss')
    parser.add_argument('-wf', '--weight_function', default='exp', type=str, help='weighting function, choose from exp and rec')
    parser.add_argument('--start_point', default=None, type=str, help='starting point, must be saved in ./checkpoint/')
    parser.add_argument('--aug_method', type=str, default='SimCLR', help='augmentation method, choose from SimCLR and Auto')
    parser.add_argument('--target_features', nargs='+', type=str, help='target features')
    parser.add_argument('--weights', nargs='+', type=float, help='weights in the loss function, pos 0 for contras loss')
    args = parser.parse_args()
    # here we go
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, toy=args.toy, MoCo_type=args.MoCo_type,
          start_point=args.start_point, weight_func=args.weight_function, alpha=args.alpha, local=args.local, batch_size=args.batchsize,
          job_id=args.jobid, memsize=args.memsize, aug_method=args.aug_method, target_features=args.target_features, weights=args.weights)

