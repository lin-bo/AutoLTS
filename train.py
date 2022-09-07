import numpy as np
from utils import StreetviewDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Res50FC
from tqdm import tqdm
import argparse
from validate import validation


def train_one_epoch(net, optimizer, epoch, train_loader, device):
    net.train()
    criterion = nn.CrossEntropyLoss(reduction='sum')
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


def train(device='mps', n_epoch=10, n_check=5, local=True, batch_size=32, lr=0.0003, job_id=None, toy=False):
    # set parameters
    check_path = './checkpoint/' if local else f'/checkpoint/linbo/{job_id}/'
    # load training data
    train_loader = DataLoader(StreetviewDataset(purpose='training', toy=toy, local=local), batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(StreetviewDataset(purpose='validation', toy=toy, local=local), batch_size=batch_size, shuffle=True)
    # initialization
    net = Res50FC(pretrained=True).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_records = []
    print('start training ...')
    for epoch in range(n_epoch):
        train_loss, train_acc = train_one_epoch(net, optimizer, epoch, train_loader, device)
        vali_loss, vali_acc = validation(net, vali_loader, device)
        loss_records.append((train_loss, vali_loss))
        np.savetxt(check_path + 'Res50FC_loss.txt', loss_records, delimiter=',')
        if (epoch + 1) % n_check == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': (train_loss, vali_loss),
                        'hyper-parameters': {'n_epoch': n_epoch, 'n_check': n_check, 'device': device, 'batch_size': batch_size, 'lr': lr}
                        },
                       check_path + f'Res50FC_{epoch}.pt')
        print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, vali loss: {vali_loss:.4f}, vali accuracy: {vali_acc:.2f}%')


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='device for training')
    parser.add_argument('--jobid', type=str, help='job id')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size')
    parser.add_argument('-ne', '--nepoch', type=int, help='the number of epoch')
    parser.add_argument('-nc', '--ncheck', type=int, help='checkpoint frequency')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--toy', action='store_true', help='use the toy example or not')
    parser.add_argument('--no-toy', dest='toy', action='store_false')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    args = parser.parse_args()
    train(device=args.device, n_epoch=args.nepoch, n_check=args.ncheck, local=args.local,
          batch_size=args.batchsize, lr=args.lr, job_id=args.jobid, toy=args.toy)
