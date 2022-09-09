import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from utils import StreetviewDataset
from model import Res50FC
import argparse


def eval(net, test_loader, device):
    tot_cnt = 0
    corr_cnt = 0
    total_loss = 0.
    criterion = nn.CrossEntropyLoss(reduction='sum')
    pred_records, true_records = [], []
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y-1)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            predicted += 1
            pred_records += predicted.tolist()
            true_records += y.tolist()
            tot_cnt += predicted.shape[0]
            corr_cnt += (predicted == y).sum().item()
    conf_mat = confusion_matrix(y_pred=pred_records, y_true=true_records, normalize='true')
    res = {'conf_mat': conf_mat, 'total_loss': total_loss, 'accuracy': corr_cnt/tot_cnt * 100}
    return res


def complete_eval(net, device, local):
    # training
    print('evaluating the training set')
    loader_train = DataLoader(StreetviewDataset(purpose='training', local=local, toy=False), batch_size=batch_size, shuffle=True)
    res_train = eval(net, loader_train, device)
    # validation
    print('evaluating the validation set')
    loader_vali = DataLoader(StreetviewDataset(purpose='validation', local=local, toy=False), batch_size=batch_size, shuffle=True)
    res_vali = eval(net, loader_vali, device)
    # test
    print('evaluating the test set')
    loader_test = DataLoader(StreetviewDataset(purpose='test', local=local, toy=False), batch_size=batch_size, shuffle=True)
    res_test = eval(net, loader_test, device)
    return res_train, res_vali, res_test


if __name__ == '__main__':
    # set argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, help='the name of the model in the checkpoint folder w/o .pt')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    args = parser.parse_args()
    # load checkpoint
    checkpoint = torch.load(f'./checkpoint/{args.modelname}.pt')
    # set parameters
    device = checkpoint['hyper-parameters']['device']
    batch_size = checkpoint['hyper-parameters']['batch_size']
    # initialization
    net = Res50FC(pretrained=False, local=args.local).to(device=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    # eval
    res_train, res_vali, res_test = complete_eval(net, device, args.local)
    res = {'training': res_train, 'validation': res_vali, 'test': res_test}
    torch.save(res, f'./res/{args.modelname}_res.pt')
