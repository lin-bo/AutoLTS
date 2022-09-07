import torch
from torch import nn
from tqdm import tqdm
from torchmetrics.functional import confusion_matrix
from torch.utils.data import DataLoader
from utils import StreetviewDataset
from model import Res50FC


def eval(net, test_loader, device):
    tot_cnt = 0
    corr_cnt = 0
    total_loss = 0.
    criterion = nn.CrossEntropyLoss(reduction='sum')
    pred_records, true_records = [], []
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
    pred_records, true_records = torch.tensor(pred_records) - 1, torch.tensor(true_records) - 1
    conf_mat = confusion_matrix(pred_records, true_records, num_classes=4, normalize='true')
    res = {'conf_mat': conf_mat, 'total_loss': total_loss, 'accuracy': corr_cnt/tot_cnt * 100}
    return res


def complete_eval(net, device):
    # training
    print('evaluating the training set')
    loader_train = DataLoader(StreetviewDataset(purpose='training'), batch_size=batch_size, shuffle=True)
    res_train = eval(net, loader_train, device)
    # validation
    print('evaluating the validation set')
    loader_vali = DataLoader(StreetviewDataset(purpose='validation'), batch_size=batch_size, shuffle=True)
    res_vali = eval(net, loader_vali, device)
    # test
    print('evaluating the test set')
    loader_test = DataLoader(StreetviewDataset(purpose='test'), batch_size=batch_size, shuffle=True)
    res_test = eval(net, loader_test, device)
    return res_train, res_vali, res_test


if __name__ == '__main__':
    model_name = 'Res50FC_5'
    # load checkpoint
    checkpoint = torch.load(f'./checkpoint/{model_name}.pt')
    # set parameters
    device = checkpoint['hyper-parameters']['device']
    batch_size = checkpoint['hyper-parameters']['batch_size']
    # initialization
    net = Res50FC(pretrained=False).to(device=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    # eval
    res_train, res_vali, res_test = complete_eval(net, device)
    res = {'training': res_train, 'validation': res_vali, 'test': res_test}
    torch.save(res, f'./res/{model_name}_res.pt')
