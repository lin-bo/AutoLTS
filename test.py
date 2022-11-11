import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from utils import StreetviewDataset, accuracy, agg_accuracy, mae, mse, ob, kt
from model import Res50FC, MoCoClf
import argparse


def init_mdl(mdl_name, device):
    if mdl_name == 'res50':
        mdl = Res50FC(pretrained=False).to(device=device)
    elif mdl_name == 'MoCoClf':
        mdl = MoCoClf(vali=True).to(device=device)
    else:
        ValueError(f'Model {mdl_name} not found')
    return mdl


def eval(net, test_loader, device, purpose):
    net.eval()
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
    pred_records, true_records = torch.tensor(pred_records).to(torch.float), torch.tensor(true_records).to(torch.float)
    # accuracy
    acc = accuracy(pred_records, true_records)
    aggacc = agg_accuracy(pred_records, true_records)
    mae_score = mae(pred_records, true_records)
    mse_score = mse(pred_records, true_records)
    ob_score = ob(pred_records, true_records)
    if purpose != 'training':
        # training matrix might be too big
        kt_score = kt(pred_records, true_records)
    else:
        kt_score = 0
    # aggregated accuracy
    # confusion matrix
    conf_mat = confusion_matrix(y_pred=pred_records, y_true=true_records, normalize='true')
    res = {'conf_mat': conf_mat, 'total_loss': total_loss,
           'accuracy': acc, 'aggregated_accuracy': aggacc,
           'mae': mae_score, 'mse': mse_score,
           'ob': ob_score, 'kt': kt_score}
    return res


def complete_eval(net, device, local):
    # training
    print('evaluating the training set')
    loader_train = DataLoader(StreetviewDataset(purpose='training', local=local, toy=False), batch_size=batch_size, shuffle=True)
    res_train = eval(net, loader_train, device, 'training')
    # validation
    print('evaluating the validation set')
    loader_vali = DataLoader(StreetviewDataset(purpose='validation', local=local, toy=False), batch_size=batch_size, shuffle=True)
    res_vali = eval(net, loader_vali, device, 'validation')
    # test
    print('evaluating the test set')
    loader_test = DataLoader(StreetviewDataset(purpose='test', local=local, toy=False), batch_size=batch_size, shuffle=True)
    res_test = eval(net, loader_test, device, 'test')
    return res_train, res_vali, res_test


if __name__ == '__main__':
    # set argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpointname', type=str, help='the name of the model in the checkpoint folder w/o .pt')
    parser.add_argument('--modelname', type=str, help='name of the architecture')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.add_argument('--device', type=str, help='device name')
    args = parser.parse_args()
    # load checkpoint
    if args.device == 'mps':
        checkpoint = torch.load(f'./checkpoint/{args.checkpointname}.pt',  map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(f'./checkpoint/{args.checkpointname}.pt')
    # set parameters
    device = checkpoint['hyper-parameters']['device']
    batch_size = checkpoint['hyper-parameters']['batch_size']
    # initialization
    # net = Res50FC(pretrained=False).to(device=device)
    net = init_mdl(args.modelname, device)
    net.load_state_dict(checkpoint['model_state_dict'])
    # eval
    res_train, res_vali, res_test = complete_eval(net, device, args.local)
    res = {'training': res_train, 'validation': res_vali, 'test': res_test}
    torch.save(res, f'./res/{args.checkpointname}_res.pt')
