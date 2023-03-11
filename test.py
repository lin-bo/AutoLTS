import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from utils import init_mdl
from utils import StreetviewDataset, accuracy, agg_accuracy, mae, mse, ob, kt, flr, fhr, cal_dim
import argparse


def eval(net, test_loader, device, purpose, side_fea, label, criterion):
    net.eval()
    total_loss = 0.
    pred_records, true_records = [], []
    with torch.no_grad():
        if not side_fea:
            for x, y in tqdm(test_loader):
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                if label == 'lts' or label == 'lts_wo_volume':
                    loss = criterion(outputs, y-1)
                else:
                    loss = criterion(outputs, y)
                total_loss += loss.item()
                if label != 'speed_actual':
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = outputs
                if label == 'lts' or label == 'lts_wo_volume':
                    predicted += 1
                pred_records += predicted.tolist()
                true_records += y.tolist()
        else:
            for x, s, y in tqdm(test_loader):
                x, s, y = x.to(device), s.to(device).to(torch.float), y.to(device)
                outputs = net(x, s)
                if label == 'lts' or label == 'lts_wo_volume':
                    loss = criterion(outputs, y-1)
                else:
                    loss = criterion(outputs, y)
                total_loss += loss.item()
                if label != 'speed_actual':
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = outputs
                if label == 'lts' or label == 'lts_wo_volume':
                    predicted += 1
                pred_records += predicted.tolist()
                true_records += y.tolist()
    pred_records, true_records = torch.tensor(pred_records).to(torch.float), torch.tensor(true_records).to(torch.float)
    # clf metrics
    acc = accuracy(pred_records, true_records) if label not in {'n_lanes', 'speed_actual'} else 0
    aggacc = agg_accuracy(pred_records, true_records) if label not in {'n_lanes', 'speed_actual'} else 0
    ob_score = ob(pred_records, true_records) if label in {'n_lanes_onehot', 'lts'} else 0
    # reg metrics
    mae_score = mae(pred_records, true_records) if label not in {'road_type', 'cyc_infras'} else 0
    mse_score = mse(pred_records, true_records) if label not in {'road_type', 'cyc_infras'} else 0
    # imbalanced metrics
    if label == 'lts' or label == 'lts_wo_volume':
        fhr_score = fhr(pred_records, true_records)
        flr_score = flr(pred_records, true_records)
    elif label == 'cyc_infras':
        fhr_score = fhr(pred_records + 2, true_records + 2)
        flr_score = flr(pred_records + 2, true_records + 2)
    else:
        fhr_score, flr_score = 0, 0
    # rank metrics
    if purpose != 'training' and (label == 'lts' or label == 'lts_wo_volume'):
        # training matrix might be too big
        kt_score = kt(pred_records, true_records)
    else:
        kt_score = 0
    # aggregated accuracy
    # confusion matrix
    if label not in {'n_lanes', 'speed_actual'}:
        conf_mat = confusion_matrix(y_pred=pred_records, y_true=true_records, normalize='true')
    else:
        conf_mat = 0
    res = {'conf_mat': conf_mat, 'total_loss': total_loss,
           'accuracy': acc, 'aggregated_accuracy': aggacc,
           'mae': mae_score, 'mse': mse_score,
           'ob': ob_score, 'kt': kt_score,
           'fhr': fhr_score, 'flr': flr_score}
    return res


def complete_eval(net, device, local, side_fea, label, loc):
    # training
    msefeas = {'speed_actual', 'n_lanes'}
    criterion = criterion = nn.MSELoss(reduction='mean') if label in msefeas else nn.CrossEntropyLoss(reduction='sum')
    print('evaluating the training set')
    loader_train = DataLoader(StreetviewDataset(purpose='training', local=local, toy=False, side_fea=side_fea, label=label, loc=loc),
                              batch_size=batch_size, shuffle=True)
    res_train = eval(net, loader_train, device, 'training', side_fea=side_fea, label=label, criterion=criterion)
    # validation
    print('evaluating the validation set')
    loader_vali = DataLoader(StreetviewDataset(purpose='validation', local=local, toy=False, side_fea=side_fea, label=label, loc=loc),
                             batch_size=batch_size, shuffle=True)
    res_vali = eval(net, loader_vali, device, 'validation', side_fea=side_fea, label=label, criterion=criterion)
    # test
    print('evaluating the test set')
    loader_test = DataLoader(StreetviewDataset(purpose='test', local=local, toy=False, side_fea=side_fea, label=label, loc=loc),
                             batch_size=batch_size, shuffle=True)
    res_test = eval(net, loader_test, device, 'test', side_fea=side_fea, label=label, criterion=criterion)
    return res_train, res_vali, res_test


if __name__ == '__main__':
    # set argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpointname', type=str, help='the name of the model in the checkpoint folder w/o .pt')
    parser.add_argument('--modelname', type=str, help='name of the architecture, choose from Res50, MoCoClf, MoCoClfV2, and MoCoClfV3')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.add_argument('--device', type=str, help='device name')
    parser.add_argument('--sidefea', nargs='+', type=str, help='side features that you want to consider, e.g. speed_limit, n_lanes')
    parser.add_argument('--label', type=str, default='lts', help='label to predict, choose from lts and speed_actual')
    parser.add_argument('--location', default=None, type=str, help='the location of the test network, choose from None, scarborough, york, and etobicoke')
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
    net = init_mdl(args.modelname, device, args.sidefea, label=args.label)
    net.load_state_dict(checkpoint['model_state_dict'])
    # eval
    res_train, res_vali, res_test = complete_eval(net, device, args.local, args.sidefea, args.label, loc=args.location)
    res = {'training': res_train, 'validation': res_vali, 'test': res_test}
    torch.save(res, f'./res/{args.checkpointname}_res.pt')
