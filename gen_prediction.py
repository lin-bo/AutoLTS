# coding: utf-8
# Author: Bo Lin

import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import StreetviewDataset, init_mdl


def gen_prediction(net, device, local, purpose, batch_size, side_fea, label, loc, prob_pred=False):
    # generate dataloader
    loader = DataLoader(StreetviewDataset(purpose=purpose, local=local, toy=False, side_fea=side_fea, label=label, loc=loc),
                        batch_size=batch_size, shuffle=False)
    pred_records, true_records = [], []
    net.eval()
    with torch.no_grad():
        if side_fea:
            for x, s, y in tqdm(loader):
                x, s, y = x.to(device), s.to(device).to(torch.float), y.to(device)
                outputs = net(x, s)
                if label != 'speed_actual':
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = outputs
                if label == 'lts':
                    predicted += 1
                if prob_pred:
                    pred_records += outputs
                else:
                    pred_records += predicted.tolist()
                true_records += y.tolist()
        else:
            for x, y in tqdm(loader):
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                if label != 'speed_actual':
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = outputs
                if label == 'lts':
                    predicted += 1
                if prob_pred:
                    pred_records += outputs
                else:
                    pred_records += predicted.tolist()
                true_records += y.tolist()
    return pred_records, true_records


def gen_all_predictions(net, device, local, batch_size, side_fea, label, loc, prob_pred):
    for purpose in ['training', 'validation', 'test']:
        print(f'generating {label} prediction for the {purpose} set')
        y_pred, y_true = gen_prediction(net, device, local, purpose, batch_size, side_fea, label, loc, prob_pred)
        records = {'pred': y_pred, 'true': y_true}
        if prob_pred:
            if loc:
                torch.save(records, f'./pred/{label}_{purpose}_{loc}_prob.pt')
            else:
                torch.save(records, f'./pred/{label}_{purpose}_prob.pt')
        else:
            if loc:
                torch.save(records, f'./pred/{label}_{purpose}_{loc}.pt')
            else:
                torch.save(records, f'./pred/{label}_{purpose}.pt')

if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='name of the prediction model, e.g. Res50FC.')
    parser.add_argument('-c', '--checkpoint', type=str, help='name of the checkpoint.')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size')
    parser.add_argument('--device', type=str, help='the device that is used to perform computation.')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not.')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.add_argument('--modelname', type=str, help='name of the architecture, choose from Res50, MoCoClf, and MoCoClfV2')
    parser.add_argument('--sidefea', nargs='+', type=str, help='side features that you want to consider, e.g. speed_limit, n_lanes')
    parser.add_argument('--label', type=str, default='lts', help='label to predict, choose from lts and speed_actual')
    parser.add_argument('--location', default=None, type=str, help='the location of the test network, choose from None, scarborough, york, and etobicoke')
    parser.add_argument('--prob_pred', default=False, action='store_true', help='whether or not to generate probability predictions instead of label predictions')
    args = parser.parse_args()
    # load checkpoint
    checkpoint = torch.load(f'./checkpoint/{args.checkpoint}.pt')
    # init net
    net = init_mdl(mdl_name=args.modelname, device=args.device, side_fea=args.sidefea, label=args.label)
    # load net
    net.load_state_dict(checkpoint['model_state_dict'])
    # generate
    gen_all_predictions(net=net, device=args.device, local=args.local, batch_size=args.batchsize, side_fea=args.sidefea, label=args.label, loc=args.location, prob_pred=args.prob_pred)
