# coding: utf-8
# Author: Bo Lin

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Res50FC
from utils import StreetviewDataset


def load_net(model_name, model_state_dict):
    if model_name == 'Res50FC':
        net = Res50FC(pretrained=False).to(device=device)
    else:
        raise ValueError(f'model {model_name} is not found')
    net.load_state_dict(model_state_dict)
    return net


def gen_prediction(net, device, local, dataset, batch_size, filename):
    # generate dataloader
    print('evaluating the test set')
    loader = DataLoader(StreetviewDataset(purpose=dataset, local=local, toy=False),
                        batch_size=batch_size, shuffle=False)
    pred_records, true_records = [], []
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs, 1)
            predicted += 1
            pred_records += predicted.tolist()
            true_records += y.tolist()
    idx = np.loadtxt(f'./data/{dataset}_idx.txt')
    df = pd.DataFrame({'idx': idx, 'true': true_records, 'pred': pred_records})
    df.to_csv(f'./res/prediction/{filename}_{dataset}.csv', index=False)


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='name of the prediction model, e.g. Res50FC.')
    parser.add_argument('-c', '--checkpoint', type=str, help='name of the checkpoint.')
    parser.add_argument('-ds', '--dataset', type=str, help='theb dataset on which we are going to evaluate the model, training/validation/test.')
    parser.add_argument('--device', type=str, help='the device that is used to perform computation.')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not.')
    parser.add_argument('--no-local', dest='local', action='store_false')
    args = parser.parse_args()
    # load checkpoint
    checkpoint = torch.load(f'./checkpoint/{args.modelname}.pt')
    # set parameters
    device = checkpoint['hyper-parameters']['device']
    batch_size = checkpoint['hyper-parameters']['batch_size']
    # load net
    net = load_net(model_name=args.model, model_state_dict=checkpoint['model_state_dict'])
    gen_prediction(net=net, device=device, local=args.local, dataset=args.dataset,
                   batch_size=batch_size, filename=f'{args.model}_{args.checkpoint}')
