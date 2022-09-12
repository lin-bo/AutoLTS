import random
import numpy as np
import pandas as pd
import geopandas
import os
from PIL import ImageFilter
import torch


def train_test_split():
    # load a list of good images
    bad_idx = pd.read_csv('./data/streetview/do_not_use.txt', header=None)[0].values
    recollected = pd.read_csv('./data/streetview/good.txt', header=None)[0].values
    bad_list = [idx for idx in bad_idx if idx not in recollected]
    img_list = os.listdir('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset/')
    img_list = [int(f[:-4]) for f in img_list if f[-4:] == '.jpg']
    good_list = [idx for idx in img_list if idx not in bad_list]
    # train test split
    good_list = np.array(good_list)
    np.random.shuffle(good_list)
    training = good_list[:27407]
    vali = good_list[27407: 27407 + 5873]
    test = good_list[27407 + 5873:]
    print(f'training : {len(training)}, vali: {len(vali)}, test: {len(test)}')
    np.savetxt('./data/good_idx.txt', good_list, delimiter=',')
    np.savetxt('./data/training_idx.txt', training, delimiter=',')
    np.savetxt('./data/validation_idx.txt', vali, delimiter=',')
    np.savetxt('./data/test_idx.txt', test, delimiter=',')


def extract_lts_labels():
    df = geopandas.read_file('./data/network/trt_network_filtered.shp')
    lts = df['LTS'].values
    np.savetxt('./data/LTS/lts_labels.txt', lts, delimiter=',')


def initialization(check_path, n_check, n_epoch, job_id, net, optimizer):
    init_epoch = 0
    loss_records = []
    for epoch in list(range(n_epoch))[::-1]:
        if (epoch + 1) % n_check != 0:
            continue
        if os.path.exists(check_path + f'{job_id}_{epoch}.pt'):
            checkpoint = torch.load(check_path + f'{job_id}_{epoch}.pt')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            init_epoch = checkpoint['epoch']
            loss_records = checkpoint['loss_records']
            return init_epoch, loss_records, net, optimizer
    return init_epoch, loss_records, net, optimizer


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

