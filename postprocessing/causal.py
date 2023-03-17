import geopandas
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

np.random.seed(0)


def est_trans_prob(target, roads, preds=None, report=True):
    """
    estimate the transition probabilities
    :param target: the target of interest (e.g. lts or speed_actual_onehot)
    :param roads: lists of roads, each road is a list of road segments
    :param preds: list of predicted values of the target associated with each road segment in the city
    :param report: whether or not to report the true probabilities, if False, assign a small probability to each zero entry
    :return: the transition probability matrix
    """
    # load tagets
    if preds is None:
        if target == 'lts':
            targets = np.loadtxt('./data/LTS/lts_labels.txt').astype(int)
        else:
            targets = np.loadtxt(f'./data/road/{target}.txt', delimiter=',').astype(int)
            targets = np.argmax(targets, axis=1)
    else:
        targets = preds
    # estimate
    dims = {'speed_actual_onehot':4, 'volume_onehot': 2, 'road_type_onehot': 4, 'cyc_infras_onehot': 4,
            'n_lanes_onehot':5, 'parking_onehot': 2, 'oneway_onehot': 2, 'lts': 4, 'lts_wo_volume': 4}
    n = dims[target]
    cnt = np.zeros((n, n))
    for road in roads:
        for i in range(len(road) - 1):
            cnt[targets[road[i]]-1, targets[road[i+1]]-1] += 1
            cnt[targets[road[i+1]]-1, targets[road[i]]-1] += 1  # to make sure that the matrix is symmetric
    if report:
        return (cnt / (cnt.sum(axis=1, keepdims=True) + 0.01)).round(2)  # add 0.01 to avoid zero denominator
    else:
        return (cnt / (cnt.sum(axis=1, keepdims=True) + 0.01) + 0.01).round(2)  # add 0.01 to avoid zero probability

