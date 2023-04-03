import numpy as np
import pandas as pd
import torch
import geopandas
from tqdm import tqdm


def continuous_roads(df_full, road_idx):
    edges = df_full[df_full['LFN_ID'] == road_idx][['FNODE', 'TNODE']].values
    roads = []
    while len(edges) > 0:
        extended = True
        road = [edges[0, 0], edges[0, 1]]
        edges = edges[1:]
        while extended:
            extended = False
            for pos in [0, -1]:
                for idx in [0, 1]:
                    flag = edges[:, idx] == road[pos]
                    selected_edges = edges[flag]
                    if len(selected_edges) > 0:
                        if pos == 0:
                            road = [selected_edges[0][1-idx]] + road
                        else:
                            road = road + [selected_edges[0][1-idx]]
                        extended = True
                        edges = edges[(1 - flag).astype(bool)]
        roads.append(road)
    return roads


def discrete_roads(cont_roads, df_local, road_idx):
    df_selected = df_local.loc[df_local['LFN_ID'] == road_idx, :]
    des_roads = []
    for road in cont_roads:
        tmp_road = []
        for idx in range(len(road) - 1):
            df_tmp = df_selected[(df_selected['TNODE'] == road[idx]) & (df_selected['FNODE'] == road[idx+1])]
            if len(df_tmp) > 0:
                tmp_road.append(list(df_tmp.index)[0])
#                 df_selected.drop(df_tmp.index, axis=0, inplace=True)
            df_tmp = df_selected[(df_selected['FNODE'] == road[idx]) & (df_selected['TNODE'] == road[idx+1])]
            if len(df_tmp) > 0:
                tmp_road.append(list(df_tmp.index)[0])
#                 df_selected.drop(df_tmp.index, axis=0, inplace=True)
        des_roads.append(tmp_road)
    return [road for road in des_roads if len(road) > 1]


def identify_roads(loc, purpose, df, df_full):
    if loc:
        if purpose == 'training':
            train_indices = np.loadtxt(f'../data/{loc}_training_idx.txt', delimiter=',')
            vali_indices = np.loadtxt(f'../data/{loc}_validation_idx.txt', delimiter=',')
            indices = np.concatenate([train_indices, vali_indices], axis=0)
        else:
            indices = np.loadtxt(f'../data/{loc}_test_idx.txt', delimiter=',')
    else:
        if purpose == 'training':
            train_indices = np.loadtxt(f'../data/training_idx.txt', delimiter=',')
            vali_indices = np.loadtxt(f'../data/validation_idx.txt', delimiter=',')
            indices = np.concatenate([train_indices, vali_indices], axis=0)
        elif purpose == 'test':
            indices = np.loadtxt(f'../data/test_idx.txt', delimiter=',')
        else:
            indices = range(len(df))
    df_local = df.loc[indices, :]  # subset the road segments so that we focus on the area of interest
    # getting roads
    road_cnts = df_local[['LFN_ID', 'OBJECTID']].groupby('LFN_ID').count().sort_values(by='OBJECTID')
    road_indices = list(road_cnts[road_cnts['OBJECTID'] > 5].index)  # get the set of "roads", each with a distinct name/ID
    roads = []
    for idx in tqdm(road_indices):
        cont_roads = continuous_roads(df_full=df_full, road_idx=idx)
        dis_roads = discrete_roads(cont_roads=cont_roads, df_local=df_local, road_idx=idx)
        roads += dis_roads
    print(f'We found {len(roads)} roads.')
    return roads


def load_roads(loc, purpose):
    if purpose is None:
        return torch.load(f'./data/road/segments')
    if loc is None:
        return torch.load(f'./data/road/segments_{purpose}')
    else:
        return torch.load(f'./data/road/segments_{loc}_{purpose}')


def load_probs(target, purpose, kind, loc):
    if loc is None:
        predicted_probs = torch.load(f'./pred/{target}_{purpose}_prob.pt', map_location=torch.device('cpu'))[kind]
    else:
        predicted_probs = torch.load(f'./pred/{target}_{purpose}_{loc}_prob.pt', map_location=torch.device('cpu'))[kind]
    if kind != 'true':
        predicted_probs = [v.tolist() for v in predicted_probs]
        predicted_probs = np.array(predicted_probs)
        predicted_probs = np.exp(predicted_probs)
        predicted_probs /= predicted_probs.sum(axis=1, keepdims=True)
    else:
        predicted_probs = np.array(predicted_probs).reshape((-1, 1))
    return predicted_probs


def load_and_organize_label_predictions(target, kind, loc):
    # load true labels
    v_train_predictions = load_probs(target, 'training', kind=kind, loc=loc)
    v_vali_predictions = load_probs(target, 'validation', kind=kind, loc=loc)
    v_test_predictions = load_probs(target, 'test', kind=kind, loc=loc)
    if loc is not None:
        indi_train = np.loadtxt(f'./data/{loc}_training_idx.txt', delimiter=',').astype(int)
        indi_vali = np.loadtxt(f'./data/{loc}_validation_idx.txt', delimiter=',').astype(int)
        indi_test = np.loadtxt(f'./data/{loc}_test_idx.txt', delimiter=',').astype(int)
    else:
        indi_train = np.loadtxt(f'./data/training_idx.txt', delimiter=',').astype(int)
        indi_vali = np.loadtxt(f'./data/validation_idx.txt', delimiter=',').astype(int)
        indi_test = np.loadtxt(f'./data/test_idx.txt', delimiter=',').astype(int)
    # organize
    df = geopandas.read_file('./data/network/trt_network_filtered.shp')
    v_predictions = np.zeros((len(df), 1))
    v_predictions = pd.DataFrame(v_predictions)
    v_predictions.iloc[indi_train] = v_train_predictions
    v_predictions.iloc[indi_vali] = v_vali_predictions
    v_predictions.iloc[indi_test] = v_test_predictions
    return v_predictions.values.astype(int).reshape(-1)


def load_and_organize_prob_predictions(target, kind, loc):
    # load
    v_train_predictions = load_probs(target, 'training', kind, loc)
    v_vali_predictions = load_probs(target, 'validation', kind, loc)
    v_test_predictions = load_probs(target, 'test', kind, loc)
    if loc is not None:
        indi_train = np.loadtxt(f'./data/{loc}_training_idx.txt', delimiter=',').astype(int)
        indi_vali = np.loadtxt(f'./data/{loc}_validation_idx.txt', delimiter=',').astype(int)
        indi_test = np.loadtxt(f'./data/{loc}_test_idx.txt', delimiter=',').astype(int)
    else:
        indi_train = np.loadtxt(f'./data/training_idx.txt', delimiter=',').astype(int)
        indi_vali = np.loadtxt(f'./data/validation_idx.txt', delimiter=',').astype(int)
        indi_test = np.loadtxt(f'./data/test_idx.txt', delimiter=',').astype(int)
    # organize
    _, dim = v_train_predictions.shape
    df = geopandas.read_file('./data/network/trt_network_filtered.shp')
    v_predictions = np.zeros((len(df), dim))
    for d in range(dim):
        v_predictions = pd.DataFrame(v_predictions)
        v_predictions.iloc[indi_train, d] = v_train_predictions[:, d]
        v_predictions.iloc[indi_vali, d] = v_vali_predictions[:, d]
        v_predictions.iloc[indi_test, d] = v_test_predictions[:, d]
    return v_predictions.values.astype(float)


if __name__ == '__main__':
    df_full = geopandas.read_file('../data/network/network_w_easy_features.shp')
    df = geopandas.read_file('../data/network/trt_network_filtered.shp')
    for loc in [None, 'scarborough', 'york', 'etobicoke']:
        for purpose in [None, 'training', 'test']:
            roads = identify_roads(loc=loc, purpose=purpose, df=df, df_full=df_full)
            if loc is None:
                if purpose is None:
                    torch.save(roads, f'../data/road/segments')
                else:
                    torch.save(roads, f'../data/road/segments_{purpose}')
            else:
                torch.save(roads, f'../data/road/segments_{loc}_{purpose}')

