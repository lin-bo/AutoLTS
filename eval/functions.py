import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


def load_fea(key, purpose, loc=None, updated=False, w_lts_pred=False):
    if loc is None:
        if updated:
            speed = np.loadtxt(f'./pred/speed_actual_onehot_{purpose}_updated.txt')
            n_lanes = np.loadtxt(f'./pred/n_lanes_onehot_{purpose}_updated.txt')
        else:
            speed = torch.load(f'./pred/speed_actual_onehot_{purpose}.pt')[key]
            n_lanes = torch.load(f'./pred/n_lanes_onehot_{purpose}.pt')[key]
        parking = torch.load(f'./pred/parking_onehot_{purpose}.pt')[key]
        oneway = torch.load(f'./pred/oneway_onehot_{purpose}.pt')[key]
        cyc_infras = torch.load(f'./pred/cyc_infras_onehot_{purpose}.pt')[key]
        road_type = torch.load(f'./pred/road_type_onehot_{purpose}.pt')[key]
        volume = torch.load(f'./pred/volume_onehot_{purpose}.pt')[key]
        if w_lts_pred:
            lts_pred = torch.load(f'./pred/lts_wo_volume_{purpose}.pt')[key]
    else:
        if updated:
            speed = np.loadtxt(f'./pred/speed_actual_onehot_{purpose}_{loc}_updated.txt')
            volume = np.loadtxt(f'./pred/volume_onehot_{purpose}_{loc}_updated.txt')
        else:
            speed = torch.load(f'./pred/speed_actual_onehot_{purpose}_{loc}.pt')
            volume = torch.load(f'./pred/volume_onehot_{purpose}_{loc}.pt')
        parking = torch.load(f'./pred/parking_onehot_{purpose}_{loc}.pt')
        oneway = torch.load(f'./pred/oneway_onehot_{purpose}_{loc}.pt')
        cyc_infras = torch.load(f'./pred/cyc_infras_onehot_{purpose}_{loc}.pt')
        n_lanes = torch.load(f'./pred/n_lanes_onehot_{purpose}_{loc}.pt')
        n_lanes = [v.item() for v in n_lanes]
        road_type = torch.load(f'./pred/road_type_onehot_{purpose}_{loc}.pt')
        if w_lts_pred:
            lts_pred = torch.load(f'./pred/lts_wo_volume_{purpose}_{loc}.pt')
    if not w_lts_pred:
        return speed, parking, oneway, cyc_infras, n_lanes, road_type, volume
    else:
        return speed, parking, oneway, cyc_infras, n_lanes, road_type, volume, lts_pred


def attr_mapping(speed, parking, oneway, cyc_infras, n_lanes, road_type, volume, lts_pred=None):
    df_speed = pd.DataFrame({'speed_actual': speed}).replace({0: 40, 1: 48, 2: 56, 3: 60})
    df_parking = pd.DataFrame({'parking_indi': 1 - np.array(parking)})
    df_oneway = pd.DataFrame({'oneway': 1 - np.array(oneway)})
    df_cyc_infras = pd.DataFrame({'cyc_infas': cyc_infras}).replace({0: 'Bike Lanes', 1: 'Cycle Tracks', 2: 'Multi-use Pathway', 3: 'No Infras'})
    df_cyc_infras = pd.get_dummies(df_cyc_infras)
    # df_cyc_infras.columns = ['Bike Lanes', 'Cycle Tracks', 'Multi-use Pathway', 'No Infras']
    df_cyc_infras.columns = [c[10:] for c in df_cyc_infras.columns]
    for c in ['Bike Lanes', 'Cycle Tracks', 'Multi-use Pathway', 'No Infras']:
        if c not in df_cyc_infras.columns:
            df_cyc_infras[c] = 0
    df_n_lanes = pd.DataFrame({'nlanes': n_lanes}) + 1
    df_road_type = pd.DataFrame({'road_type': road_type}).replace({0: 'Arterial', 1: 'Local', 2: 'Other', 3: 'Trail'})
    df_road_type = pd.get_dummies(df_road_type)
    df_road_type.columns = [c[10:] for c in df_road_type.columns]
    # df_road_type.columns = ['Arterial', 'Local', 'Other', 'Trail']
    for c in ['Arterial', 'Local', 'Other', 'Trail']:
        if c not in df_road_type.columns:
            print(c)
            df_road_type[c] = 0
    df_volume = pd.DataFrame({'volume': volume}).replace({0: 2000, 1: 4000})
    if lts_pred:
        df_lts = pd.DataFrame({'lts_pred': lts_pred})
        return pd.concat([df_speed, df_parking, df_oneway, df_cyc_infras, df_n_lanes, df_road_type, df_volume, df_lts], axis=1)
    else:
        return pd.concat([df_speed, df_parking, df_oneway, df_cyc_infras, df_n_lanes, df_road_type, df_volume], axis=1)


def prob2pred():
    for target in ['speed_actual_onehot', 'volume_onehot', 'cyc_infras_onehot', 'n_lanes_onehot', 'oneway_onehot', 'parking_onehot', 'road_type_onehot']:
        for loc in ['york', 'etobicoke', 'scarborough']:
            for purpose in ['training', 'validation', 'test']:
                pred = torch.load(f'../pred/{target}_{purpose}_{loc}_prob.pt', map_location=torch.device('cpu'))['pred']
                pred = [torch.argmax(p) for p in pred]
                torch.save(pred, f'../pred/{target}_{purpose}_{loc}.pt')


def model_eval(y_true, y_pred):
    # acc
    acc = np.sum(y_pred == y_true) / y_true.shape[0] * 100
    # h/l acc
    flag_pred = y_pred <= 2
    flag_true = y_true <= 2
    hl_acc = (flag_pred == flag_true).sum() / y_true.shape[0] * 100
    # mae
    # mae = np.abs(y_true - y_pred).mean()
    # mse
    # mse = ((y_true - y_pred) ** 2).mean()
    # flr
    n_high_stress = (y_true >= 3).sum()
    false_low_stress = ((y_pred <= 2) * (y_true >= 3)).sum()
    flr = false_low_stress / n_high_stress * 100
    # fhr
    n_low_stress = (y_true <= 2).sum()
    false_high_stress = ((y_pred >= 3) * (y_true <= 2)).sum()
    fhr = false_high_stress / n_low_stress * 100
    # afr
    afr = (flr + fhr) / 2
    # record generation
    return pd.Series([acc, hl_acc, flr, fhr, afr], index=['Accuracy', 'H/L Accuracy', 'FLR', 'FHR', 'AFR']).round(2), \
           confusion_matrix(y_true, y_pred, normalize='true').round(2)

