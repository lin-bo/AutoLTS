import numpy as np
import pandas as pd
import argparse

from eval import model_eval, load_fea, attr_mapping, lts_prediction_wo_volume


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    args = parser.parse_args()
    return args


def load_feature(sce, location):
    # ground truth
    df_true = pd.read_csv('./data/network/final.csv')
    indi_train, indi_vali, indi_test = load_indis(location)
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')

    s2indi = {'training': indi_train, 'validation': indi_vali, 'test': indi_test}
    # training data
    for purpose in ['test']:
        speed, parking, oneway, cyc_infras, n_lanes, road_type, volume = load_fea(key='pred', purpose=purpose, loc=location, updated=True)
        df = attr_mapping(speed, parking, oneway, cyc_infras, n_lanes, road_type, volume)
        if sce == 2:
            df[['Bike Lanes', 'Cycle Tracks', 'Multi-use Pathway', 'No Infras']] = df_true.loc[s2indi[purpose], ['Bike Lanes', 'Cycle Tracks', 'Multi-use Pathway', 'No Infras']].values
            df['oneway'] = df_true.loc[s2indi[purpose], 'oneway'].values
            df[['Arterial', 'Local', 'Other', 'Trail']] = df_true.loc[s2indi[purpose], ['Arterial', 'Local', 'Other', 'Trail']].values
        elif sce == 3:
            df['speed_actual'] = df_true.loc[s2indi[purpose], 'speed_limit'].values
            df['nlanes'] = df_true.loc[s2indi[purpose], 'nlanes'].values
        df.drop('volume', axis=1, inplace=True)
    return df


def load_indis(location):
    if location is None:
        indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
        indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
        indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
    else:
        indi_train = np.loadtxt(f'./data/{location}_training_idx.txt').astype(int)
        indi_vali = np.loadtxt(f'./data/{location}_validation_idx.txt').astype(int)
        indi_test = np.loadtxt(f'./data/{location}_test_idx.txt').astype(int)
    return indi_train, indi_vali, indi_test


def predict_and_eval(location):
    _, _, indi_test = load_indis(location)
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')
    y_test = y[indi_test]

    df = load_feature(sce=1, location=location)
    predictions = df.apply(lts_prediction_wo_volume, axis=1).values
    print('Scenario 1')
    print(model_eval(y_test, predictions))

    print('\n')
    df = load_feature(sce=2, location=location)
    predictions = df.apply(lts_prediction_wo_volume, axis=1).values
    print('Scenario 2')
    print(model_eval(y_test, predictions))

    print('\n')
    df = load_feature(sce=3, location=location)
    predictions = df.apply(lts_prediction_wo_volume, axis=1).values
    print('Scenario 3')
    print(model_eval(y_test, predictions))


if __name__ == '__main__':
    args = parse_args()
    predict_and_eval(location=args.location)
