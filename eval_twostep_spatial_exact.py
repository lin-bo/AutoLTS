import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from eval import model_eval, load_fea, attr_mapping, lts_prediction_wo_volume


def load_feature(sce):
    # ground truth
    df_true = pd.read_csv('./data/network/final.csv')
    indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
    indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
    indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
    s2indi = {'training': indi_train, 'validation': indi_vali, 'test': indi_test}
    # training data
    for purpose in ['test']:
        speed, parking, oneway, cyc_infras, n_lanes, road_type, volume = load_fea(key='pred', purpose=purpose, loc=None, updated=True)
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


if __name__ == '__main__':
    indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')
    y_test = y[indi_test]

    df = load_feature(sce=1)
    predictions = df.apply(lts_prediction_wo_volume, axis=1).values
    print('Scenario 1')
    print(model_eval(y_test, predictions))

    print('\n')
    df = load_feature(sce=2)
    predictions = df.apply(lts_prediction_wo_volume, axis=1).values
    print('Scenario 2')
    print(model_eval(y_test, predictions))

    print('\n')
    df = load_feature(sce=3)
    predictions = df.apply(lts_prediction_wo_volume, axis=1).values
    print('Scenario 3')
    print(model_eval(y_test, predictions))
