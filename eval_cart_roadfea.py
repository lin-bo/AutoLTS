import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from eval import model_eval

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--grid', action='store_true', default=False)
    args = parser.parse_args()
    return args


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


def best_param(location, sce):
    params = {
        None: {2: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 0.01},
               3: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 0.01}},
        'york': {2: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 0.01},
                 3: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 0.01}},
        'etobicoke': {2: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 2},
                      3: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 0.01}},
        'scarborough': {2: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2},
                        3: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 0.01}}
    }
    return params[location][sce]


def grid_search(X_train, y_train):
    dt_clf = DecisionTreeClassifier(random_state=0)
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    clf = GridSearchCV(dt_clf, params, cv=10)
    search = clf.fit(X_train, y_train)
    print('best params', search.best_params_)
    return search.best_params_


def load_data(location, sce):
    indi_train, indi_vali, indi_test = load_indis(location)
    df_true = pd.read_csv('./data/network/final.csv')
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')
    if sce == 3:
        df_true['speed_limit'] = df_true['speed_limit'].replace({0: 40, 40: 40, 20: 40, 30: 40, 10: 40, 60: 60, 15: 40, 25: 40, 50: 60, 70: 60, 80: 60})
        X = df_true[['speed_limit', 'nlanes_1', 'nlanes_2', 'nlanes_3', 'nlanes_4', 'nlanes_5']].values
    elif sce == 2:
        X = df_true[['Arterial', 'Local', 'Other', 'Trail', 'Bike Lanes',
                     'Cycle Tracks', 'Multi-use Pathway', 'No Infras',
                     'oneway']].values
    else:
        raise ValueError()
    X_train, X_test = X[indi_train], X[indi_test]
    y_train, y_test = y[indi_train], y[indi_test]
    return X_train, X_test, y_train, y_test


def solve_and_eval(location=None, grid=False):
    for sce in [2, 3]:
        # load data
        X_train, X_test, y_train, y_test = load_data(location, sce)
        # Grid Search
        param = grid_search(X_train, y_train) if grid else best_param(location=location, sce=2)
        # generate results
        clf = DecisionTreeClassifier(random_state=0,
                                     max_depth=param['max_depth'],
                                     min_samples_split=param['min_samples_split'],
                                     criterion=param['criterion']).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f'\nscenario {sce} model eval:')
        res_sce, conf_mat_sce = model_eval(y_test, y_pred)
        print(res_sce)
        print(conf_mat_sce)


if __name__ == '__main__':
    args = parse_args()
    solve_and_eval(location=args.location, grid=args.grid)



