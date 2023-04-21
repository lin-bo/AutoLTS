import numpy as np
import pandas as pd
import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from eval import model_eval, attr_mapping, load_fea


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--grid', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    return args


def grid_search(X_train, y_train):
    dt_clf = DecisionTreeClassifier(random_state=0)
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    clf = GridSearchCV(dt_clf, params, cv=10)
    search = clf.fit(X_train, y_train)
    print('best params', search.best_params_)
    return search.best_params_


def best_param(location, sce):
    params = {
        None: {1:  {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 4},
               2: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 2},
               3: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6}},
        'york': {1: {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 4},
                 2: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2},
                 3: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 4}},
        'etobicoke': {1: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2},
                      2: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 2},
                      3: {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 2}},
        'scarborough': {1: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.05},
                        2: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 2},
                        3: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 6}}
    }
    return params[location][sce]


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


def load_feature(sce, location):
    # ground truth
    df_true = pd.read_csv('./data/network/final.csv')
    indi_train, indi_vali, indi_test = load_indis(location)
    s2indi = {'training': indi_train, 'validation': indi_vali, 'test': indi_test}
    # training data
    X = []
    for purpose in ['training', 'validation', 'test']:
        speed, parking, oneway, cyc_infras, n_lanes, road_type, volume = load_fea(key='pred', purpose=purpose, loc=location, updated=True)
        if purpose == 'validation' and len(speed) > len(indi_vali):
            speed = speed[indi_vali]
        df = attr_mapping(speed, parking, oneway, cyc_infras, n_lanes, road_type, volume)
        if sce == 2:
            df[['Bike Lanes', 'Cycle Tracks', 'Multi-use Pathway', 'No Infras']] = df_true.loc[s2indi[purpose], ['Bike Lanes', 'Cycle Tracks', 'Multi-use Pathway', 'No Infras']].values
            df['oneway'] = df_true.loc[s2indi[purpose], 'oneway'].values
            df[['Arterial', 'Local', 'Other', 'Trail']] = df_true.loc[s2indi[purpose], ['Arterial', 'Local', 'Other', 'Trail']].values
        elif sce == 3:
            df['speed_actual'] = df_true.loc[s2indi[purpose], 'speed_limit'].values
            df['nlanes'] = df_true.loc[s2indi[purpose], 'nlanes'].values
        df.drop('volume', axis=1, inplace=True)
        X.append(df.values)
    return X[0], X[1], X[2]


def save_prob(clf, sce, X_train, X_vali, X_test, location):
    prob_train = clf.predict_proba(X_train)
    print(X_vali.shape)
    prob_vali = clf.predict_proba(X_vali)
    prob_test = clf.predict_proba(X_test)
    if location is None:
        np.savetxt(f'./data/step_one_feature/sce{sce}_prob_training.txt', prob_train, delimiter=',')
        np.savetxt(f'./data/step_one_feature/sce{sce}_prob_vali.txt', prob_vali, delimiter=',')
        np.savetxt(f'./data/step_one_feature/sce{sce}_prob_test.txt', prob_test, delimiter=',')
    else:
        np.savetxt(f'./data/step_one_feature/{location}_sce{sce}_prob_training.txt', prob_train, delimiter=',')
        np.savetxt(f'./data/step_one_feature/{location}_sce{sce}_prob_vali.txt', prob_vali, delimiter=',')
        np.savetxt(f'./data/step_one_feature/{location}_sce{sce}_prob_test.txt', prob_test, delimiter=',')


def predict_and_eval(location, save=False, grid=False):
    indi_train, indi_vali, indi_test = load_indis(location)
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')
    y_train, y_test = y[indi_train], y[indi_test]

    # scenario 1
    for sce in [1, 2, 3]:
        print()
        X_train, X_vali, X_test = load_feature(sce=sce, location=location)
        param = grid_search(X_train, y_train) if grid else best_param(location=location, sce=sce)
        # generate results
        clf = DecisionTreeClassifier(random_state=0,
                                     max_depth=param['max_depth'],
                                     min_samples_split=param['min_samples_split'],
                                     criterion=param['criterion']).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f'Scenario {sce} model eval:')
        res, conf_mat = model_eval(y_test, y_pred)
        print(res)
        print(conf_mat)
        if save:
            save_prob(clf, 1, X_train, X_vali, X_test, location)


if __name__ == '__main__':
    args = parse_args()
    predict_and_eval(args.location, args.save, args.grid)
