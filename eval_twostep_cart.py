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
    X = []
    for purpose in ['training', 'validation', 'test']:
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
        X.append(df.values)
    return X


if __name__ == '__main__':
    # load targets
    indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
    indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
    indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')
    y_train, y_test = y[indi_train], y[indi_test]

    # scenario 1
    X_train, X_vali, X_test = load_feature(sce=1)
    # load data
    # Grid Search
    # dt_clf = DecisionTreeClassifier(random_state=0)
    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    # clf = GridSearchCV(dt_clf, params, cv=10)
    # search = clf.fit(X_train, y_train)
    # print(search.best_params_)
    print('best params: ', {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 4})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_split=4, criterion='entropy').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario one model eval:')
    res_sce, conf_mat_sce = model_eval(y_test, y_pred)
    print(res_sce)
    print(conf_mat_sce)

    # scenario 2
    X_train, X_vali, X_test = load_feature(sce=2)
    # load data
    # Grid Search
    # dt_clf = DecisionTreeClassifier(random_state=0)
    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    # clf = GridSearchCV(dt_clf, params, cv=10)
    # search = clf.fit(X_train, y_train)
    # print(search.best_params_)
    print('best params: ', {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 2})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=8, min_samples_split=2, criterion='gini').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario one model eval:')
    res_sce, conf_mat_sce = model_eval(y_test, y_pred)
    print(res_sce)
    print(conf_mat_sce)
