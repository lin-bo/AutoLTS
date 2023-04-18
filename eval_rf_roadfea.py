import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from eval import model_eval

if __name__ == '__main__':
    indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
    indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
    indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
    df_true = pd.read_csv('./data/network/final.csv')
    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')

    # scenario two: number of lanes + speed limit
    # load data
    df_true['speed_limit'] = df_true['speed_limit'].replace({0: 40, 40: 40, 20: 40, 30: 40, 10: 40, 60: 60, 15: 40, 25: 40, 50: 60, 70: 60, 80: 60})
    X = df_true[['speed_limit', 'nlanes_1', 'nlanes_2', 'nlanes_3', 'nlanes_4', 'nlanes_5']].values
    X_train, X_test = X[indi_train], X[indi_test]
    y_train, y_test = y[indi_train], y[indi_test]
    # Grid Search
    # dt_clf = DecisionTreeClassifier(random_state=0)
    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    # clf = GridSearchCV(dt_clf, params, cv=10)
    # search = clf.fit(X_train, y_train)
    # print(search.best_params_)
    print('best params: ', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 0.01})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=6, min_samples_split=0.01, criterion='entropy').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario two model eval:')
    res_sce2, conf_mat_sce2 = model_eval(y_test, y_pred)
    print(res_sce2)
    print(conf_mat_sce2)

    print('\n\n')
    # scenario three: road type, cyc type, oneway
    # load data
    X = df_true[['Arterial', 'Local', 'Other', 'Trail', 'Bike Lanes',
                 'Cycle Tracks', 'Multi-use Pathway', 'No Infras',
                 'oneway']].values
    X_train, X_test = X[indi_train], X[indi_test]
    y_train, y_test = y[indi_train], y[indi_test]
    # Grid search
    # dt_clf = DecisionTreeClassifier(random_state=0)
    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    # clf = GridSearchCV(dt_clf, params, cv=10)
    # search = clf.fit(X_train, y_train)
    # print(search.best_params_)
    print('best params: ', {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 0.01})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=6, min_samples_split=0.01, criterion='gini').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario two model eval:')
    res_sce2, conf_mat_sce2 = model_eval(y_test, y_pred)
    print(res_sce2)
    print(conf_mat_sce2)



