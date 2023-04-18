import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from eval import model_eval


def load_feature(sce):
    X_train = np.loadtxt(f'./data/step_one_feature/sce{sce}_training.txt', delimiter=',')
    X_vali = np.loadtxt(f'./data/step_one_feature/sce{sce}_validation.txt', delimiter=',')
    X_test = np.loadtxt(f'./data/step_one_feature/sce{sce}_test.txt', delimiter=',')
    return X_train, X_vali, X_test


if __name__ == '__main__':
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
    print('best params: ', {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=2, criterion='gini').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario one model eval:')
    res_sce, conf_mat_sce = model_eval(y_test, y_pred)
    print(res_sce)
    print(conf_mat_sce)

    print('\n\n')
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
    print('best params: ', {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 6})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=7, min_samples_split=6, criterion='gini').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario two model eval:')
    res_sce, conf_mat_sce = model_eval(y_test, y_pred)
    print(res_sce)
    print(conf_mat_sce)

    print('\n\n')
    # scenario 3
    X_train, X_vali, X_test = load_feature(sce=3)
    # load data
    # Grid Search
    # dt_clf = DecisionTreeClassifier(random_state=0)
    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'min_samples_split': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 2, 4, 6]}
    # clf = GridSearchCV(dt_clf, params, cv=10)
    # search = clf.fit(X_train, y_train)
    # print(search.best_params_)
    print('best params: ', {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6})
    # generate results
    clf = DecisionTreeClassifier(random_state=0, max_depth=8, min_samples_split=6, criterion='gini').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario three model eval:')
    res_sce, conf_mat_sce = model_eval(y_test, y_pred)
    print(res_sce)
    print(conf_mat_sce)

