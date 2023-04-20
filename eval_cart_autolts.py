import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from eval import model_eval


def load_feature(sce, encoder='9404446_29'):
    X_train = np.loadtxt(f'./data/step_one_feature/sce{sce}_training.txt', delimiter=',')
    emb_train = np.loadtxt(f'./emb/{encoder}_training.txt', delimiter=',') * 0
    X_test = np.loadtxt(f'./data/step_one_feature/sce{sce}_test.txt', delimiter=',')
    emb_test = np.loadtxt(f'./emb/{encoder}_test.txt', delimiter=',') * 0
    return np.concatenate([X_train, emb_train], axis=1),  np.concatenate([X_test, emb_test], axis=1)


if __name__ == '__main__':
    indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
    indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
    indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)

    y = np.loadtxt('./data/LTS/lts_labels_wo_volume.txt')
    y_train, y_test = y[indi_train], y[indi_test]

    X_train, X_test = load_feature(sce=1)
    clf = DecisionTreeClassifier(random_state=0, max_depth=10).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Scenario one model eval:')
    res_sce, conf_mat_sce = model_eval(y_test, y_pred)
    print(res_sce)
    print(conf_mat_sce)
