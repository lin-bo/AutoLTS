import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def multi2two_dim(feature, method='pca'):
    if method == 'svd':
        centered = feature - np.mean(feature, axis=0)
        covariance = 1.0 / feature.shape[0] * centered.T.dot(centered)
        U, S, V = np.linalg.svd(covariance)
        coord = centered.dot(U[:, 0:2])
    elif method == 'pca':
        coord = PCA(random_state=0).fit_transform(feature)[:, :2]
    return coord


def two_dim_visual(feature, y):
    # check dimension
    n, d = feature.shape
    if d != 2:
        feature = multi2two_dim(feature, method='pca')
    # extract and visualize
    colors = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red'}
    for i in range(1, 5):
        flag = y == i
        plt.scatter(feature[flag].T[0], feature[flag].T[1], color=colors[i], label=f'LTS {i}', alpha=0.3)
    # visualize points
    plt.legend()
    plt.show()
