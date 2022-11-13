import numpy as np

from utils import two_dim_visual


def load_data(encoder_name, purpose):
    X = np.loadtxt(f'./emb/{encoder_name}_{purpose}.txt', delimiter=',').astype(float)
    y = np.loadtxt('./data/LTS/lts_labels.txt').astype(int)
    indi = np.loadtxt(f'./data/{purpose}_idx.txt').astype(int)
    y = y[indi]
    return X, y


if __name__ == '__main__':
    X, y = load_data('8763825_290', 'test')
    two_dim_visual(X, y)
