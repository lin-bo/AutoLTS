import geopandas
import numpy as np

from postprocessing import est_trans_prob, load_roads

np.random.seed(0)


if __name__ == '__main__':
    roads_all = load_roads(loc=None, purpose='training')
    trans_prob_mat = est_trans_prob(target='speed_actual_onehot', roads=roads_all, preds=None, report=True)
    print(trans_prob_mat)
