import geopandas
import numpy as np

from postprocessing import est_trans_prob, load_roads, causal_adaptation, causal_adaptation_full_network

np.random.seed(0)


def save_updated_predictions(corrected, loc, target):
    if loc is None:
        indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
        indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
        indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
        np.savetxt(f'./pred/{target}_training_updated.txt', corrected[indi_train])
        np.savetxt(f'./pred/{target}_validation_updated.txt', corrected[indi_vali])
        np.savetxt(f'./pred/{target}_test_updated.txt', corrected[indi_test])
    else:
        indi_train = np.loadtxt(f'./data/{loc}_training_idx.txt').astype(int)
        indi_vali = np.loadtxt(f'./data/{loc}_validation_idx.txt').astype(int)
        indi_test = np.loadtxt(f'./data/{loc}_test_idx.txt').astype(int)
        np.savetxt(f'./pred/{target}_training_{loc}_updated.txt', corrected[indi_train])
        np.savetxt(f'./pred/{target}_validation_{loc}_updated.txt', corrected[indi_vali])
        np.savetxt(f'./pred/{target}_test_{loc}_updated.txt', corrected[indi_test])


def adapt(loc, target, purpose):
    roads_train = load_roads(loc=None, purpose='training')
    if loc is None:
        roads_full = load_roads(loc=None, purpose=None)
        corrected = causal_adaptation_full_network(roads_train, roads_full, loc=None, target=target, quiet=False)
    else:
        roads_test = load_roads(loc=None, purpose='test')
        corrected = causal_adaptation(roads_train, roads_test, loc=loc, target=target, quiet=False)
    return corrected


if __name__ == '__main__':
    roads_train = load_roads(loc=None, purpose='training')
    roads_full = load_roads(loc=None, purpose=None)
    roads_test = load_roads(loc=None, purpose='test')
    # causal_adaptation(roads_train, roads_test, loc='scarborough', target='cyc_infras_onehot', quiet=False)
    corrected = causal_adaptation_full_network(roads_train, roads_full, loc=None, target='speed_actual_onehot', quiet=False)
    # save_updated_predictions(corrected, loc=None, target='n_lanes_onehot')
