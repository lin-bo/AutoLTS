import geopandas
import numpy as np
import argparse

from postprocessing import est_trans_prob, load_roads, causal_adaptation, causal_adaptation_full_network

np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    return args


def save_updated_predictions(corrected, loc, target):
    if loc is None:
        indi_train = np.loadtxt(f'./data/training_idx.txt').astype(int)
        indi_vali = np.loadtxt(f'./data/validation_idx.txt').astype(int)
        indi_test = np.loadtxt(f'./data/test_idx.txt').astype(int)
        np.savetxt(f'./pred/{target}_training_updated_tmp.txt', corrected[indi_train])
        np.savetxt(f'./pred/{target}_validation_updated_tmp.txt', corrected[indi_vali])
        np.savetxt(f'./pred/{target}_test_updated_tmp.txt', corrected[indi_test])
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


def process(location, save):
    if location is None:
        roads_train = load_roads(loc=None, purpose='training')
        roads_full = load_roads(loc=None, purpose=None)
        corrected = causal_adaptation_full_network(roads_train, roads_full, loc=None, target='speed_actual_onehot', quiet=False)
    else:
        roads_train = load_roads(loc=location, purpose='training')
        roads_test = load_roads(loc=location, purpose='test')
        corrected = causal_adaptation(roads_train, roads_test, loc='scarborough', target='cyc_infras_onehot', quiet=False)
    if save:
        save_updated_predictions(corrected, loc=None, target='speed_actual_onehot')


if __name__ == '__main__':
    args = parse_args()
    process(location=args.location, save=args.save)
