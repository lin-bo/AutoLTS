import numpy as np

from .functions import load_and_organize_label_predictions, load_and_organize_prob_predictions

np.random.seed(0)


def est_trans_prob(target, roads, preds=None, report=True):
    """
    estimate the transition probabilities
    :param target: the target of interest (e.g. lts or speed_actual_onehot)
    :param roads: lists of roads, each road is a list of road segments
    :param preds: list of predicted values of the target associated with each road segment in the city
    :param report: whether or not to report the true probabilities, if False, assign a small probability to each zero entry
    :return: the transition probability matrix
    """
    # load tagets
    if preds is None:
        if target == 'lts':
            targets = np.loadtxt('./data/LTS/lts_labels.txt').astype(int)
        else:
            targets = np.loadtxt(f'./data/road/{target}.txt', delimiter=',').astype(int)
            targets = np.argmax(targets, axis=1)
    else:
        targets = preds
    # estimate
    dims = {'speed_actual_onehot':4, 'volume_onehot': 2, 'road_type_onehot': 4, 'cyc_infras_onehot': 4,
            'n_lanes_onehot':5, 'parking_onehot': 2, 'oneway_onehot': 2, 'lts': 4, 'lts_wo_volume': 4}
    n = dims[target]
    cnt = np.zeros((n, n))
    for road in roads:
        for i in range(len(road) - 1):
            cnt[targets[road[i]]-1, targets[road[i+1]]-1] += 1
            cnt[targets[road[i+1]]-1, targets[road[i]]-1] += 1  # to make sure that the matrix is symmetric
    if report:
        return (cnt / (cnt.sum(axis=1, keepdims=True) + 0.01)).round(2)  # add 0.01 to avoid zero denominator
    else:
        return (cnt / (cnt.sum(axis=1, keepdims=True) + 0.01) + 0.01).round(2)  # add 0.01 to avoid zero probability


def causal_adaptation(roads_train, roads_test, loc, target, quiet=False):
    # load indices
    if loc is not None:
        indices = np.loadtxt(f'./data/{loc}_test_idx.txt', delimiter=',').astype(int)
    else:
        indices = np.loadtxt(f'./data/test_idx.txt', delimiter=',').astype(int)
    # load predictions and true values
    prob_preds = load_and_organize_prob_predictions(target=target, kind='pred', loc=loc)
    target_trues = load_and_organize_label_predictions(target=target, kind='true', loc=loc)
    target_predictions = prob_preds.argmax(axis=1)
    # get trans_prob
    trans_prob = est_trans_prob(target=target, roads=roads_train, preds=target_trues, report=False)
    # iteratively adapt
    _, dim = prob_preds.shape
    for road in roads_test:
        y = prob_preds[road].argmax(axis=1)
        y_hat = -1
        while (y == y_hat).sum() != len(y):
            y_hat = y.copy()
            for i, e in enumerate(road):
                if i == 0:
                    probs = [trans_prob[j, y_hat[i+1]] * prob_preds[e][j] for j in range(dim)]
                elif i == len(road) - 1:
                    probs = [trans_prob[j, y_hat[i-1]] * prob_preds[e][j] for j in range(dim)]
                else:
                    probs = [trans_prob[j, y_hat[i+1]] * trans_prob[j, y_hat[i-1]] * prob_preds[e][j] for j in range(dim)]
                y[i] = np.argmax(probs)
                prob_preds[e] = np.array(probs).copy()
    updated_preds = prob_preds.argmax(axis=1)
    # assess
    updated_cnt = (1 - updated_preds[indices] == target_predictions[indices]).sum()
    if not quiet:
        print('# of labels corrected:', updated_cnt)
        print(f'transition matrix from training:')
        print(trans_prob)
    old_mat = est_trans_prob(target=target, roads=roads_test, preds=target_predictions)
    if not quiet:
        print(f'old transition matrix in {loc}')
        print(old_mat)
    new_mat = est_trans_prob(target=target, roads=roads_test, preds=updated_preds)
    if not quiet:
        print(f'new transition matrix in {loc}')
        print(new_mat)
    old_acc = ((target_trues[indices] == target_predictions [indices]).sum() / len(indices) * 100).round(2)
    new_acc = ((target_trues[indices] == updated_preds[indices]).sum() / len(indices) * 100).round(2)
    old_cnt = (target_trues[indices] == target_predictions[indices]).sum()
    new_cnt = (target_trues[indices] == updated_preds[indices]).sum()
    if not quiet:
        print('old acc', old_acc)
        print('new acc', new_acc)
        print('old # correct', old_cnt)
        print('new # correct', new_cnt)
    return updated_preds


def causal_adaptation_full_network(roads_train, roads_full, loc, target, quiet=False):
    # load indices
    if loc is not None:
        indices = np.loadtxt(f'./data/{loc}_test_idx.txt', delimiter=',').astype(int)
    else:
        indices = np.loadtxt(f'./data/test_idx.txt', delimiter=',').astype(int)
    # load predictions and true values
    prob_preds = load_and_organize_prob_predictions(target=target, kind='pred', loc=loc)
    target_trues = load_and_organize_label_predictions(target=target, kind='true', loc=loc)
    target_predictions = prob_preds.argmax(axis=1)
    # get trans_prob
    trans_prob = est_trans_prob(target=target, roads=roads_train, preds=target_trues, report=False)
    # fix training segments
    fixed_segments = set()
    for road in roads_train:
        for segment in road:
            fixed_segments.add(segment)
    # iteratively adapt
    _, dim = prob_preds.shape
    for road in roads_full:
        y = prob_preds[road].argmax(axis=1)
        y_hat = -1
        while (y == y_hat).sum() != len(y):
            y_hat = y.copy()
            for i, e in enumerate(road):
                if e in fixed_segments:
                    prob_preds[e] = np.zeros(dim)
                    prob_preds[e][y[i]] = 1
                else:
                    if i == 0:
                        probs = [trans_prob[j, y_hat[i+1]] * prob_preds[e][j] for j in range(dim)]
                    elif i == len(road) - 1:
                        probs = [trans_prob[j, y_hat[i-1]] * prob_preds[e][j] for j in range(dim)]
                    else:
                        probs = [trans_prob[j, y_hat[i+1]] * trans_prob[j, y_hat[i-1]] * prob_preds[e][j] for j in range(dim)]
                    y[i] = np.argmax(probs)
                    prob_preds[e] = np.array(probs).copy()
    updated_preds = prob_preds.argmax(axis=1)
    # assess
    updated_cnt = (1 - updated_preds[indices] == target_predictions[indices]).sum()
    if not quiet:
        print('# of labels corrected:', updated_cnt)
        print(f'transition matrix from training:')
        print(trans_prob)
    full_mix = target_predictions.copy()
    full_mix[~indices] = target_trues[~indices]
    old_mat = est_trans_prob(target=target, roads=roads_full, preds=full_mix)
    if not quiet:
        print(f'old transition matrix in {loc}')
        print(old_mat)
    full_mix = target_predictions.copy()
    full_mix[~indices] = target_trues[~indices]
    full_mix[indices] = updated_preds[indices]
    new_mat = est_trans_prob(target=target, roads=roads_full, preds=full_mix)
    if not quiet:
        print(f'new transition matrix in {loc}')
        print(new_mat)
    old_acc = ((target_trues[indices] == target_predictions [indices]).sum() / len(indices) * 100).round(2)
    new_acc = ((target_trues[indices] == updated_preds[indices]).sum() / len(indices) * 100).round(2)
    old_cnt = (target_trues[indices] == target_predictions[indices]).sum()
    new_cnt = (target_trues[indices] == updated_preds[indices]).sum()
    if not quiet:
        print('old acc', old_acc)
        print('new acc', new_acc)
        print('old # correct', old_cnt)
        print('new # correct', new_cnt)
    return updated_preds
