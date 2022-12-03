import torch


def accuracy(y_pred, y_true):
    score = (y_pred == y_true).sum() / y_true.shape[0] * 100
    return score.item()


def agg_accuracy(y_pred, y_true):
    flag_pred = (y_pred <= 2).to(torch.long)
    flag_true = (y_true <= 2).to(torch.long)
    score = accuracy(flag_pred, flag_true)
    return score


def mae(y_pred, y_true):
    score = torch.abs(y_pred - y_true).to(torch.float).mean()
    return score.item()


def mse(y_pred, y_true):
    score = ((y_pred - y_true) ** 2).mean()
    return score.item()


def ob(y_pred, y_true, k=1):
    score = (torch.abs(y_pred - y_true) <= k).sum() / y_true.shape[0] * 100
    return score.item()


def kt(y_pred, y_true):
    pred_mat = torch.sign(y_pred.reshape((-1, 1)) - y_pred.reshape((1, -1)))
    true_mat = torch.sign(y_true.reshape((-1, 1)) - y_true.reshape((1, -1)))
    score = ((pred_mat == true_mat).sum() - y_true.shape[0])/2
    score /= ((y_pred.shape[0] - 1)** 2)
    return score.item()


def fhr(y_pred, y_true):
    n_low_stress = (y_true <= 2).sum()
    false_high_stress = ((y_pred >= 3) * (y_true <= 2)).sum()
    score = false_high_stress / n_low_stress
    return score.item()


def flr(y_pred, y_true):
    n_high_stress = (y_true >= 3).sum()
    false_low_stress = ((y_pred <= 2) * (y_true >= 3)).sum()
    score = false_low_stress / n_high_stress
    return score.item()
