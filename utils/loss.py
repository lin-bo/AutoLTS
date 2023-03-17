# coding: utf-8
# Author: Bo Lin

import torch
from torch import nn


class LabelMoCoLoss(nn.Module):
    """
    label aware MoCo loss
    inner = True: take summation within log
    inner = False: take summation outside log
    """

    def __init__(self, inner=True):
        super(LabelMoCoLoss, self).__init__()
        self.inner = inner

    def forward(self, logits, targets):
        """
        :param logits: N_batchsize x N_queuesize
        :param targets: N_batchsize x N_queuesize
        :return:
        """
        logits = torch.exp(logits)
        targets = targets.to(torch.float)
        denominator = logits.sum(dim=1, keepdim=True)
        logits = logits / denominator
        logits = logits * targets
        if self.inner:
            logits = logits.sum(dim=1) / targets.sum(dim=1)
            logits = - torch.log(logits)
        else:
            logits = - torch.log(logits).sum(dim=1) / targets.sum(dim=1)
        return logits.mean()


class OrdLabelMoCoLoss(nn.Module):
    """
    label aware MoCo loss
    inner = True: take summation within log
    inner = False: take summation outside log
    """

    def __init__(self, inner=True):
        super(OrdLabelMoCoLoss, self).__init__()
        self.inner = inner

    def forward(self, logits, targets):
        """
        :param logits: N_batchsize x N_queuesize
        :param targets: N_batchsize x N_queuesize
        :return:
        """
        # transform logits to probabilities
        logits = torch.exp(logits)
        denominator = logits.sum(dim=1, keepdim=True)
        logits = logits / denominator
        logits = - torch.log(logits)
        logits = logits * targets
        return logits.mean()


class MultitaskLoss(nn.Module):

    def __init__(self, contras='SupMoCo', weights=None, target_features=None):
        super(MultitaskLoss, self).__init__()
        self.weights = weights
        self.target_features = target_features
        # contrastive loss
        if contras == 'MoCo':
            self.contras_loss = 0
        elif contras == 'SupMoCo':
            self.contras_loss = LabelMoCoLoss()
        elif contras == 'OrdMoCo':
            self.contras_loss = OrdLabelMoCoLoss()
        # prediction losses
        mse_fea = {'speed_actual', 'n_lanes'}
        self.pred_losses = nn.ModuleList()
        for fea in target_features:
            loss = nn.MSELoss(reduction='mean') if fea in mse_fea else nn.CrossEntropyLoss(reduction='mean')
            self.pred_losses.append(loss)

    def forward(self, logits, targets, preds, trues):
        """
        :param logits: N_batchsize x N_queuesize
        :param targets: N_batchsize x N_queuesize
        :param preds: a list of [N_batchsize]
        :param trues: a list of [N_batchsize]
        :return:
        """
        records = []
        # contrastive loss
        loss = self.weights[0] * self.contras_loss(logits, targets)
        records.append(loss.item())
        # prediction losses
        for idx, fea in enumerate(self.target_features):
            val = self.pred_losses[idx](preds[idx], trues[idx])
            records.append(val.item())
            loss += self.weights[idx+1] * val
        records.append(loss.item())
        return loss, records
