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
        return logits.sum()

