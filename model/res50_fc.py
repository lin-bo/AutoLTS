import torch
import torchvision
from torch import nn


class Res50FC(nn.Module):

    def __init__(self, pretrained=False, frozen=False):
        super(Res50FC, self).__init__()
        self.res50 = torchvision.models.resnet50(pretrained=pretrained)
        if frozen:
            self.res50.requires_grad = False
        self.l1 = nn.Linear(1000, 100)
        self.l2 = nn.Linear(100, 4)
        self.l3 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.res50(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
