import torch
import torchvision
from torch import nn


class Res50FC(nn.Module):

    def __init__(self, pretrained=False, frozen=False):
        super(Res50FC, self).__init__()
        self.res50 = torchvision.models.resnet50(pretrained=pretrained)
        if frozen:
            for name, param in self.res50.named_parameters():
                param.requires_grad = False
        dim = self.res50.fc.weight.shape[1]
        self.res50.fc = nn.Sequential(nn.Linear(dim, 100), nn.ReLU(), nn.Linear(100, 4))

    def forward(self, x):
        return self.res50(x)
