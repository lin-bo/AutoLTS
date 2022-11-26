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


class Res50FCFea(nn.Module):

    def __init__(self, pretrained=False, frozen=False, n_fea=1):
        super(Res50FCFea, self).__init__()
        self.res50 = torchvision.models.resnet50(pretrained=pretrained)
        if frozen:
            for name, param in self.res50.named_parameters():
                param.requires_grad = False
        dim = self.res50.fc.weight.shape[1]
        self.res50.fc = nn.Sequential(nn.Linear(dim, 100), nn.ReLU(), nn.Linear(100, 16), nn.ReLU())
        self.fea_emb = nn.Sequential(nn.Linear(n_fea, 16), nn.ReLU())
        self.clf = nn.Linear(16, 4)

    def forward(self, x, fea):
        x = self.res50(x)
        fea = self.fea_emb(fea)
        x = (x + fea) / 2
        return self.clf(x)
