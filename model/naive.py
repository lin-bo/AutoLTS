from torch import nn


class FeaFC(nn.Module):

    def __init__(self, n_fea=1):
        super(FeaFC, self).__init__()
        self.fea_emb = nn.Sequential(nn.Linear(n_fea, 4))
        # self.clf = nn.Linear(16, 4)

    def forward(self, x, fea):
        fea = self.fea_emb(fea)
        return fea
