import torch
import torchvision
from torch import nn


class MoCoClf(nn.Module):

    def __init__(self, checkpoint_name=None, local=True):
        super(MoCoClf, self).__init__()
        # initialize the model
        model = torchvision.models.resnet50(pretrained=True)
        for name, param in model.named_parameters():
            param.requires_grad = False
        # # load the checkpoint
        if local:
            checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
        else:
            checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
        state_dict = checkpoint['model_state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                state_dict[k[len('encoder_q.'):]] = state_dict[k]
            del state_dict[k]
        _ = model.load_state_dict(state_dict, strict=False)
        # # add FC layers
        dim = model.fc.weight.shape[0]
        # model = nn.Sequential(model, nn.Linear(dim, 4), nn.Softmax(dim=-1))
        # self.emb = model
        # self.clf1 = nn.Linear(dim, 4)
        # self.clf2 = nn.Softmax(dim=-1)
        dim = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim, 4), nn.Softmax(dim=-1))
        self.clf = model

    def forward(self, x):
        # x = self.emb(x)
        # x = self.clf1(x)
        # return self.clf2(x)
        return self.clf(x)

