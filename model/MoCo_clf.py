import torch
import torchvision
from torch import nn


class MoCoClf(nn.Module):

    def __init__(self, checkpoint_name=None, local=True):
        super(MoCoClf, self).__init__()
        # initialize the model
        model = torchvision.models.resnet50(pretrained=True)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128))
        for name, param in model.named_parameters():
            param.requires_grad = False
        # # load the checkpoint
        if local:
            checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
        else:
            checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
        state_dict = checkpoint['model_state_dict']
        for k in list(state_dict.keys()):
            # if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            if k.startswith('encoder_q'):
                state_dict[k[len('encoder_q.'):]] = state_dict[k]
            del state_dict[k]
        _ = model.load_state_dict(state_dict, strict=False)
        # add FC layers
        # dim = model.fc.weight.shape[1]
        # model.fc = nn.Sequential(nn.Linear(dim, 100), nn.ReLU(), nn.Linear(100, 4))
        # self.clf = model
        self.emb = model
        self.pred = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        #return self.clf(x)
        x = self.emb(x)
        x = nn.functional.normalize(x, dim=1)
        return torch.einsum('nc,ck->nk', [x, x.T]) * (1 - torch.eye(x.shape[0])).to(device='mps')
        x = self.pred(x)
        return x

