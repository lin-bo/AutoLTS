import torch
import torchvision
from torch import nn


class MoCoClf(nn.Module):

    def __init__(self, checkpoint_name=None, local=True, vali=False, out_dim=4):
        super(MoCoClf, self).__init__()
        # initialize the model
        model = torchvision.models.resnet50(pretrained=True)
        for name, param in model.named_parameters():
            param.requires_grad = False
        # # load the checkpoint
        if not vali:
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
        # add FC layers
        dim = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim, 100), nn.ReLU(), nn.Linear(100, out_dim))
        self.clf = model

    def forward(self, x):
        return self.clf(x)


class MoCoClfV2(nn.Module):

    def __init__(self, checkpoint_name=None, local=True, vali=False, out_dim=4):
        super(MoCoClfV2, self).__init__()
        # initialize the model
        model = torchvision.models.resnet50(pretrained=True)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128))
        for name, param in model.named_parameters():
            param.requires_grad = False
        # # load the checkpoint
        if not vali:
            if local:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
            else:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
            state_dict = checkpoint['model_state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder_q'):
                    state_dict[k[len('encoder_q.'):]] = state_dict[k]
                del state_dict[k]
            _ = model.load_state_dict(state_dict, strict=False)
        # add FC layers
        self.emb = model
        self.proj = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, x):
        x = self.emb(x)
        x = nn.functional.normalize(x, dim=1)
        return self.proj(x)


class MoCoClfV3(nn.Module):

    def __init__(self, checkpoint_name=None, local=True, vali=False, out_dim=4, enc_frozen=True):
        super(MoCoClfV3, self).__init__()
        # initialize the model
        self.encoder = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]), nn.ReLU())
        if enc_frozen:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        # # load the checkpoint
        if not vali:
            if local:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
            else:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
            state_dict = checkpoint['model_state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder_q'):
                    state_dict[k[len('encoder_q.'):]] = state_dict[k]
                del state_dict[k]
            _ = self.encoder.load_state_dict(state_dict, strict=False)
        # add FC layers
        self.predictor = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.normalize(x, dim=1)
        x = torch.flatten(x, 1)
        return self.predictor(x)


class MoCoClfV2Fea(nn.Module):

    def __init__(self, checkpoint_name=None, local=True, vali=False, n_fea=1, out_dim=4):
        super(MoCoClfV2Fea, self).__init__()
        # initialize the model
        model = torchvision.models.resnet50(pretrained=True)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128))
        for name, param in model.named_parameters():
            param.requires_grad = False
        # # load the checkpoint
        if not vali:
            if local:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
            else:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
            state_dict = checkpoint['model_state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder_q'):
                    state_dict[k[len('encoder_q.'):]] = state_dict[k]
                del state_dict[k]
            _ = model.load_state_dict(state_dict, strict=False)
        # add FC layers
        self.emb = model
        self.proj = nn.Sequential(nn.Linear(128, 16), nn.ReLU())
        self.fea_emb = nn.Sequential(nn.Linear(n_fea, 16), nn.ReLU())
        self.clf = nn.Linear(16, out_dim)

    def forward(self, x, fea):
        x = self.emb(x)
        x = nn.functional.normalize(x, dim=0)
        x = self.proj(x)
        fea = self.fea_emb(fea)
        x = (x + fea) / 2
        return self.clf(x)


class MoCoClfV3Fea(nn.Module):

    def __init__(self, checkpoint_name=None, local=True, vali=False, n_fea=1, out_dim=4, enc_frozen=True):
        super(MoCoClfV3Fea, self).__init__()
        # initialize the model
        self.img_encoder = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.img_encoder.fc.weight.shape[1]
        self.img_encoder = torch.nn.Sequential(*(list(self.img_encoder.children())[:-1]), nn.ReLU())
        if enc_frozen:
            for name, param in self.img_encoder.named_parameters():
                param.requires_grad = False
        # # load the checkpoint
        if not vali:
            if local:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
            else:
                checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
            state_dict = checkpoint['model_state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder_q'):
                    state_dict[k[len('encoder_q.'):]] = state_dict[k]
                del state_dict[k]
            _ = self.img_encoder.load_state_dict(state_dict, strict=False)
        # add FC layers
        self.fea_encoder = nn.Sequential(nn.Linear(n_fea, 2048), nn.ReLU())
        self.clf = nn.Linear(2048, out_dim)

    def forward(self, x, fea):
        x = self.img_encoder(x)
        x = nn.functional.normalize(x, dim=1)
        x = torch.flatten(x, 1)
        fea = self.fea_encoder(fea)
        fea = nn.functional.normalize(fea, dim=1)
        x = (x + fea) / 2
        return self.clf(x)
