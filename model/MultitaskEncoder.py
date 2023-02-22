import torch
from torch import nn
import torchvision

# from utils import fea2dim


def fea2dim(fea):
    l2d = {'lts': 4,
           'oneway': 1, 'oneway_onehot': 2,
           'parking': 1, 'parking_onehot': 2,
           'volume': 1, 'volume_onehot': 2,
           'speed_actual': 1, 'speed_actual_onehot': 4,
           'cyc_infras': 2, 'cyc_infras_onehot': 4,
           'n_lanes': 1, 'n_lanes_onehot': 5,
           'road_type': 9, 'road_type_onehot': 4}
    return l2d[fea]


class MultiTaskEncoder(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4, target_features=None,
                 device='mps', simple_shuffle=False, MoCo_type='sup', weight_func='exp', alpha=2):
        super(MultiTaskEncoder, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        self.MoCo_type = MoCo_type
        self.weight_func = weight_func
        self.alpha = alpha
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        # remove the last fc layer
        self.encoder_q = torch.nn.Sequential(*(list(self.encoder_q.children())[:-1]), nn.ReLU())
        self.encoder_k = torch.nn.Sequential(*(list(self.encoder_k.children())[:-1]), nn.ReLU())
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create a projector
        self.proj = torch.nn.Sequential(nn.Linear(dim_mlp, dim))
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_lts', - torch.ones(self.queue_size, dtype=torch.long))
        self.vali = False
        # create a feature predictor
        self.target_features = target_features
        self.predictors = nn.ModuleList()
        for fea in target_features:
            self.predictors.append(torch.nn.Sequential(nn.Linear(dim_mlp, 32), nn.ReLU(), nn.Linear(32, fea2dim(fea))))

    def forward(self, im_q, im_k, label):
        batch_size = im_q.shape[0]
        # compute query features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():
            # update the key encoder
            self._momentum_update_key_encoder()
            # map the keys
            if self.simple_shuffle:
                chunk_size = int(im_k.shape[0] / self.n_chunk)
                # shuffle the keys
                im_k, idx_unshuffle = self._simple_shuffle(im_k)
                for j in range(self.n_chunk):
                    k_chunks = [self.encoder_k(im_k[chunk_size * j: chunk_size * (j + 1)]) for j in range(self.n_chunk)]
                k = torch.cat(k_chunks, dim=0)
                # un-shuffle the keys
                k = self._simple_unshuffle(k, idx_unshuffle)
                k = nn.functional.normalize(k, dim=1)
            else:
                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)
        # flatten
        q = torch.flatten(q, 1)
        k = torch.flatten(k, 1)
        # predictions
        preds = []
        for predictor in self.predictors:
            preds.append(predictor(q))
        # go through the projector
        q = self.proj(q)
        q = nn.functional.normalize(q, dim=1)
        k = self.proj(k)
        k = nn.functional.normalize(k, dim=1)
        # compute positive and negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_other = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_other], dim=1) / self.temperature
        # set the label to the positive key indicator
        with torch.no_grad():
            if self.MoCo_type == 'sup':
                targets = (self.queue_lts.repeat(batch_size, 1) == torch.unsqueeze(label, 1)).to(torch.long)
                targets = torch.cat([torch.unsqueeze(torch.ones(batch_size, dtype=torch.long), 1).to(self.device), targets], dim=1)
            elif self.MoCo_type == 'ord':
                if self.weight_func == 'exp':
                    targets = torch.exp((- torch.abs(self.queue_lts.repeat(batch_size, 1) - torch.unsqueeze(label, 1)) * self.alpha).to(torch.float))
                elif self.weight_func == 'rec':
                    targets = 1 / ((1 + torch.abs(self.queue_lts.repeat(batch_size, 1) - torch.unsqueeze(label, 1))) ** self.alpha)
                else:
                    raise ValueError(f'weighting function {self.weight_func} not found')
                targets = torch.cat([torch.unsqueeze(torch.ones(batch_size, dtype=torch.float), 1).to(self.device), targets], dim=1)
            else:
                raise ValueError('MoCo type not found')
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k, label)
        return logits, targets, preds

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr
        # self.queue[:, ptr: ptr + batch_size] = keys.T
        if ptr == 0:
            self.queue = torch.cat([keys.T, self.queue[:, ptr+batch_size:]], dim=1)
            self.queue_lts = torch.cat([label, self.queue_lts[ptr+batch_size: ]], dim=0)
        elif ptr + batch_size == self.queue_size:
            self.queue = torch.cat([self.queue[:, :ptr], keys.T], dim=1)
            self.queue_lts = torch.cat([self.queue_lts[:ptr], label], dim=0)
        else:
            self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr + batch_size:]], dim=1)
            self.queue_lts = torch.cat([self.queue_lts[:ptr], label, self.queue_lts[ptr + batch_size:]], dim=0)
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    @torch.no_grad()
    def _simple_shuffle(self, im_k):
        batch_size = im_k.shape[0]
        idx_shuffle = torch.randperm(batch_size).to(device=self.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return im_k[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _simple_unshuffle(self, k, idx_unshuffle):
        return k[idx_unshuffle]
