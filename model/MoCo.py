import torch
from torch import nn
import torchvision


class MoCoV0(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4,
                 mlp=True, local=True, device='mps', simple_shuffle=False):
        super(MoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.vali = False

    def forward(self, im_q, im_k):
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
        # compute positive and negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        # set the label to the positive key indicator
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k)
        return logits, labels

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr
        # self.queue[:, ptr: ptr + batch_size] = keys.T
        if ptr == 0:
            self.queue = torch.cat([keys.T, self.queue[:, ptr+batch_size:]], dim=1)
        elif ptr + batch_size == self.queue_size:
            self.queue = torch.cat([self.queue[:, :ptr], keys.T], dim=1)
        else:
            self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr+batch_size:]], dim=1)
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


class MoCo(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4,
                 mlp=True, local=True, device='mps', simple_shuffle=False):
        super(MoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        self.mlp = mlp
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        if mlp:
            # remove the last fc layer
            self.encoder_q = torch.nn.Sequential(*(list(self.encoder_q.children())[:-1]), nn.ReLU())
            self.encoder_k = torch.nn.Sequential(*(list(self.encoder_k.children())[:-1]), nn.ReLU())
            # create a projector
            self.proj = torch.nn.Sequential(nn.Linear(dim_mlp, dim))
        else:
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        # sync the two encoders
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.vali = False

    def forward(self, im_q, im_k):
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
            else:
                k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
        # go through the projector
        if self.mlp:
            q = torch.flatten(q, 1)
            k = torch.flatten(k, 1)
            q = self.proj(q)
            q = nn.functional.normalize(q, dim=1)
            k = self.proj(k)
            k = nn.functional.normalize(k, dim=1)
        # compute positive and negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        # set the label to the positive key indicator
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k)
        return logits, labels

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr
        # self.queue[:, ptr: ptr + batch_size] = keys.T
        if ptr == 0:
            self.queue = torch.cat([keys.T, self.queue[:, ptr+batch_size:]], dim=1)
        elif ptr + batch_size == self.queue_size:
            self.queue = torch.cat([self.queue[:, :ptr], keys.T], dim=1)
        else:
            self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr+batch_size:]], dim=1)
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


class LabelMoCoV0(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4,
                 mlp=True, local=True, device='mps', simple_shuffle=False):
        super(LabelMoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_lts', - torch.ones(self.queue_size, dtype=torch.long))
        self.vali = False

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
        # compute positive and negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_other = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_other], dim=1) / self.temperature
        # set the label to the positive key indicator
        with torch.no_grad():
            targets = (self.queue_lts.repeat(batch_size, 1) == torch.unsqueeze(label, 1)).to(torch.long)
            targets = torch.cat([torch.unsqueeze(torch.ones(batch_size, dtype=torch.long), 1).to(self.device), targets], dim=1)
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k, label)
        return logits, targets

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


class LabelMoCo(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4,
                 mlp=True, local=True, device='mps', simple_shuffle=False):
        super(LabelMoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        self.mlp = mlp
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        if mlp:
            # remove the last fc layer
            self.encoder_q = torch.nn.Sequential(*(list(self.encoder_q.children())[:-1]), nn.ReLU())
            self.encoder_k = torch.nn.Sequential(*(list(self.encoder_k.children())[:-1]), nn.ReLU())
            # create a projector
            self.proj = torch.nn.Sequential(nn.Linear(dim_mlp, dim))
        else:
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_lts', - torch.ones(self.queue_size, dtype=torch.long))
        self.vali = False

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
        # go through the projector
        if self.mlp:
            q = torch.flatten(q, 1)
            k = torch.flatten(k, 1)
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
            targets = (self.queue_lts.repeat(batch_size, 1) == torch.unsqueeze(label, 1)).to(torch.long)
            targets = torch.cat([torch.unsqueeze(torch.ones(batch_size, dtype=torch.long), 1).to(self.device), targets], dim=1)
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k, label)
        return logits, targets

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


class OrdLabelMoCoV0(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4, alpha=1,
                 mlp=True, local=True, device='mps', simple_shuffle=False, weight_func='exp'):
        super(OrdLabelMoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        self.alpha = alpha
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_lts', - torch.ones(self.queue_size, dtype=torch.long))
        self.vali = False
        self.weight_func = weight_func

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
        # compute positive and negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_other = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_other], dim=1) / self.temperature
        # set the label to the positive key indicator
        with torch.no_grad():
            if self.weight_func == 'exp':
                targets = torch.exp((- torch.abs(self.queue_lts.repeat(batch_size, 1) - torch.unsqueeze(label, 1)) * self.alpha).to(torch.float))
            elif self.weight_func == 'rec':
                targets = 1 / ((1 + torch.abs(self.queue_lts.repeat(batch_size, 1) - torch.unsqueeze(label, 1))) ** self.alpha)
            else:
                raise ValueError(f'weighting function {self.weight_func} not found')
            targets = torch.cat([torch.unsqueeze(torch.ones(batch_size, dtype=torch.float), 1).to(self.device), targets], dim=1)
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k, label)
        return logits, targets

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


class OrdLabelMoCo(nn.Module):

    def __init__(self, dim, queue_size=6400, momentum=0.999, temperature=0.07, n_chunk=4, alpha=1,
                 mlp=True, local=True, device='mps', simple_shuffle=False, weight_func='exp'):
        super(OrdLabelMoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.simple_shuffle = simple_shuffle
        self.n_chunk = n_chunk
        self.alpha = alpha
        self.mlp = mlp
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        self.encoder_k = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        if mlp:
            # remove the last fc layer
            self.encoder_q = torch.nn.Sequential(*(list(self.encoder_q.children())[:-1]), nn.ReLU())
            self.encoder_k = torch.nn.Sequential(*(list(self.encoder_k.children())[:-1]), nn.ReLU())
            # create a projector
            self.proj = torch.nn.Sequential(nn.Linear(dim_mlp, dim))
        else:
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the negative example queue
        self.register_buffer('queue', torch.randn(dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_lts', - torch.ones(self.queue_size, dtype=torch.long))
        self.vali = False
        self.weight_func = weight_func

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
            else:
                k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
        # go through the projector
        if self.mlp:
            q = torch.flatten(q, 1)
            k = torch.flatten(k, 1)
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
            if self.weight_func == 'exp':
                targets = torch.exp((- torch.abs(self.queue_lts.repeat(batch_size, 1) - torch.unsqueeze(label, 1)) * self.alpha).to(torch.float))
            elif self.weight_func == 'rec':
                targets = 1 / ((1 + torch.abs(self.queue_lts.repeat(batch_size, 1) - torch.unsqueeze(label, 1))) ** self.alpha)
            else:
                raise ValueError(f'weighting function {self.weight_func} not found')
            targets = torch.cat([torch.unsqueeze(torch.ones(batch_size, dtype=torch.float), 1).to(self.device), targets], dim=1)
        # dequeue and enqueue
        if not self.vali:
            self.dequeue_and_enqueue(k, label)
        return logits, targets

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


class MoCoEmb(nn.Module):

    def __init__(self, dim, device='mps', checkpoint_name=None, local=False):
        super(MoCoEmb, self).__init__()
        # initialize parameters
        self.device = device
        # initialize encoders
        self.encoder_q = torchvision.models.resnet50(pretrained=True)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        if local:
            checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt', map_location='cpu')
        else:
            checkpoint = torch.load(f'./checkpoint/{checkpoint_name}.pt')
        state_dict = checkpoint['model_state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('encoder_q'):
                state_dict[k[len('encoder_q.'):]] = state_dict[k]
            del state_dict[k]
        _ = self.encoder_q.load_state_dict(state_dict, strict=False)

    def forward(self, img):
        # compute query features
        q = self.encoder_q(img)
        q = nn.functional.normalize(q, dim=1)
        return q


class MoCoEmbV3(nn.Module):

    def __init__(self, dim, device='mps', checkpoint_name=None, local=False):
        super(MoCoEmbV3, self).__init__()
        # initialize parameters
        self.device = device
        # initialize the model
        self.img_encoder = torchvision.models.resnet50(pretrained=True)
        self.img_encoder = torch.nn.Sequential(*(list(self.img_encoder.children())[:-1]), nn.ReLU())
        for name, param in self.img_encoder.named_parameters():
            param.requires_grad = False
        # # load the checkpoint
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

    def forward(self, img):
        # compute query features
        q = self.img_encoder(img)
        q = torch.flatten(q, 1)
        q = nn.functional.normalize(q, dim=1)
        return q
