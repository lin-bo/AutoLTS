import torch
from torch import nn
import torchvision


class MoCo(nn.Module):

    def __init__(self, dim, queue_size=8400, momentum=0.999, temperature=0.07, mlp=True, local=True, device='mps'):
        super(MoCo, self).__init__()
        # initialize parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        # initialize encoders
        if local:
            weights = 'DEFAULT'
            self.encoder_q = torchvision.models.resnet50(weights=weights)
            self.encoder_k = torchvision.models.resnet50(weights=weights)
        else:
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
            # TODO: shuffle the keys
            # map the keys
            k = self.encoder_k(im_k)
            # TODO: un-shuffle the keys
        # compute positive and negative logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.concat([l_pos, l_neg], dim=1) / self.temperature
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
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)




