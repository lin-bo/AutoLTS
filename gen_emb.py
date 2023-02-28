# coding: utf-8
# Author: Bo Lin

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from utils import StreetviewDataset
from model import MoCoEmb, MoCoEmbV3


def gen_emb(dim=128, device='mps', checkpoint_name=None, local=False, purpose='test', toy=False, batch_size=32, MoCoVersion=2, loc=None):
    if MoCoVersion == 2:
        net = MoCoEmb(dim, device=device, checkpoint_name=checkpoint_name, local=local).to(device)
    elif MoCoVersion == 3:
        net = MoCoEmbV3(dim, device=device, checkpoint_name=checkpoint_name, local=local).to(device)
    else:
        raise ValueError('MoCo version not found')
    loader = DataLoader(StreetviewDataset(purpose=purpose, toy=toy, local=local, augmentation=False, biased_sampling=False, loc=loc),
                        batch_size=batch_size, shuffle=False)
    # forward
    net.eval()
    embs = []
    print(f'Generating embeddings for the {purpose} set')
    for x, _ in tqdm(loader):
        x = x.to(device)
        net.zero_grad()
        emb = net.forward(x)
        embs += emb.to(device='cpu').tolist()
    embs = np.array(embs)
    np.savetxt(f'./emb/{checkpoint_name}_{purpose}.txt', embs, delimiter=',')


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='device for training')
    parser.add_argument('--checkpoint', type=str, help='checkpoint name {JobID}_{Epoch}')
    parser.add_argument('--local', action='store_true', help='is the training on a local device or not')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.add_argument('--toy', action='store_true', help='use the toy example or not')
    parser.add_argument('--no-toy', dest='toy', action='store_false')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size')
    parser.add_argument('--MoCoVersion', default=2, type=int, help='choose from 2 and 3')
    parser.add_argument('--location', default=None, type=str, help='the location of the test network, choose from None, scarborough, york, and etobicoke')
    args = parser.parse_args()

    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local, loc=args.location,
            purpose='training', toy=args.toy, batch_size=args.batchsize, MoCoVersion=args.MoCoVersion)
    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local, loc=args.location,
            purpose='validation', toy=args.toy, batch_size=args.batchsize, MoCoVersion=args.MoCoVersion)
    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local, loc=args.location,
            purpose='test', toy=args.toy, batch_size=args.batchsize, MoCoVersion=args.MoCoVersion)

