# coding: utf-8
# Author: Bo Lin

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from utils import StreetviewDataset
from model import MoCoEmb


def gen_emb(dim=128, device='mps', checkpoint_name=None, local=False, purpose='test', toy=False, batch_size=32):
    net = MoCoEmb(dim, device=device, checkpoint_name=checkpoint_name, local=local).to(device)
    loader = DataLoader(StreetviewDataset(purpose=purpose, toy=toy, local=local, augmentation=False, biased_sampling=False),
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
    args = parser.parse_args()

    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local,
            purpose='training', toy=args.toy, batch_size=args.batchsize)
    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local,
            purpose='validation', toy=args.toy, batch_size=args.batchsize)
    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local,
            purpose='test', toy=args.toy, batch_size=args.batchsize)

