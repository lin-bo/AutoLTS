from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import StreetviewDataset
from model import MoCoEmb, MoCoEmbV3


def dim_reduction( feature, method='tsne'):
    if method == 'svd':
        centered = feature - np.mean(feature, axis=0)
        covariance = 1.0 / feature.shape[0] * centered.T.dot(centered)
        U, S, V = np.linalg.svd(covariance)
        coord = centered.dot(U[:, 0:2])
    elif method == 'pca':
        coord = PCA(random_state=0).fit_transform(feature)[:, :2]
    elif method == 'tsne':
        coord = TSNE(2, verbose=0, learning_rate='auto', init='random').fit_transform(feature)
    else:
        raise ValueError('method not found')
    return coord


def gen_emb(dim=128, device='mps', checkpoint_name=None, local=False, purpose='test', toy=False, batch_size=32, MoCoVersion=2, loc=None, reduction=True):
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
    embs = np.array(embs).astype(float)
    if reduction:
        embs = dim_reduction(embs, method='tsne')
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
    parser.add_argument('--reduction', default=True, type=bool, help='reduction')
    parser.add_argument('--no-reduction', dest='reduction', action='store_false')
    parser.add_argument('--location', default=None, type=str, help='the location of the test network, choose from None, scarborough, york, and etobicoke')
    args = parser.parse_args()

    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local, loc=args.location,
            purpose='training', toy=args.toy, batch_size=args.batchsize, MoCoVersion=args.MoCoVersion, reduction=args.reduction)
    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local, loc=args.location,
            purpose='validation', toy=args.toy, batch_size=args.batchsize, MoCoVersion=args.MoCoVersion, reduction=args.reduction)
    gen_emb(dim=128, device=args.device, checkpoint_name=args.checkpoint, local=args.local, loc=args.location,
            purpose='test', toy=args.toy, batch_size=args.batchsize, MoCoVersion=args.MoCoVersion, reduction=args.reduction)

