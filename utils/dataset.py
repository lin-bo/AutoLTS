from abc import ABC

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image


class StreetviewDataset(Dataset):

    def __init__(self, purpose='training', toy=False, local=True):
        super().__init__()
        # load indices
        indi = np.loadtxt(f'./data/{purpose}_idx.txt').astype(int)
        if toy:
            np.random.seed(31415926)
            np.random.shuffle(indi)
            indi = indi[:1000]
        # load labels
        lts = np.loadtxt('./data/LTS/lts_labels.txt').astype(int)
        self.y = lts[indi]
        self.x = []
        # load images
        if local:
            img_path = '/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset'
        else:
            img_path = './data/streetview/dataset'
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(224)
        ])
        print(f'loading {purpose} images ...')
        for idx in tqdm(indi):
            img = Image.open(img_path + f'/{idx}.jpg')
            self.x.append(transform(img).float())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

