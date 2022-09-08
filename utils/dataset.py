from abc import ABC

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image


class StreetviewDataset(Dataset):

    def __init__(self, purpose='training', toy=False, local=True, augmentation=False):
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
        # load images
        if local:
            img_folder = '/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset'
        else:
            img_folder = './data/streetview/dataset'
        if not augmentation:
            self.transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.PILToTensor(),

            ])
        self.img_path = [img_folder + f'/{idx}.jpg' for idx in indi]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = self.transform(img).float()
        return img, torch.tensor(self.y[idx])


