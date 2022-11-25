import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
from utils import GaussianBlur


class StreetviewDataset(Dataset):

    def __init__(self, purpose='training', toy=False, local=True, augmentation=False, biased_sampling=False, return_speed=False):
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
        # posted speed
        self.return_speed = return_speed
        if return_speed:
            speed = np.loadtxt('./data/road/speed_limit.txt').astype(np.double)
            self.speed = speed[indi]
            mu = self.speed.mean()
            std = self.speed.std()
            self.speed -= mu
            self.speed /= std
            self.speed = self.speed.astype(int)
        # load images
        if local:
            img_folder = '/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset'
        else:
            img_folder = './data/streetview/dataset'
        if not augmentation:
            self.transform = transforms.Compose([
                # transforms.PILToTensor(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.img_path = np.array([img_folder + f'/{idx}.jpg' for idx in indi])
        if biased_sampling:
            flag3 = (self.y == 3).astype(bool)
            flag4 = (self.y == 4).astype(bool)
            y_series = [self.y, self.y[flag3], self.y[flag4], self.y[flag3], self.y[flag4]]
            x_series = [self.img_path, self.img_path[flag3], self.img_path[flag4], self.img_path[flag3], self.img_path[flag4]]
            if return_speed:
                speed_series = [self.speed, self.speed[flag3], self.speed[flag4], self.speed[flag3], self.speed[flag4]]
                self.speed = np.concatenate(speed_series, axis=0)
            self.y = np.concatenate(y_series, axis=0)
            self.img_path = np.concatenate(x_series, axis=0)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = self.transform(img).float()
        if self.return_speed:
            return img, torch.tensor([self.speed[idx]]), torch.tensor(self.y[idx])
        else:
            return img, torch.tensor(self.y[idx])

    def __len__(self):
        return len(self.y)


class MoCoDataset(Dataset):

    def __init__(self, purpose='training', local=True, toy=False):
        super().__init__()
        # load index and labels
        indi = np.loadtxt(f'./data/{purpose}_idx.txt').astype(int)
        if toy:
            np.random.seed(31415926)
            np.random.shuffle(indi)
            indi = indi[:1000]
        # load images
        if local:
            img_folder = '/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset'
        else:
            img_folder = './data/streetview/dataset'
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(sigma=[.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.img_path = np.array([img_folder + f'/{idx}.jpg' for idx in indi])

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        q = self.transform(img)
        k = self.transform(img)
        return [q, k]

    def __len__(self):
        return len(self.img_path)


class LabelMoCoDataset(Dataset):

    def __init__(self, purpose='training', local=True, toy=False):
        super().__init__()
        # load index and labels
        indi = np.loadtxt(f'./data/{purpose}_idx.txt').astype(int)
        if toy:
            np.random.seed(31415926)
            np.random.shuffle(indi)
            indi = indi[:1000]
        # load images
        if local:
            img_folder = '/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset'
        else:
            img_folder = './data/streetview/dataset'
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(sigma=[.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.img_path = np.array([img_folder + f'/{idx}.jpg' for idx in indi])
        # load LTS label
        lts = np.loadtxt('./data/LTS/lts_labels.txt').astype(int)
        self.y = lts[indi]

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        q = self.transform(img)
        k = self.transform(img)
        return q, k, self.y[idx]

    def __len__(self):
        return len(self.img_path)
