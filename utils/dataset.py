import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
from utils import GaussianBlur


class StreetviewDataset(Dataset):

    def __init__(self, purpose='training', toy=False, local=True, augmentation=False, biased_sampling=False, side_fea=[],
                 label='lts', transform=False):
        super().__init__()
        # load indices
        indi = np.loadtxt(f'./data/{purpose}_idx.txt').astype(int)
        if toy:
            np.random.seed(31415926)
            np.random.shuffle(indi)
            indi = indi[:1000]
        # load labels
        if label == 'lts':
            self.y = np.loadtxt('./data/LTS/lts_labels.txt').astype(int)
        else:
            true_label = label[:-7] if label[-7:] == '_onehot' else label
            self.y = np.loadtxt(f'./data/road/{true_label}.txt', delimiter=',').astype(int)
        # transform labels
        if label == 'speed_actual' or label == 'n_lanes':
            self.y = self.y.reshape((-1, 1))
            self.y = self.y.astype(np.single)
            if transform:
                self.y = np.log(self.y + 0.1)
        elif label == 'road_type':
            self.y = np.argmax(self.y, axis=1)
        self.y = self.y[indi]
        # posted speed
        self.side_fea = side_fea
        if side_fea:
            fea_list = []
            for fea in side_fea:
                vec = np.loadtxt(f'./data/road/{fea}.txt', delimiter=',').astype(np.single)
                vec = vec.reshape((len(vec), -1))
                vec = vec[indi]
                fea_list.append(vec)
            self.fea = np.concatenate(fea_list, axis=1)
            mu = self.fea.mean(axis=0, keepdims=True)
            std = self.fea.std(axis=0, keepdims=True)
            self.fea -= mu
            self.fea /= std
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
        if biased_sampling and label == 'lts':
            flag3 = (self.y == 3).astype(bool)
            flag4 = (self.y == 4).astype(bool)
            y_series = [self.y, self.y[flag3], self.y[flag4], self.y[flag3], self.y[flag4]]
            x_series = [self.img_path, self.img_path[flag3], self.img_path[flag4], self.img_path[flag3], self.img_path[flag4]]
            if side_fea:
                fea_series = [self.fea, self.fea[flag3], self.fea[flag4], self.fea[flag3], self.fea[flag4]]
                self.speed = np.concatenate(fea_series, axis=0)
            self.y = np.concatenate(y_series, axis=0)
            self.img_path = np.concatenate(x_series, axis=0)
        elif biased_sampling and label == 'cyc_infras':
            flag = (self.y == 1).astype(bool)
            y_series = [self.y] + [self.y[flag] for _ in range(10)]
            x_series = [self.img_path] + [self.img_path[flag] for _ in range(10)]
            if side_fea:
                fea_series = [self.fea] + [self.fea[flag] for _ in range(10)]
                self.speed = np.concatenate(fea_series, axis=0)
            self.y = np.concatenate(y_series, axis=0)
            self.img_path = np.concatenate(x_series, axis=0)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = self.transform(img).float()
        if self.side_fea:
            return img, torch.tensor(self.fea[idx], dtype=torch.float32), torch.tensor(self.y[idx])
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
