from torch.utils.data import DataLoader
from utils import StreetviewDataset
import torch


if __name__ == '__main__':
    # create
    dataset_train = StreetviewDataset(purpose='training', toy=False, local=True)
    dataset_vali = StreetviewDataset(purpose='validation', toy=False, local=True)
    dataset_test = StreetviewDataset(purpose='test', toy=False, local=True)
    # save
    torch.save(dataset_train, './data/training.pt')
    torch.save(dataset_vali, './data/validation.pt')
    torch.save(dataset_test, './data/test.pt')

