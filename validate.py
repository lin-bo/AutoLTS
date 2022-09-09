from utils import StreetviewDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Res50FC
from tqdm import tqdm


def validation(net, vali_loader, device):
    tot_cnt = 0
    corr_cnt = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(vali_loader):
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y-1)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            predicted += 1
            tot_cnt += predicted.shape[0]
            corr_cnt += (predicted == y).sum().item()
    net.train()
    return total_loss, corr_cnt/tot_cnt * 100
    # print(f'Accuracy on validation set: {corr_cnt/tot_cnt * 100} %')


if __name__ == '__main__':
    epoch = 5
    # load checkpoint
    checkpoint = torch.load(f'./checkpoint/Res50FC_{epoch}.pt')
    # set parameters
    device = checkpoint['hyper-parameters']['device']
    batch_size = checkpoint['hyper-parameters']['batch_size']
    # load training data
    vali_loader = DataLoader(StreetviewDataset(purpose='validation'), batch_size=batch_size, shuffle=True)
    # initialization
    net = Res50FC(pretrained=False).to(device=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    validation(net, vali_loader, device)

