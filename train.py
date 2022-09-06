from utils import StreetviewDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Res50FC
from tqdm import tqdm


def train(net, optimizer, epoch, train_loader, device):
    net.train()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.
    tot_cnt = 0
    curr_cnt = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        outputs = net.forward(x)
        loss = criterion(outputs, y-1)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, y_pred = torch.max(outputs, dim=1)
        y_pred += 1
        tot_cnt += len(y_pred)
        curr_cnt += (y_pred == y).sum().item()
    print(f'Epoch: {epoch}, loss: {total_loss:.4f}, accuracy: {curr_cnt/tot_cnt * 100:.2f}%')
    return total_loss


if __name__ == '__main__':
    # set parameters
    n_epoch = 100
    n_check = 3
    device = 'mps'
    batch_size = 32
    lr = 0.0003
    # load training data
    train_loader = DataLoader(StreetviewDataset(purpose='validation', toy=True), batch_size=batch_size, shuffle=True)
    # initialization
    net = Res50FC(pretrained=True).to(device=device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    print('start training ...')
    for epoch in range(n_epoch):
        loss = train(net, optimizer, epoch, train_loader, device)
        # if (epoch + 1) % n_check == 0:
        #     torch.save({'epoch': epoch,
        #                 'model_state_dict': net.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'hyper-parameters': {'n_epoch': n_epoch, 'n_check': n_check, 'device': device, 'batch_size': batch_size, 'lr': lr}
        #                 },
        #                f'./checkpoint/Res50FC_{epoch}.pt')
