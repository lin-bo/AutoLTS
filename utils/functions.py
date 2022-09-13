import random
import os
from PIL import ImageFilter
import torch


def initialization(check_path, n_check, n_epoch, job_id, net, optimizer):
    init_epoch = 0
    loss_records = []
    for epoch in list(range(n_epoch))[::-1]:
        if (epoch + 1) % n_check != 0:
            continue
        if os.path.exists(check_path + f'{job_id}_{epoch}.pt'):
            checkpoint = torch.load(check_path + f'{job_id}_{epoch}.pt')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            init_epoch = checkpoint['epoch'] + 1
            loss_records = checkpoint['loss_records']
            output_records = checkpoint['output_records'] if 'output_records' in checkpoint else []
            if len(output_records) > 0:
                for msg in output_records:
                    print(msg)
            return init_epoch, loss_records, net, optimizer, output_records
    return init_epoch, loss_records, net, optimizer, []


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

