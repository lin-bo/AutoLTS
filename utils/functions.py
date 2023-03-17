import random
import os
from PIL import ImageFilter
import torch
from model import Res50FC, MoCoClf, MoCoClfV2, MoCoClfV3, MoCoClfV2Fea, MoCoClfV3Fea, Res50FCFea


def load_checkpoint(checkpoint_path, net, optimizer):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    loss_records = checkpoint['loss_records']
    output_records = checkpoint['output_records'] if 'output_records' in checkpoint else []
    if len(output_records) > 0:
        for msg in output_records:
            print(msg)
    return init_epoch, loss_records, net, optimizer, output_records


def initialization(check_path, n_check, n_epoch, job_id, net, optimizer, start_point=None):
    init_epoch = 0
    loss_records = []
    for epoch in list(range(n_epoch))[::-1]:
        if (epoch + 1) % n_check != 0:
            continue
        checkpoint_path = check_path + f'{job_id}_{epoch}.pt'
        if os.path.exists(checkpoint_path):
            return load_checkpoint(checkpoint_path, net, optimizer)
    if start_point:
        checkpoint_path = f'./checkpoint/{start_point}.pt'
        return load_checkpoint(checkpoint_path, net, optimizer)
    return init_epoch, loss_records, net, optimizer, []


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def cal_dim(side_fea):
    # s2d = {'speed_limit': 1, 'n_lanes': 1, 'road_type': 9, 'cyc_infras': 1}
    s2d = {'oneway': 1,
           'oneway_onehot': 2,
           'road_type_onehot': 4,
           'cyc_infras_onehot': 4,
           'speed_limit': 1,
           'n_lanes': 1,
           'sce1': 12, 'sce2': 12, 'sce3': 12}
    cnt = 0
    for fea in side_fea:
        cnt += s2d[fea]
    return cnt


def init_mdl(mdl_name, device, side_fea, label):
    l2d = {'lts': 4, 'lts_wo_volume': 4,
           'oneway': 1, 'oneway_onehot': 2,
           'parking': 1, 'parking_onehot': 2,
           'volume': 1, 'volume_onehot': 2,
           'speed_actual': 1, 'speed_actual_onehot': 4,
           'cyc_infras': 2, 'cyc_infras_onehot': 4,
           'n_lanes': 1, 'n_lanes_onehot': 5,
           'road_type': 9, 'road_type_onehot': 4}
    if mdl_name == 'Res50':
        mdl = Res50FC(pretrained=False, out_dim=l2d[label]).to(device=device)
    elif mdl_name == 'Res50Fea':
        n_fea = cal_dim(side_fea)
        mdl = Res50FCFea(pretrained=False, n_fea=n_fea, out_dim=l2d[label]).to(device=device)
    elif mdl_name == 'MoCoClf':
        mdl = MoCoClf(vali=True, out_dim=l2d[label]).to(device=device)
    elif mdl_name == 'MoCoClfV2':
        mdl = MoCoClfV2(vali=True, out_dim=l2d[label]).to(device=device)
    elif mdl_name == 'MoCoClfV3':
        mdl = MoCoClfV3(vali=True, out_dim=l2d[label]).to(device=device)
    elif mdl_name == 'MoCoClfFeaV2':
        n_fea = cal_dim(side_fea)
        mdl = MoCoClfV2Fea(vali=True, n_fea=n_fea, out_dim=l2d[label]).to(device=device)
    elif mdl_name == 'MoCoClfFeaV3':
        n_fea = cal_dim(side_fea)
        mdl = MoCoClfV3Fea(vali=True, n_fea=n_fea, out_dim=l2d[label]).to(device=device)
    else:
        raise ValueError(f'Model {mdl_name} not found')
    return mdl


def fea2dim(fea):
    l2d = {'lts': 4,
           'oneway': 1, 'oneway_onehot': 2,
           'parking': 1, 'parking_onehot': 2,
           'volume': 1, 'volume_onehot': 2,
           'speed_actual': 1, 'speed_actual_onehot': 4,
           'cyc_infras': 2, 'cyc_infras_onehot': 4,
           'n_lanes': 1, 'n_lanes_onehot': 5,
           'road_type': 9, 'road_type_onehot': 4}
    return l2d[fea]


def assess_batch_pred(preds, trues, feas, device):
    mse_fea = {'speed_actual', 'n_lanes'}
    cnts = torch.zeros(len(feas)).to(device)
    for idx, fea in enumerate(feas):
        if fea in mse_fea:
            cnts[idx] = torch.abs(preds[idx] - trues[idx]).sum()
        else:
            _, pred = torch.max(preds[idx], dim=1)
            cnts[idx] += (pred == trues[idx]).sum()
    return cnts


def save_checkpoint(net, optimizer, epoch, loss_records, n_epoch, n_check, device,
                    batch_size, lr, check_path, job_id, output_records):
    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_records': loss_records,
                'output_records': output_records,
                'hyper-parameters': {'n_epoch': n_epoch, 'n_check': n_check, 'device': device, 'batch_size': batch_size, 'lr': lr}
                },
               check_path + f'{job_id}_{epoch}.pt')
