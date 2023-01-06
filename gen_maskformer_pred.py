import time

import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from mask_former.mask_former import add_mask_former_config
from utils import StreetviewDatasetMaskFormer


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "-bs", "--batch_size",
        help="batch size",
        type=int,
        default=32,
    )
    return parser


def load_mdl(cfg):
    cfg_cp = cfg.clone()  # cfg can be modified by model
    # load model
    mdl = build_model(cfg_cp)
    mdl.eval()
    # load checkpoint
    checkpointer = DetectionCheckpointer(mdl)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return mdl


if __name__ == '__main__':
    # initialization
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    mdl = load_mdl(cfg)
    # create data loader
    loader = DataLoader(StreetviewDatasetMaskFormer(purpose='training', local=False, toy=True),
                        batch_size=32, shuffle=False, collate_fn=lambda x: x)
    for x in tqdm(loader):
        tick = time.time()
        predictions = mdl(x)
        predictions = [p['sem_seg'] for p in predictions]
        print(time.time() - tick)
        break
