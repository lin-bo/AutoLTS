import time
import os
import cv2

import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer, ColorMode
from mask_former.mask_former import add_mask_former_config
from utils import StreetviewDatasetMaskFormer

# constants
WINDOW_NAME = "MaskFormer demo"


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
    parser.add_argument(
        '--local',
        action='store_true',
        default=False,
        help='is the training on a local device or not'
    )
    parser.add_argument(
        '--no-local',
        dest='local',
        action='store_false'
    )
    parser.add_argument(
        '--toy',
        action='store_true',
        default=False,
        help='use the toy example or not'
    )
    parser.add_argument(
        '--no-toy',
        dest='toy',
        action='store_false'
    )
    parser.add_argument(
        '--visual',
        action='store_true',
        default=False,
        help='visualization mode or not, affect dataset generation'
    )
    parser.add_argument(
        '--no-visual',
        dest='visual',
        action='store_false'
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


def bike_or_not(predictions):
    ans = []
    for p in predictions:
        flag = ((p['sem_seg'].argmax(axis=0) == 7).sum() > 0).to(torch.int).item()
        ans.append(flag)
    return ans


if __name__ == '__main__':
    # initialization
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    mdl = load_mdl(cfg)
    # create data loader
    loader = DataLoader(StreetviewDatasetMaskFormer(cfg=cfg, local=args.local, toy=args.toy),
                        batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)
    records = []
    cnt = 0
    tick = time.time()
    with torch.no_grad():
        for x in tqdm(loader):
            tick = time.time()
            predictions = mdl(x)
            label = predictions[0]['sem_seg'].argmax(axis=0)
            np.savetxt(f'./data/maskformer_pred/pixel_labels/test.txt', label, delimiter=',')
            if args.visual:
                image = x[0]['orig_img'][:, :, [2, 1, 0]]
                metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
                visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
                vis_output = visualizer.draw_sem_seg(
                    predictions[0]["sem_seg"].argmax(dim=0).to(torch.device("cpu"))
                )
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, vis_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            predictions = bike_or_not(predictions)
            records += predictions
            cnt += 1
            if cnt > 0 and cnt % 100 == 0:
                print(cnt, f'{time.time() - tick}')
                np.savetxt(f'./data/maskformer_pred/bike_lane_binary.txt', records, delimiter=',')
            break
    np.savetxt(f'./data/maskformer_pred/bike_lane_binary.txt', records, delimiter=',')
