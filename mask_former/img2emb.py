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
from mask_former import add_mask_former_config


class StreetviewDataset(Dataset):

    def __init__(self, purpose='training', toy=False, local=True):
        super().__init__()
        # load images and indices
        if local:
            img_folder = '/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/Streetview2LTS/dataset'
            indi = np.loadtxt(f'/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/AutoLTS/data/{purpose}_idx.txt').astype(int)
        else:
            img_folder = './data/streetview/dataset'
            indi = np.loadtxt(f'./data/{purpose}_idx.txt').astype(int)
        if toy:
            np.random.seed(31415926)
            np.random.shuffle(indi)
            indi = indi[:1000]
        self.img_path = np.array([img_folder + f'/{idx}.jpg' for idx in indi])
        # transforms
        self.transform = transforms.Compose([
                # transforms.PILToTensor(),
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = self.transform(img)
        return {"image": img, "height": img.shape[1], "width": img.shape[2]}

    def __len__(self):
        return len(self.img_path)


def load_mdl(cfg):
    cfg_cp = cfg.clone()  # cfg can be modified by model
    # load model
    mdl = build_model(cfg_cp)
    mdl.eval()
    # load checkpoint
    checkpointer = DetectionCheckpointer(mdl)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return mdl


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
    return parser


def collate_fn(batch):
    return batch


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    mdl = load_mdl(cfg)
    loader = DataLoader(StreetviewDataset('training', True, True), batch_size=2, shuffle=False, collate_fn=collate_fn)
    for x in tqdm(loader):
        print(len(x))
        predictions = mdl(x)
        print(len(predictions))
        print(predictions[0]['sem_seg'].shape)
        print(predictions[1]['sem_seg'].shape)
        break

    # img = None
    # predictions = mdl(img)["sem_seg"]
