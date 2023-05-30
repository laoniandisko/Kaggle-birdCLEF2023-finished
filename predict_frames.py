import argparse
import os

import zoo

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings

import cv2
import numpy as np

from training.config import load_config
from training.datasets import BirdDatasetOOF

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.simplefilter("ignore")


def load_model(conf: dict, prefix: str, suffix: str, fold: int):
    snapshot_name = "{}{}_{}_{}_{}".format(prefix, conf["network"], conf["encoder_params"]["encoder"], fold, suffix)
    weights_path = os.path.join("weights", snapshot_name)
    model = zoo.__dict__[conf["network"]](**conf["encoder_params"])
    model = torch.nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}''".format(weights_path))
    checkpoint = torch.load(weights_path, map_location="cpu")
    print("epoch", checkpoint["epoch"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sounds Test Predictor")
    arg = parser.add_argument
    arg("--config", metavar="CONFIG_FILE", default="configs/v2s.json", help="path to configuration file")
    arg("--data-path", type=str, default="/home/selim/kaggle/data/",
        help="Path to test images")
    arg("--folds-csv", type=str, default="val_folds.csv")
    arg("--gpu", type=str, default="0", help="List of GPUs for parallel training, e.g. 0,1,2,3")
    arg("--prefix", type=str, default="val_only_")
    arg("--suffix", type=str, default="last", choices=["best_lwrap", "best_logloss", "last"])
    arg("--out", type=str, default="oof")
    arg("--folds", type=int, default=5)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    conf = load_config(args.config)
    for fold in range(args.folds):
        model = load_model(conf, args.prefix, args.suffix, fold).float()

        test_dataset = BirdDatasetOOF(
            dataset_dir=args.data_path,
            fold=fold,
            folds_csv=args.folds_csv,
            duration=5,  # 5 second clips
            step=1  # infer each second
        )
        print("Predicting fold {} ({}) for {}".format(fold, args.folds_csv, args.config))
        data = []
        with torch.no_grad():
            loader = DataLoader(test_dataset, batch_size=1, num_workers=16, shuffle=False)
            for sample in tqdm(loader):
                frames = sample["wavs"][0]
                file_name = sample["file_name"][0]
                data_year = sample["data_year"][0]
                frame_num = frames.size(0)
                all_probs = []
                for frame in range(frame_num):
                    output = model(frames[frame:frame + 1])["logit"][0]
                    probs = torch.sigmoid(output).cpu().cpu().numpy()
                    probs[np.isnan(probs)] = 0.000001
                    all_probs.append(probs)
                all_probs = np.array(all_probs)
                base, _ = os.path.splitext(file_name)
                cls, fid = base.split("/")[-2:]
                out_dir = os.path.join(args.out, str(int(data_year)), cls)
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, fid), all_probs)
