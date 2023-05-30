import os
import warnings

from sklearn.metrics import classification_report

from class_config import CLASSES_21
from training.config import load_config

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import torch

from training.datasets import BirdDatasetShortClassifier

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from torch.cuda import empty_cache

torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120
import torch.distributed as dist
from tqdm import tqdm

from metrics import bird_metric

warnings.filterwarnings("ignore")
import argparse
import os
from typing import Dict

import audiomentations as AA
import numpy as np
import torch.distributed
from torch.utils.data import DataLoader

from training.trainer import Evaluator, PytorchTrainer, TrainConfiguration

train_aug = AA.Compose(
    [
        AA.AddBackgroundNoise(
            sounds_path="input/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/aicrowd2020_noise_30sec/noise_30sec", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
        # AA.LowPassFilter(min_cutoff_freq=150, max_cutoff_freq=7500, min_rolloff=12, max_rolloff=24, zero_phase=False, p=0.25)
    ]
)


class BirdEvaluator(Evaluator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def init_metrics(self) -> Dict:
        return {"f1_score": 0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:
        conf_name = os.path.splitext(os.path.basename(self.args.config))[0]
        val_dir = os.path.join(self.args.val_dir, conf_name, str(self.args.fold))
        os.makedirs(val_dir, exist_ok=True)

        ## TODO: thresholding?
        val_out = {"gts": [], "preds": []}

        for sample in tqdm(dataloader):
            wav = sample["wav"]
            labels = sample["labels"].numpy()

            outs = model(wav)
            outs = outs['logit'].sigmoid().cpu().detach().numpy()

            val_out['gts'].extend(labels)
            val_out['preds'].extend(outs)

        val_out_path = os.path.join(val_dir, f"{conf_name}_val_outs.npy")
        np.save(val_out_path, val_out)

        if distributed:
            dist.barrier()
        f1s = 0
        if self.args.local_rank == 0:
            outs = np.load(val_out_path, allow_pickle=True)
            gts = np.array(outs[()]['gts'])
            preds = np.array(outs[()]['preds'])

            f1s = bird_metric.get_f1(gts, preds, threshold=0.5)
            print(classification_report(gts, preds > 0.5, target_names=CLASSES_21))

        if distributed:
            dist.barrier()
        empty_cache()
        return {"f1_score": f1s}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        if current_metrics["f1_score"] > prev_metrics["f1_score"]:
            print(
                "f1_score improved from {:.6f} to {:.6f}".format(prev_metrics["f1_score"], current_metrics["f1_score"]))
            improved["f1_score"] = current_metrics["f1_score"]
        else:
            print("f1_score {:.6f} current {:.6f}".format(prev_metrics["f1_score"], current_metrics["f1_score"]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/cls_b3.json")
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='val_')
    arg('--data-dir', type=str, default="/kaggle/input/")
    arg('--val-dir', type=str, default="/mnt/viper/xview3/oof")
    arg('--folds-csv', type=str, default='folds4val.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--fp16', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg('--multiplier', type=int, default=1)
    arg("--val", action='store_true', default=False)
    arg("--freeze-bn", action='store_true', default=False)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    print(f"creating dataset for fold {args.fold}")
    conf = load_config(args.config)
    train_dataset = BirdDatasetShortClassifier(mode="train", folds_csv="train_folds_v3.csv", dataset_dir=args.data_dir, fold=args.fold,
                                multiplier=conf.get("multiplier", 1), n_samples=conf.get("train_samples", 5), transforms=train_aug)
    val_dataset = BirdDatasetShortClassifier(mode="val", folds_csv="val_folds_v3.csv", dataset_dir=args.data_dir, fold=args.fold, n_samples=conf.get("val_samples", 5))
    return train_dataset, val_dataset


def main():
    args = parse_args()
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        fp16=args.fp16,
        freeze_bn=args.freeze_bn
    )

    data_train, data_val = create_data_datasets(args)
    birds_evaluator = BirdEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=birds_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)

    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == '__main__':
    main()
