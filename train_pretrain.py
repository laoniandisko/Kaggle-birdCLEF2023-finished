import os
import warnings

from training.pretrain_dataset import PretrainDataset
from training.config import load_config
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import torch

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120

warnings.filterwarnings("ignore")
import argparse
from typing import Dict

import torch.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.trainer import Evaluator, PytorchTrainer, TrainConfiguration
import numpy as np
import sklearn.metrics
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
        sum0 = 0
        index = 0

        for sample in tqdm(dataloader):

            wav = sample["wav"]
            labels = sample["labels"].numpy()

            outs_tem = model(wav)
            outs_tem = outs_tem['logit'].sigmoid().cpu().detach().numpy()
            if index == 0:
                outs = outs_tem
                outs1 = outs_tem
            if index == 1:
                outs = outs_tem * 0.8 + outs1 * 0.2
                outs2 = outs1
                outs1 = outs_tem
            if index >= 2 and index != len(dataloader) - 1:
                outs = outs_tem * 0.7 + outs1 * 0.15 + outs2 * 0.15
                outs2 = outs1
                outs1 = outs_tem
            if index == len(dataloader) - 1:
                outs = outs_tem

            solution = np.array([[outs[j][i] for j in range(len(outs))] for i in range(len(outs[0]))])
            new_rows = np.array([[labels[j][i] for j in range(len(labels))] for i in range(len(labels[0]))])

            a = sklearn.metrics.average_precision_score(new_rows, solution, average='macro', )
            sum0 += a
            index += 1



        print("*******===========================**************")
        print(sum0 / index)





        return { "score": sum0 / index}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        return {}


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/v2s.json")
    arg('--workers', type=int, default=12, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='weights/')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='pretrain_')
    arg('--data-dir', type=str, default="E:/Kaggle/birdclef-2022-main/kaggle/input/")
    arg('--val-dir', type=str, default="/mnt/viper/xview3/oof")
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
    arg("--freeze-bn", action='store_true', default=False)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    print(f"creating dataset for fold {args.fold}")
    train_dataset = PretrainDataset(dataset_dir=args.data_dir,duration = load_config(args.config)["encoder_params"].get("duration") )
    val_dataset = PretrainDataset(dataset_dir=args.data_dir,duration = load_config(args.config)["encoder_params"].get("duration"))
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

    trainer.fit()


if __name__ == '__main__':
    main()
