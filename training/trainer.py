import dataclasses
import logging
import math
import os
import random
import re
from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, List

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter
from timm.utils import AverageMeter
from torch.distributions import Beta
from torch.nn import DataParallel, SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import zoo
from training import losses
from training.config import load_config
from training.losses import LossCalculator
from training.sampler import DistributedWeightedRandomSampler
from training.sync_bn_timm import SyncBatchNormAct
from training.utils import create_optimizer

import zoo_transforms
from training.val_dataset import BirdDataset
def create_data_datasets(args):
    conf = load_config(args.config)
    train_period = conf["encoder_params"].get("duration")
    infer_period = conf["encoder_params"].get("val_duration")

    print(f"""
    creating dataset for fold {args.fold}
    transforms                {conf.get("train_transforms")}
    train_period              {train_period}
    infer_period              {infer_period} 
    """)

    train_transforms = zoo_transforms.__dict__[conf.get("train_transforms")]

    ## set 1 csv
    train_dataset = BirdDataset(mode="train", folds_csv=args.folds_csv, dataset_dir=args.data_dir, fold=args.fold,
                                multiplier=conf.get("multiplier", 1), duration=train_period, transforms=train_transforms,
                                n_classes=conf['encoder_params']['classes'])
    val_dataset = BirdDataset(mode="val", folds_csv=args.folds_csv, dataset_dir=args.data_dir, fold=args.fold, duration=infer_period,
                              n_classes=conf['encoder_params']['classes'])
    return train_dataset, val_dataset
@dataclasses.dataclass
class TrainConfiguration:
    config_path: str
    gpu: str = "0"
    distributed: bool = False
    from_zero: bool = False
    zero_score: bool = False
    local_rank: int = 0
    freeze_epochs: int = 0
    test_every: int = 1
    world_size: int = 1
    output_dir: str = "weights"
    prefix: str = ""
    resume_checkpoint: str = None
    workers: int = 8
    log_dir: str = "logs"
    fp16: bool = True
    freeze_bn: bool = False
    mixup_prob: float = 0.0

class config:
    seed = 630
    train = False

    transforms = {
        "train": [{"name": "Normalize"}],
        "valid": [{"name": "Normalize"}]
    }

    duration = 5
    n_mels = 256
    fmin = 16
    fmax = 16386
    n_fft = 2048
    hop_length = 512
    sr = 32000

    target_columns = 'afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
                          barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul \
                          brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea \
                          cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea \
                          comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig \
                          fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo \
                          hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1 \
                          jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae \
                          madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin \
                          norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh \
                          reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff \
                          saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan \
                          towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov'.split()

    base_model_name = "tf_efficientnet_b0_ns"
    pretrained = False
    num_classes = 264
    in_channels = 1

    ckpt_path = [
        "../input/tf-efficientnet-b0-ns/fold-0_0.8157349896480331.bin",
        "../input/tf-efficientnet-b0-ns/fold-1_0.8130277442702051.bin",
        "../input/tf-efficientnet-b0-ns/fold-2_0.81753840842396.bin",
    ]
class Evaluator(ABC):
    @abstractmethod
    def init_metrics(self) -> Dict:
        pass

    @abstractmethod
    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False,
                 local_rank: int = 0, snapshot_name: str = "") -> Dict:
        pass

    @abstractmethod
    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        pass


class LossFunction:

    def __init__(self, loss: LossCalculator, name: str, weight: float = 1, display: bool = False):
        super().__init__()
        self.loss = loss
        self.name = name
        self.weight = weight
        self.display = display


class PytorchTrainer(ABC):
    def __init__(self, train_config: TrainConfiguration, evaluator: Evaluator,
                 fold: int,
                 train_data: Dataset,
                 val_data: Dataset,
                 args = None) -> None:
        super().__init__()
        self.fold = fold
        self.train_config = train_config
        self.conf = load_config(train_config.config_path)
        self._init_distributed()
        self.evaluator = evaluator
        self.current_metrics = evaluator.init_metrics()
        self.current_epoch = 0
        self.model = self._init_model()
        self.losses = self._init_loss_functions()
        self.optimizer, self.scheduler = create_optimizer(self.conf['optimizer'], self.model, len(train_data),
                                                          train_config.world_size)
        self._init_amp()
        self.train_data = train_data
        self.val_data = val_data

        self.args = args
        if self.train_config.local_rank == 0:
            self.summary_writer = SummaryWriter(os.path.join(train_config.log_dir, self.snapshot_name))

    def validate(self):
        self.model.eval()
        metrics = self.evaluator.validate(self.get_val_loader(), self.model,
                                          distributed=self.train_config.distributed,
                                          local_rank=self.train_config.local_rank,
                                          snapshot_name=self.snapshot_name)
        print(metrics)

    def fit(self):
        for epoch in range(self.current_epoch, self.conf["optimizer"]["schedule"]["epochs"]):
            self.train_data, self.val_data = create_data_datasets(self.args)
            self.current_epoch = epoch
            self.model.train()
            self._freeze()
            self._run_one_epoch_train(self.get_train_loader())
            self.model.eval()
            if self.train_config.local_rank == 0:
                self._save_last(score=9)

            if (self.current_epoch + 1) % self.train_config.test_every == 0:
                metrics = self.evaluator.validate(self.get_val_loader(), self.model,
                                                  distributed=self.train_config.distributed,
                                                  local_rank=self.train_config.local_rank,
                                                  snapshot_name=self.snapshot_name)

                self._save_last(score=metrics["score"])

                if self.train_config.local_rank == 0:
                    improved_metrics = self.evaluator.get_improved_metrics(self.current_metrics, metrics)
                    self.current_metrics.update(improved_metrics)
                    self._save_best(improved_metrics)
                    for k, v in metrics.items():
                        self.summary_writer.add_scalar('val/{}'.format(k), float(v), global_step=self.current_epoch)

    def _save_last(self,score):
        self.model = self.model.eval()
        torch.save({
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'metrics': self.current_metrics,

        }, os.path.join(self.train_config.output_dir, str(self.current_epoch)+self.snapshot_name + "_last"+str(score)))

    def _save_best(self, improved_metrics: Dict):
        self.model = self.model.eval()
        for metric_name in improved_metrics.keys():
            torch.save({
                'epoch': self.current_epoch,
                'state_dict': self.model.state_dict(),
                'metrics': self.current_metrics,

            }, os.path.join(self.train_config.output_dir, self.snapshot_name + "_" + metric_name))

    def _run_one_epoch_train(self, loader: DataLoader):
        iterator = tqdm(loader)
        loss_meter = AverageMeter()
        avg_meters = {"loss": loss_meter}
        for loss_def in self.losses:
            if loss_def.display:
                avg_meters[loss_def.name] = AverageMeter()

        if self.conf["optimizer"]["schedule"]["mode"] == "epoch":
            self.scheduler.step(self.current_epoch)

        for i, sample in enumerate(iterator):
            # todo: make configurable
            imgs = sample["wav"].cuda().float()
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.train_config.fp16):
                targets = sample["labels"]
                audio = sample["wav"]
                if random.random() < self.train_config.mixup_prob:
                    bs = targets.size(0)
                    coeffs = Beta(1., 1.).rsample(torch.Size((bs,))).to(audio.device)
                    perm_idx = torch.randperm(bs)
                    shuffled = audio[perm_idx]
                    c = coeffs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    audio = (audio * c + (1 - c) * shuffled)
                    c1 = coeffs.unsqueeze(-1).repeat(1,self.conf['encoder_params']['classes'])
                    targets = targets * c1 + targets[perm_idx] * (1 - c1)
                    targets = targets.clamp(0, 1)
                    sample["labels"] = targets
                    sample["wav"] = audio

                output = self.model(imgs)
                total_loss = 0
                
                for loss_def in self.losses:
                    l = loss_def.loss.calculate_loss(output, sample)
                    if loss_def.display:
                        avg_meters[loss_def.name].update(l if isinstance(l, Number) else l.item(), imgs.size(0))
                    total_loss += loss_def.weight * l

            loss_meter.update(total_loss.item(), imgs.size(0))
            if math.isnan(total_loss.item()) or math.isinf(total_loss.item()):
                raise ValueError("NaN loss !!")
            avg_metrics = {k: f"{v.avg:.4f}" for k, v in avg_meters.items()}
            iterator.set_postfix({"lr": float(self.scheduler.get_lr()[-1]),
                                  "epoch": self.current_epoch,
                                  **avg_metrics
                                  })
            ## TODO: clip value in config
            if self.train_config.fp16:
                self.gscaler.scale(total_loss).backward()
                self.gscaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
            torch.cuda.synchronize()
            if self.train_config.distributed:
                dist.barrier()
            if self.conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
                self.scheduler.step(i + self.current_epoch * len(loader))

        if self.train_config.local_rank == 0:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                self.summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=self.current_epoch)
            self.summary_writer.add_scalar('train/loss', float(loss_meter.avg), global_step=self.current_epoch)

    @property
    def train_batch_size(self):
        return self.conf["optimizer"]["train_bs"]

    @property
    def val_batch_size(self):
        return self.conf["optimizer"]["val_bs"]

    def get_train_loader(self) -> DataLoader:
        train_sampler = None
        if self.train_config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
            if hasattr(self.train_data, "get_weights"):
                print('Train dataloader will have sampler in DDP')
                train_sampler = DistributedWeightedRandomSampler(self.train_data, self.train_data.get_weights())
            train_sampler.set_epoch(self.current_epoch)
        elif hasattr(self.train_data, "get_weights"):
            print("Using WeightedRandomSampler")
            train_sampler = WeightedRandomSampler(self.train_data.get_weights(), len(self.train_data))
        train_data_loader = DataLoader(self.train_data, batch_size=self.train_batch_size,
                                       num_workers=self.train_config.workers,
                                       shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                       drop_last=True)

        return train_data_loader

    def get_val_loader(self) -> DataLoader:
        val_sampler = None
        if self.train_config.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_data, shuffle=False)
        val_data_loader = DataLoader(self.val_data, sampler=val_sampler, batch_size=self.val_batch_size,
                                     num_workers=self.train_config.workers,
                                     shuffle=False,
                                     pin_memory=False)
        return val_data_loader

    @property
    def snapshot_name(self):
        return "{}{}_{}_{}".format(self.train_config.prefix, self.conf["network"],
                                   self.conf["encoder_params"]["encoder"], self.fold)

    def _freeze(self):
        if hasattr(self.model.module, "encoder"):
            encoder = self.model.module.encoder
        elif hasattr(self.model.module, "encoder_stages"):
            encoder = self.model.module.encoder_stages
        else:
            logging.warn("unknown encoder model")
            return
        if self.current_epoch < self.train_config.freeze_epochs:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
        else:
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad = True
        if self.train_config.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
    def _init_amp(self):
        self.gscaler = torch.cuda.amp.GradScaler()

        if self.train_config.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.train_config.local_rank],
                                                 output_device=self.train_config.local_rank,
                                                 find_unused_parameters=True)
        else:
            self.model = DataParallel(self.model).cuda()

    def _init_distributed(self):
        if self.train_config.distributed:
            self.pg = dist.init_process_group(backend="nccl",
                                              rank=self.train_config.local_rank,
                                              world_size=self.train_config.world_size)

            torch.cuda.set_device(self.train_config.local_rank)
        else:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = self.train_config.gpu

    def _load_checkpoint(self, model: torch.nn.Module):
        checkpoint_path = self.train_config.resume_checkpoint
        if not checkpoint_path:
            return
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            print(checkpoint.keys())
            state_dict_name = "state_dict"
            if state_dict_name not in checkpoint:
                state_dict_name = "model"
            if state_dict_name in checkpoint:
                state_dict = checkpoint[state_dict_name]
                state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
                orig_state_dict = model.state_dict()
                mismatched_keys = []
                for k, v in state_dict.items():
                    ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
                    if v.size() != ori_size:
                        print("SKIPPING!!! Shape of {} changed from {} to {}".format(k, v.size(), ori_size))
                        mismatched_keys.append(k)
                for k in mismatched_keys:
                    del state_dict[k]
                model.load_state_dict(state_dict, strict=False)
                if not self.train_config.from_zero:
                    self.current_epoch = checkpoint['epoch']
                    if not self.train_config.zero_score:
                        self.current_metrics = checkpoint.get('metrics', self.evaluator.init_metrics())
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_path, checkpoint['epoch']))
            else:


                del checkpoint['att_block.att.weight']
                del checkpoint['att_block.att.bias']
                del checkpoint['att_block.cla.weight']
                del checkpoint['att_block.cla.bias']

                model.load_state_dict(checkpoint, strict=False)

                # encodername = ["encoder.0.weight", "encoder.1.weight", "encoder.1.bias", "encoder.4.weight",
                #                "encoder.5.weight", "encoder.5.bias"]
                # for name, param in model.named_parameters():
                #     if "conv" in name or name in encodername:
                #         param.requires_grad = False
                print("=> loaded checkpoint '{}' "
                      .format(checkpoint_path))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
        if self.train_config.from_zero:
            self.current_metrics = self.evaluator.init_metrics()
            self.current_epoch = 0

    def _init_model(self):
        print(self.train_config)

        model = zoo.__dict__[self.conf['network']](base_model_name=config.base_model_name,
            config=config,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            in_channels=config.in_channels)
        # model = zoo.__dict__[self.conf['network']](**self.conf["encoder_params"])
        model = model.cuda()
        self._load_checkpoint(model)

        if self.train_config.distributed and not self.train_config.freeze_bn:
            model = SyncBatchNormAct.convert_sync_batchnorm(model, self.pg)
        channels_last = self.conf.get("channels_last", False)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    def _init_loss_functions(self) -> List[LossFunction]:
        assert self.conf['losses']
        loss_functions = []
        for loss_def in self.conf['losses']:
            if 'params' in loss_def:
                loss_fn = losses.__dict__[loss_def["type"]](**loss_def["params"])
            else:
                loss_fn = losses.__dict__[loss_def["type"]]()
            loss_weight = loss_def["weight"]
            display = loss_def["display"]
            loss_functions.append(LossFunction(loss_fn, loss_def["name"], loss_weight, display))

        return loss_functions
