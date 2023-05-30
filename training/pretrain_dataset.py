import math
import os
import ast
import librosa

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import albumentations as A
import audiomentations as AA
import zoo_transforms

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
    ]
)
def mono_to_color3(X, eps=1e-6, mean=None, std=None):

    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V
mean2 = (0.485, 0.456, 0.406) # RGB
std2 = (0.229, 0.224, 0.225) # RGB
albu_transforms2 = {
    'train' : A.Compose([
            A.Normalize(mean2, std2),
    ]),
    'valid' : A.Compose([
            A.Normalize(mean2, std2),
    ]),
}
class params:
    batch_size = 4
    num_workers = 4

    n_mels = 256
    fmin = 16
    fmax = 16386
    n_fft = 2048
    hop_length = 512
    sr = 32000
    root_dir = "E:\Kaggle\Faster_eff_SED"

    base_model_name = "tf_efficientnet_b0_ns"
    pretrained = False
    num_classes = 509
    in_channels = 1
class PretrainDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            duration,
            transforms=zoo_transforms.set_2,
    ):
        ignore_labels = tuple(list(pd.read_csv(os.path.join(dataset_dir, "birdclef-2023/train_metadata.csv"))["primary_label"].value_counts().keys()))
        df2021 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2021/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2022 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2022/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2020 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2020/train_metadata.csv"))[
                ["primary_label", "secondary_labels", "filename"]]
        df2021["data_year"] = 2021
        df2022["data_year"] = 2022
        df2020["data_year"] = 2020
        df = pd.concat([df2021, df2022], ignore_index=True)
        print(len(df))
        # df = df[~df.primary_label.isin(ignore_labels)]
        print(len(df))
        labels = list(set(df.primary_label.unique()))
        labels.sort()
        self.labels = labels
        self.df = df
        self.dataset_dir = dataset_dir
        self.transforms = transforms

        self.duration = duration
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.bird2id = {x: idx for idx, x in enumerate(labels)}

    def load_one(self, filename, offset, duration):
        try:
            wav, _ = librosa.load(filename, sr=self.sr, offset=offset, duration=duration,mono=True)
            if wav.shape[0] < self.dsr:
                wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        except:
            print("failed reading", filename)
        return wav


    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_short_audio" if data_year == 2021 else "train_audio", row["primary_label"], row['filename'].split("/")[-1])

        ## wav

        wav_len_sec = librosa.get_duration(filename=filename, sr=None)
        duration = self.duration
        max_offset = wav_len_sec - duration
        max_offset = max(max_offset, 1)
        offset = np.random.randint(max_offset)


        wav = self.load_one(filename, offset=offset, duration=self.duration)

        # 加入\加速语音\音调增强\时间移动
        '''zengqiang
        '''
        # 加入\加速语音\音调增强\时间移动
        if self.mode == "train":

            if row["is_zengqiang"] == 1:
                wav = self.zengqiang.shijianyidong(wav)
        if self.transforms:
            wav = self.transforms(wav, self.sr)
        image = librosa.feature.melspectrogram(y=wav, sr=params.sr, n_mels=params.n_mels, n_fft=params.n_fft,
                                               hop_length=params.hop_length, fmin=params.fmin, fmax=params.fmax, )
        image = librosa.power_to_db(image.astype(np.float32), ref=np.max)
        ## labels
        labels = torch.zeros((len(self.labels),))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 1.0
            except:
                continue

        return {
            "wav": torch.tensor(wav),
            "labels": labels,
        }
    class zengqiang:
        # 加速语音
        def jiasuyuyin(filename, sr=32000, min_speed=0.8, max_speed=1.2):
            """
            librosa时间拉伸
            :param samples: 音频数据，一维
            :param max_speed: 不要低于0.9，太低效果不好
            :param min_speed: 不要高于1.1，太高效果不好
            :return:
            """

            rng = np.random.default_rng()
            speed = rng.uniform(min_speed, max_speed)
            samples, _ = librosa.load(filename, sr=sr, duration=5 * speed)
            samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
            data_type = samples[0].dtype
            samples = samples.astype(np.float)
            samples = librosa.effects.time_stretch(samples, speed)
            samples = samples.astype(data_type)
            return samples

        # 音调增强
        def yindiaozenqiang(filename, sr=16000, ratio=5):
            samples, _ = librosa.load(filename, sr=32000, duration=5)
            samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
            data_type = samples[0].dtype
            samples = samples.astype('float')
            ratio = random.uniform(-ratio, ratio)
            samples = librosa.effects.pitch_shift(samples, sr, n_steps=ratio)
            samples = samples.astype(data_type)
            return samples

        # 时间移动
        def shijianyidong(filename, sr=16000, max_ratio=0.3):
            """
            改进:
            1.为在一定比例范围内随机偏移，不再需要时间
            2.循环移动
            :param samples: 音频数据
            :param max_ratio:
            :return:
            """
            samples, _ = librosa.load(filename, sr=sr, duration=5)
            samples = samples.copy()
            frame_num = samples.shape[0]
            max_shifts = frame_num * max_ratio  # around 5% shift
            shifts_num = np.random.randint(-max_shifts, max_shifts)
            print(shifts_num)
            if shifts_num > 0:
                # time advance
                temp = samples[:shifts_num]
                samples[:-shifts_num] = samples[shifts_num:]
                # samples[-shifts_num:] = 0
                samples[-shifts_num:] = temp
            elif shifts_num < 0:
                # time delay
                temp = samples[shifts_num:]
                samples[-shifts_num:] = samples[:shifts_num]
                # samples[:-shifts_num] = 0
                samples[:-shifts_num] = temp
            return samples

        # 时域掩盖
        def shiyuyangai(inputs, sr=32000, max_mask_time=30, mask_num=10):
            """
            时间遮掩，
            :param inputs: 三维numpy或tensor，(batch, time_step,  feature_dim)
            :param max_mask_time:
            :param mask_num:
            :return:
            """
            time_len = len(inputs)
            for i in range(mask_num):
                t = np.random.uniform(low=0.0, high=max_mask_time)
                t = int(t)
                t0 = random.randint(0, time_len - t)
                inputs[:, t0:t0 + t] = 0

            return inputs

        # 频域掩盖
        def pinyuyangai(inputs, max_mask_frequency=30, mask_num=10):
            """

            :param inputs: 三维numpy或tensor，(batch, time_step,  feature_dim)
            :param max_mask_frequency:
            :param mask_num:
            :return:
            """
            feature_len = inputs.shape[1]
            for i in range(mask_num):
                f = np.random.uniform(low=0.0, high=max_mask_frequency)
                f = int(f)
                f0 = random.randint(0, feature_len - f)
                inputs[f0:f0 + f, :] = 0
            return inputs

    def __len__(self):
        return len(self.df)
