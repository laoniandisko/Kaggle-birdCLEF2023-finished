import ast
import math
import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BirdDatasetOOF(Dataset):
    def __init__(
            self,
            folds_csv: str,
            dataset_dir: str,
            fold: int = 0,
            transforms=None,
            duration: int = 5,
            step: int = 1,
    ):
        self.df = pd.read_csv(folds_csv)
        self.df = self.df[self.df['fold'] == fold]

        self.dataset_dir = dataset_dir

        self.step = step
        self.duration = duration
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.transforms = transforms

    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year == 2022 else "train_short_audio", row['filename'])

        duration_seconds = librosa.get_duration(filename=filename, sr=None)
        if duration_seconds < self.duration + self.step:
            start_offsets = [0]
        else:
            last_offset = int(duration_seconds - self.duration)
            start_offsets = list(range(0, last_offset + self.step, self.step))
        wavs = []
        for offset in start_offsets:
            wav, _ = librosa.load(filename, sr=None, offset=offset, duration=self.duration)
            if wav.shape[0] < self.dsr:
                wav = np.pad(wav, (0, self.dsr - wav.shape[0]))
            if self.transforms:
                wav = self.transforms(wav, self.sr)
            wavs.append(np.expand_dims(wav, 0))
        wavs = np.array(wavs)

        return {
            "wavs": torch.tensor(wavs),
            "data_year": data_year,
            "file_name": filename
        }

    def __len__(self):
        return len(self.df)


class BirdDatasetShortClassifier(Dataset):
    def __init__(
            self,
            mode: str,
            folds_csv: str,
            dataset_dir: str,
            fold: int = 0,
            oof_dir="oof",
            transforms=None,
            duration: int = 5,
            step: int = 1,
            n_classes: int = 21,
            multiplier: int = 1,
            n_samples: int = 5,

    ):
        self.df = pd.read_csv(folds_csv)
        self.df = self.df[self.df['fold'] == fold]
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.n_classes = n_classes
        self.step = step
        self.duration = duration
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.transforms = transforms
        self.oof_dir = oof_dir
        self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)
        vc = self.df.primary_label.value_counts()
        dataset_length = len(self.df)
        label_weight = {}
        for row in vc.items():
            label, count = row
            label_weight[label] = math.pow(dataset_length / count, 1 / 2)

        self.df["label_weight"] = self.df.primary_label.apply(lambda x: label_weight[x])

        birds = np.array(['akiapo', 'aniani', 'apapan', 'barpet', 'crehon', 'elepai', 'ercfra',
                          'hawama', 'hawcre', 'hawgoo', 'hawhaw', 'hawpet1', 'houfin', 'iiwi',
                          'jabwar', 'maupar', 'omao', 'puaioh', 'skylar', 'warwhe1', 'yefcan'])
        self.bird2id = {x: idx for idx, x in enumerate(birds)}
        self.n_samples = n_samples
        if self.mode == "train":
            if multiplier > 1:
                self.df = pd.concat([self.df] * multiplier, ignore_index=True)

    def get_weights(self):
        return self.df.label_weight.values

    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year == 2022 else "train_short_audio", row['filename'])
        is_validation = self.mode == "val"


        labels = torch.zeros((self.n_classes,))
        primary_cls = self.bird2id[row['primary_label']]
        labels[primary_cls] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                sec_cls = self.bird2id[x]
                labels[sec_cls] = 1.0
            except:
                continue
        file_dr = librosa.get_duration(filename=filename, sr=None)
        wavs = []
        if file_dr < self.duration:
            wav, _ = librosa.load(filename, sr=None, offset=0, duration=self.duration)
            wav = np.pad(wav, (0, self.dsr - wav.shape[0]))
            wavs = [np.expand_dims(wav, 0)] * self.n_samples
        else:
            offsets = np.linspace(0, file_dr - self.duration, self.n_samples)
            if not is_validation:
                offsets = []
                for i in range(self.n_samples):
                    offsets.append(random.random() * (file_dr - self.duration))
            for offset in offsets:
                wav, _ = librosa.load(filename, sr=None, offset=offset, duration=self.duration)
                if self.transforms:
                    wav = self.transforms(wav, self.sr)
                wavs.append(np.expand_dims(wav, 0))
        wavs = np.array(wavs)
        return {
            "wav": torch.from_numpy(wavs).float(),
            "labels": labels,
        }


    def __len__(self):
        return len(self.df)
