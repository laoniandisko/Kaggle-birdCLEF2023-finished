import math
import os
import ast
import random
import traceback
from builtins import Exception

import librosa

import numpy as np
import pandas as pd
import audiomentations as AA

import torch
from torch.utils.data import Dataset
import albumentations as A
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
def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [1 x H x W] -- RGB numpy array
    """
    # X = np.stack([X, X, X], axis=-1)
    X = np.expand_dims(X, axis=-1)

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

config = {
    'in_channels':1
}
mean = (0.485) # R only for RGB
std = (0.229) # R only for RGB
albu_transforms = {
    'train' : A.Compose([
            A.Normalize(mean, std),
    ]),
    'valid' : A.Compose([
            A.Normalize(mean, std),
    ]),
}
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
    target_columns = "abethr1 abhori1 abythr1 afbfly1 afdfly1 afecuc1 affeag1 afgfly1 afghor1 afmdov1 afpfly1 afpkin1 afpwag1 afrgos1 afrgrp1 afrjac1 afrthr1 amesun2 augbuz1 bagwea1 barswa bawhor2 bawman1 bcbeat1 beasun2 bkctch1 bkfruw1 blacra1 blacuc1 blakit1 blaplo1 blbpuf2 blcapa2 blfbus1 blhgon1 blhher1 blksaw1 blnmou1 blnwea1 bltapa1 bltbar1 bltori1 blwlap1 brcale1 brcsta1 brctch1 brcwea1 brican1 brobab1 broman1 brosun1 brrwhe3 brtcha1 brubru1 brwwar1 bswdov1 btweye2 bubwar2 butapa1 cabgre1 carcha1 carwoo1 categr ccbeat1 chespa1 chewea1 chibat1 chtapa3 chucis1 cibwar1 cohmar1 colsun2 combul2 combuz1 comsan crefra2 crheag1 crohor1 darbar1 darter3 didcuc1 dotbar1 dutdov1 easmog1 eaywag1 edcsun3 egygoo equaka1 eswdov1 eubeat1 fatrav1 fatwid1 fislov1 fotdro5 gabgos2 gargan gbesta1 gnbcam2 gnhsun1 gobbun1 gobsta5 gobwea1 golher1 grbcam1 grccra1 grecor greegr grewoo2 grwpyt1 gryapa1 grywrw1 gybfis1 gycwar3 gyhbus1 gyhkin1 gyhneg1 gyhspa1 gytbar1 hadibi1 hamerk1 hartur1 helgui hipbab1 hoopoe huncis1 hunsun2 joygre1 kerspa2 klacuc1 kvbsun1 laudov1 lawgol lesmaw1 lessts1 libeat1 litegr litswi1 litwea1 loceag1 lotcor1 lotlap1 luebus1 mabeat1 macshr1 malkin1 marsto1 marsun2 mcptit1 meypar1 moccha1 mouwag1 ndcsun2 nobfly1 norbro1 norcro1 norfis1 norpuf1 nubwoo1 pabspa1 palfly2 palpri1 piecro1 piekin1 pitwhy purgre2 pygbat1 quailf1 ratcis1 raybar1 rbsrob1 rebfir2 rebhor1 reboxp1 reccor reccuc1 reedov1 refbar2 refcro1 reftin1 refwar2 rehblu1 rehwea1 reisee2 rerswa1 rewsta1 rindov rocmar2 rostur1 ruegls1 rufcha2 sacibi2 sccsun2 scrcha1 scthon1 shesta1 sichor1 sincis1 slbgre1 slcbou1 sltnig1 sobfly1 somgre1 somtit4 soucit1 soufis1 spemou2 spepig1 spewea1 spfbar1 spfwea1 spmthr1 spwlap1 squher1 strher strsee1 stusta1 subbus1 supsta1 tacsun1 tafpri1 tamdov1 thrnig1 trobou1 varsun2 vibsta2 vilwea1 vimwea1 walsta1 wbgbir1 wbrcha2 wbswea1 wfbeat1 whbcan1 whbcou1 whbcro2 whbtit5 whbwea1 whbwhe3 whcpri2 whctur2 wheslf1 whhsaw1 whihel1 whrshr1 witswa1 wlwwar wookin1 woosan wtbeat1 yebapa1 yebbar1 yebduc1 yebere1 yebgre1 yebsto1 yeccan1 yefcan yelbis1 yenspu1 yertin1 yesbar1 yespet1 yetgre1 yewgre1".split()

    base_model_name = "tf_efficientnet_b0_ns"
    pretrained = False
    num_classes = 264
    in_channels = 1
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


class BirdDataset(Dataset):
    def __init__(
            self,
            mode: str,
            folds_csv: str,
            dataset_dir: str,
            fold: int = 0,
            n_classes: int = 264,
            transforms=False,
            multiplier: int = 1,
            duration: int = 30,
            val_duration: int = 5,
    ):
        ## many parts from https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/main/data/ps_ds_2.py
        self.folds_csv = folds_csv
        self.df = pd.read_csv(folds_csv)
        # take sorted labels from full df
        birds = sorted(list(set(self.df.primary_label.values)))
        print('Number of primary labels ', len(birds))
        if mode =="train":
            self.df = self.df[self.df['fold'] != fold]
        else:
            self.df = self.df[self.df['fold'] == fold]

        self.dataset_dir = dataset_dir

        self.mode = mode

        self.duration = duration if mode == "train" else val_duration
        self.sr = 32000
        self.dsr = self.duration * self.sr

        self.n_classes = n_classes
        self.transforms = transforms

        self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)
        vc = self.df.primary_label.value_counts()
        dataset_length = len(self.df)
        label_weight = {}
        for row in vc.items():
            label, count = row
            label_weight[label] = math.pow(dataset_length / count, 1 / 2)

        self.df["label_weight"] = self.df.primary_label.apply(lambda x: label_weight[x])

        self.bird2id = {x: idx for idx, x in enumerate(birds)}

        ## TODO: move augmentation assignment outside of dataset
        if self.mode == "train":
            print(f"mode {self.mode} - augmentation is active {train_aug}")
            self.transforms = train_aug
            if multiplier > 1:
                self.df = pd.concat([self.df] * multiplier, ignore_index=True)

    def load_one(self, filename, offset, duration):
        try:
            wav, _ = librosa.load(filename, sr=None, offset=offset, duration=duration)
            if wav.shape[0] < self.dsr:
                wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        except:
            print("failed reading", filename)
        return wav


    def get_weights(self):
        return self.df.label_weight.values

    def __getitem__(self, i):
        tries = 0
        while tries < 20:
            try:
                tries += 1
                return self.getitem(i)
            except:
                traceback.print_stack()
                return self.getitem(random.randint(0, len(self) - 1))
        raise Exception("OOPS, something is wrong!!!")

    def getitem(self, i):
        row = self.df.iloc[i]
        if 'pretrain' in self.folds_csv:
            filename = os.path.join(self.dataset_dir, f"{row['filename'].split('.')[0]}.ogg")
            if not os.path.exists(filename):
                filename = filename.replace(".ogg", ".wav")
        else:
            if 'only_ml' in self.folds_csv:
                filename = os.path.join(self.dataset_dir, 'shared', row['filename'])
            elif 'pseudo' in self.folds_csv:
                filename = os.path.join(self.dataset_dir, 'shared', row['filename'])
            else:
                data_year = row['data_year']
                filename = os.path.join(self.dataset_dir, f"birdclef-{int(data_year)}",
                                        "train_short_audio" if data_year == 2021 else "train_audio", row['filename'])

        ## wav
        if self.mode == "train":
            wav_len_sec =  librosa.get_duration(filename=filename)
            duration = self.duration
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        if self.mode == "val": offset = 0

        wav = self.load_one(filename, offset=offset, duration=self.duration)
        #加入\加速语音\音调增强\时间移动
        '''zengqiang
        '''
        # 加入\加速语音\音调增强\时间移动
        if self.mode == "train":

            if row["is_zengqiang"] ==1:
                wav = zengqiang.shijianyidong(wav)
        if self.transforms:
            wav = self.transforms(wav, self.sr)
        image = librosa.feature.melspectrogram(y=wav, sr=params.sr, n_mels=params.n_mels, n_fft=params.n_fft,
                                               hop_length=params.hop_length, fmin=params.fmin, fmax=params.fmax, )
        image = librosa.power_to_db(image.astype(np.float32), ref=np.max)
        #加入\时域掩盖\频域掩盖\
        '''mask
        '''
        #加入\时域掩盖\频域掩盖\
        if self.mode == "train":

            if random.random()>0.5:
                image = zengqiang.pinyuyangai(image)
            if random.random()>0.5:
                image = zengqiang.shiyuyangai(image)

        if config["in_channels"] == 3:
            image = mono_to_color3(image)
            image = image.astype(np.uint8)
            image = albu_transforms2['valid'](image=image)['image'].T
        else:
            image = mono_to_color(image)
            image = image.astype(np.uint8)
            image = albu_transforms['valid'](image=image)['image'].T


        ## labels
        labels = torch.zeros((self.n_classes,))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 1.0
            except:
                ## if not in 21 classes, ignore
                continue

        ## weight
        weight = torch.tensor(row['weight'])

        return {
            "wav": torch.tensor(image),
            "labels": labels,
            "weight": weight
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
class zengqiang:
    # 加速语音
    # def jiasuyuyin(filename, sr=32000, min_speed=0.8, max_speed=1.2):
    #     """
    #     librosa时间拉伸
    #     :param samples: 音频数据，一维
    #     :param max_speed: 不要低于0.9，太低效果不好
    #     :param min_speed: 不要高于1.1，太高效果不好
    #     :return:
    #     """
    #
    #     rng = np.random.default_rng()
    #     speed = rng.uniform(min_speed, max_speed)
    #     samples, _ = librosa.load(filename, sr=sr, duration=5 * speed)
    #     samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    #     data_type = samples[0].dtype
    #     samples = samples.astype(np.float)
    #     samples = librosa.effects.time_stretch(samples, speed)
    #     samples = samples.astype(data_type)
    #     return samples

    # 音调增强
    def yindiaozenqiang(wav, sr=32000, ratio=5):

        samples = wav.copy()  # frombuffer()导致数据不可更改因此使用拷贝
        data_type = samples[0].dtype
        samples = samples.astype('float')
        ratio = random.uniform(-ratio, ratio)
        samples = librosa.effects.pitch_shift(samples, sr, n_steps=ratio)
        samples = samples.astype(data_type)
        return samples

    # 时间移动
    def shijianyidong(wav, sr=32000, max_ratio=0.3):
        """
        改进:
        1.为在一定比例范围内随机偏移，不再需要时间
        2.循环移动
        :param samples: 音频数据
        :param max_ratio:
        :return:
        """

        samples = wav.copy()
        frame_num = samples.shape[0]
        max_shifts = frame_num * max_ratio  # around 5% shift
        shifts_num = np.random.randint(-max_shifts, max_shifts)
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

if __name__ == "__main__":
    print("run")
    dataset = BirdDataset(mode="train", folds_csv="../pseudo_set_1.csv", dataset_dir="/mnt/d/kaggle/input/", fold=0, transforms=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    for batch in dataloader: break
    print(batch['wav'].shape)
