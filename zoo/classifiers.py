from functools import partial
import random
from typing import Dict
from torchlibrosa.augmentation import SpecAugmentation
import timm
import torch
from timm.models.convnext import LayerNorm2d
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from nnAudio.Spectrogram import STFT
import numpy as np
import torchaudio as ta
import time

from zoo.oned import OneDConvNet


from .oned import OneDConvNet


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


default_config = {'sample_rate': 32000,
                  'window_size': 1024,
                  'n_fft': 1024,
                  'hop_size': 320,
                  'fmin': 50,
                  'fmax': 14000,
                  'mel_bins': 128,
                  'power': 2,
                  'top_db': None,
                  'in_channels':1}


class SED(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=False,
                 classes=21,
                 attn_activation='linear',
                 mel_config: Dict = default_config,
                 **kwargs
                 ):
        super().__init__()

        print("initing SED model...")

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['window_size'],
            win_length=mel_config['window_size'],
            hop_length=mel_config['hop_size'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            pad=0,
            n_mels=mel_config['mel_bins'],
            power=mel_config['power'],
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=mel_config['top_db'])
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

        base_model = timm.create_model(
            encoder, pretrained=pretrained, **kwargs)
        self.encoder = base_model

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        elif hasattr(base_model, "num_features"):
            in_features = base_model.num_features
        else:
            in_features = base_model.classifier.in_features
        self.features = self.encoder.forward_features
        if encoder.startswith("vgg"):
            in_features = 512
            base_model.pre_logits = None
            self.features = self.encoder.features
        self.norm_layer = None
        if encoder.startswith("convnext"):
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            self.norm_layer = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, in_features, bias=True)

        print("ATT activation:", attn_activation)
        self.att_block = AttBlockV2(
            in_features, classes, activation=attn_activation)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    ## TODO: optional normalization of mel
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)  # (bs, mel, time)
            x = (x + 80) / 80

        frames_num = x.size(3)

        # (batch_size, channels, freq, frames)
        x = self.features(x)
        if self.norm_layer:
            x = self.norm_layer(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        #todo: check how it works
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        return {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }


class TimmSED(nn.Module):
    def __init__(
            self,
            base_model_name: str,
            config=None,
            pretrained=False,
            num_classes=24,
            in_channels=1
    ):
        super().__init__()

        self.config = config

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64 // 2,
            time_stripes_num=2,
            freq_drop_width=8 // 2,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(self.config.n_mels)

        base_model = timm.create_model(
            base_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
        )

        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input_data):

        if self.config.in_channels == 3:
            x = input_data
        else:
            x = input_data[:, [0], :, :]  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            if random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = x.transpose(2, 3)

        x = self.encoder(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict


class TimmClassifier_v1(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=False,
                 classes=21,
                 enable_masking=False,
                 **kwargs
                 ):
        super().__init__()

        print(f"initing CLS features model {kwargs['duration']} duration...")

        mel_config = kwargs['mel_config']
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['window_size'],
            win_length=mel_config['window_size'],
            hop_length=mel_config['hop_size'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            pad=0,
            n_mels=mel_config['mel_bins'],
            power=mel_config['power'],
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=mel_config['top_db'])
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.enable_masking = enable_masking
        if enable_masking:
            self.freq_mask = ta.transforms.FrequencyMasking(24, iid_masks=True)
            self.time_mask = ta.transforms.TimeMasking(64, iid_masks=True)

        ## fix https://github.com/rwightman/pytorch-image-models/issues/488#issuecomment-796322390
        import pathlib
        import timm.models.nfnet as nfnet

        model_name = "eca_nfnet_l0"
        checkpoint_path = "weights/pretrained_eca_nfnet_l0.pth"
        checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()

        nfnet.default_cfgs[model_name]["url"] = checkpoint_path_url

        print("pretrained model...")
        print(kwargs['backbone_params'])
        base_model = timm.create_model(
            encoder, pretrained=pretrained,
            features_only=True, out_indices=([4]),
            **kwargs['backbone_params']
         )

        self.encoder = base_model

        self.gem = GeM(p=3, eps=1e-6)
        self.head1 = nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True)
        
        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        self.factor = int(wav_crop_len / 5.0)

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):
        if  is_test == False:
            x = x[:, 0, :] # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :] # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)   # bs, ch, mel, time
            x = (x + 80) / 80
    
        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]
        
        ## TODO: better loop
        xss = []
        for x in self.encoder(x):
            if self.training:
                b, c, t, f = x.shape
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(b // self.factor, self.factor * t, c, f)
                x = x.permute(0, 2, 1, 3)

            x = self.gem(x)
            x = x[:, :, 0, 0]
            xss.append(x)

        logit = self.head1(xss[0])
        return {"logit": logit}


class TimmClassifier2021(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=False,
                 classes=21,
                 enable_masking=False,
                 **kwargs
                 ):
        super().__init__()

        print(f"initing CLS features model {kwargs['duration']} duration...")

        mel_config = kwargs['mel_config']
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['window_size'],
            win_length=mel_config['window_size'],
            hop_length=mel_config['hop_size'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            pad=0,
            n_mels=mel_config['mel_bins'],
            power=mel_config['power'],
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=mel_config['top_db'])
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.enable_masking = enable_masking
        if enable_masking:
            self.freq_mask = ta.transforms.FrequencyMasking(24, iid_masks=True)
            self.time_mask = ta.transforms.TimeMasking(64, iid_masks=True)

        print("pretrained model...")
        print(kwargs['backbone_params'])

        base_model = timm.create_model(
            encoder,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            **kwargs['backbone_params']
        )
        if "efficientnet" in encoder:
            backbone_out = base_model.num_features
        else:
            backbone_out = base_model.feature_info[-1]["num_chs"]

        self.backbone = base_model

        self.global_pool = GeM(p=3, eps=1e-6)
        self.head = nn.Linear(backbone_out, classes, bias=True)

        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        self.factor = int(wav_crop_len / 5.0)

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):
        if is_test == False:
            x = x[:, 0, :]  # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :]  # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)  # bs, ch, mel, time
            x = (x + 80) / 80

        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        x = self.backbone(x)
        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)

        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logit = self.head(x)
        return {"logit": logit}


class TimmClassifier(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=False,
                 classes=21,
                 enable_masking=False,
                 **kwargs
                 ):
        super().__init__()

        print(f"initing CLS features model {kwargs['duration']} duration...")

        mel_config = kwargs['mel_config']
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['window_size'],
            win_length=mel_config['window_size'],
            hop_length=mel_config['hop_size'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            pad=0,
            n_mels=mel_config['mel_bins'],
            power=mel_config['power'],
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=mel_config['top_db'])
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.enable_masking = enable_masking
        if enable_masking:
            self.freq_mask = ta.transforms.FrequencyMasking(24, iid_masks=True)
            self.time_mask = ta.transforms.TimeMasking(64, iid_masks=True)

        print("pretrained model...")
        print(kwargs['backbone_params'])


        base_model = timm.create_model(
            encoder,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            **kwargs['backbone_params']
         )
        if "efficientnet" in encoder:
            backbone_out = base_model.num_features
        else:
            backbone_out = base_model.feature_info[-1]["num_chs"]

        self.encoder = base_model

        self.gem = GeM(p=3, eps=1e-6)
        self.head1 = nn.Linear(backbone_out, classes, bias=True)

        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        self.factor = int(wav_crop_len / 5.0)

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):
        if  is_test == False:
            x = x[:, 0, :] # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :] # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)   # bs, ch, mel, time
            x = (x + 80) / 80

        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        x = self.encoder(x)
        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)

        x = self.gem(x)
        x = x[:, :, 0, 0]
        logit = self.head1(x)
        return {"logit": logit}



class SEDTrainableFFT(nn.Module):
    def __init__(self, encoder: str, mel_config: Dict = default_config, classes=21, attn_activation="linear",
                 pretrained=True, trainable_fft=True, **kwargs):
        super().__init__()
        self.stft = STFT(n_fft=mel_config["window_size"],
                         win_length=mel_config["window_size"],
                         hop_length=mel_config["hop_size"],
                         window=('tukey', 0.25),
                         freq_scale='no',
                         pad_mode='reflect',
                         sr=mel_config["sample_rate"],
                         fmin=mel_config["fmin"],
                         fmax=mel_config["fmax"],
                         output_format="Magnitude", trainable=trainable_fft)
        base_model = timm.create_model(
            encoder, pretrained=pretrained, **kwargs)
        self.encoder = base_model

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        elif hasattr(base_model, "num_features"):
            in_features = base_model.num_features
        else:
            in_features = base_model.classifier.in_features
        self.features = self.encoder.forward_features
        self.norm_layer = None
        if encoder.startswith("convnext"):
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            self.norm_layer = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, in_features, bias=True)

        print("ATT activation:", attn_activation)
        self.att_block = AttBlockV2(
            in_features, classes, activation=attn_activation)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            bs = x.size(0)
            x = x.reshape(bs, -1)
            x = self.stft(x)
            _, h, w = x.shape
            x = x.reshape(bs, 1, h, w)

        frames_num = x.size(3)

        # (batch_size, channels, freq, frames)
        x = self.features(x)
        if self.norm_layer:
            x = self.norm_layer(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        return {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }


class C1C2(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=True,
                 classes=21,
                 attn_activation='linear',
                 **kwargs
                 ):
        super().__init__()

        print("initing SED model...")

        base_model = timm.create_model(
            encoder, pretrained=pretrained, **kwargs)
        self.encoder = base_model
        self.conv1d = OneDConvNet(32, 64)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        elif hasattr(base_model, "num_features"):
            in_features = base_model.num_features
        else:
            in_features = base_model.classifier.in_features
        self.features = self.encoder.forward_features
        if encoder.startswith("vgg"):
            in_features = 512
            base_model.pre_logits = None
            self.features = self.encoder.features
        self.norm_layer = None
        if encoder.startswith("convnext"):
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            self.norm_layer = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, in_features, bias=True)

        print("ATT activation:", attn_activation)
        self.att_block = AttBlockV2(
            in_features, classes, activation=attn_activation)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    ## TODO: optional normalization of mel
    def forward(self, x):
        x = self.conv1d(x)
        x = x.unsqueeze(1)
        frames_num = x.size(3)

        # (batch_size, channels, freq, frames)
        x = self.features(x)
        if self.norm_layer:
            x = self.norm_layer(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        #todo: check how it works
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        return {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

class TimmClassifierSplitCrop_v1(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=False,
                 classes=21,
                 **kwargs
                 ):
        super().__init__()

        print(f"initing CLS features model {kwargs['duration']} duration...")

        mel_config = kwargs['mel_config']
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['window_size'],
            win_length=mel_config['window_size'],
            hop_length=mel_config['hop_size'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            pad=0,
            n_mels=mel_config['mel_bins'],
            power=mel_config['power'],
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=mel_config['top_db'])
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)


        ## fix https://github.com/rwightman/pytorch-image-models/issues/488#issuecomment-796322390
        import pathlib
        import timm.models.nfnet as nfnet

        # model_name = "eca_nfnet_l0"
        # checkpoint_path = "weights/pretrained_eca_nfnet_l0.pth"
        # checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()
        #
        # nfnet.default_cfgs[model_name]["url"] = checkpoint_path_url

        print("pretrained model...")
        base_model = timm.create_model(
            encoder, pretrained=pretrained,
            features_only=True, out_indices=([4]),
            **kwargs['backbone_params']
         )

        self.encoder = base_model
        in_features = base_model.feature_info[-1]["num_chs"]

        self.gem = GeM(p=3, eps=1e-6)
        self.head1 = nn.Linear(in_features, classes, bias=True)

        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        self.factor = int(wav_crop_len / 5.0)

    def forward(self, x, is_test=False):
        if  is_test == False:
            x = x[:, 0, :] # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :] # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)   # bs, ch, mel, time
            x = (x + 80) / 80

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        for x in self.encoder(x):
            x = self.gem(x)[:, :, 0, 0]
            logit = self.head1(x)
            b, c = logit.shape

            if self.training:
                logit = logit.reshape(b // self.factor, self.factor, c)
                logit = logit.max(dim=1).values


        return {"logit": logit}


if __name__ == "__main__":
    net = C1C2(encoder="resnet34", in_chans=1)
    out = net(torch.zeros((1, 1, 32000 * 30)))
    print(out)
