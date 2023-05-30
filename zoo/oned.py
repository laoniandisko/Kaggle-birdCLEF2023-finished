import math
from typing import Optional

import timm
import torch
from torch import nn, conv1d
from torch.nn import SiLU, MaxPool1d, BatchNorm1d, AdaptiveAvgPool2d
from torch.nn.functional import pad
from torch.nn import functional as F

import zoo


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: int, s: int, d: int = 1, value: float = 0):
    iw = x.size()[-1]
    pad_w = get_same_padding(iw, k, s, d)
    if pad_w > 0:
        x = pad(x, [pad_w // 2, pad_w - pad_w // 2], value=value)
    return x


def conv1d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: int = 1,
        padding=0, dilation: int = 1, groups: int = 1):
    x = pad_same(x, weight.shape[-1], stride, dilation)
    return conv1d(x, weight, bias, stride, 0, dilation, groups)


class Conv1dSame(nn.Conv1d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv1d_same(x, self.weight, self.bias, self.stride[0], self.padding[0], self.dilation[0], self.groups)


class OneDConvNet(nn.Module):

    def __init__(self, filters_start=32, kernel_start=128, in_chans=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv1dSame(in_chans, filters_start, kernel_start, bias=False),
            BatchNorm1d(filters_start),
            SiLU(),
            Conv1dSame(filters_start, filters_start, kernel_start // 2, bias=False),
            BatchNorm1d(filters_start),
            SiLU(),
            MaxPool1d(kernel_size=8, stride=8)
        )
        self.conv2 = nn.Sequential(
            Conv1dSame(filters_start, filters_start * 2, kernel_start // 2, bias=False),
            BatchNorm1d(filters_start * 2),
            SiLU(),
            Conv1dSame(filters_start * 2, filters_start * 2, kernel_start // 4, bias=False),
            BatchNorm1d(filters_start * 2),
            SiLU(),
            MaxPool1d(kernel_size=8, stride=8)
        )
        self.conv3 = nn.Sequential(
            Conv1dSame(filters_start * 2, filters_start * 4, kernel_start // 4, bias=False),
            BatchNorm1d(filters_start * 4),
            SiLU(),
            Conv1dSame(filters_start * 4, filters_start * 4, kernel_start // 4, bias=False),
            BatchNorm1d(filters_start * 4),
            SiLU(),
            MaxPool1d(kernel_size=4, stride=4)
        )
        self.conv4 = nn.Sequential(
            Conv1dSame(filters_start * 4, filters_start * 8, kernel_start // 8, bias=False),
            BatchNorm1d(filters_start * 8),
            SiLU(),
            Conv1dSame(filters_start * 8, filters_start * 8, kernel_start // 8, bias=False),
            BatchNorm1d(filters_start * 8),
            SiLU(),
            MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            Conv1dSame(filters_start * 8, filters_start * 8, kernel_start // 8, bias=False),
            BatchNorm1d(filters_start * 8),
            SiLU(),
            Conv1dSame(filters_start * 8, filters_start * 8, kernel_start // 8, bias=False),
            BatchNorm1d(filters_start * 8),
            SiLU(),
        )
        _initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

