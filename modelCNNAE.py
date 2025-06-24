import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.autograd import Function

import args


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class UVCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels=6,
        output_dims=64,
    ):
        super(UVCNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d((2,2),padding=1)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d((2, 2), padding=1)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d((3, 3), padding=1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.fc(x)
        return x


class UVCNNDecoder(nn.Module):
    def __init__(
        self,
        in_dims=64,
        output_dims=6,
    ):
        super(UVCNNDecoder, self).__init__()
        self.in_channels = in_dims
        self.fc = _fc(in_dims, 256, bias=False)
        self.conv1 = _conv2d(256, 128, 3, padding=1, bias=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = _conv2d(128, 64, 3, padding=1, bias=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = _conv2d(64, output_dims, 3, padding=1, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.contiguous().view(batch_size, -1, 1, 1)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        return x


class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()
        self.encoder = UVCNNEncoder(in_channels=6, output_dims=args.featureUV_dim)
        self.decoder = UVCNNDecoder(in_dims=args.featureUV_dim, output_dims=6)

    def forward(self, X):
        feat_emb = self.encoder(X)
        feat_pred = self.decoder(feat_emb)

        return feat_pred
