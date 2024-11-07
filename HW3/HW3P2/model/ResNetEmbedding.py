#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/5/2024 9:12 PM
# @Author  : Loading

import torch.nn as nn
from model.permute_block import PermuteBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout)
        self.identity = nn.Identity()

        if in_channels != out_channels:
            self.identity = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetEmbedding(nn.Module):
    def __init__(self, input_dim, expand_dims, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ResNetEmbedding, self).__init__()
        self.permute = PermuteBlock()
        self.initial_conv = nn.Conv1d(input_dim, expand_dims[0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.residual_block1 = ResidualBlock(expand_dims[0], expand_dims[1], kernel_size, stride, padding, dropout)
        self.residual_block2 = ResidualBlock(expand_dims[1], expand_dims[1], kernel_size, stride, padding, dropout)
        self.final_conv = nn.Conv1d(expand_dims[1], expand_dims[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.permute(x)
        out = self.initial_conv(out)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.final_conv(out)
        out = self.relu(out)
        out = self.permute(out)
        return out