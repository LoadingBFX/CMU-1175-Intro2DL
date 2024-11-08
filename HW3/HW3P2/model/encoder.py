#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 10:22 PM
# @Author  : Loading
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.ResNetEmbedding import ResNetEmbedding
from model.pBLSTM import pBLSTM
from model.permute_block import PermuteBlock


class LockedDropout(nn.Module):
    def __init__(self, drop_prob):
        super(LockedDropout, self).__init__()
        self.prob = drop_prob

    def forward(self, x):
        if not self.training or self.prob == 0:  # Turn off during inference
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.prob)
        mask = mask / (1 - self.prob)
        mask = mask.expand_as(x)
        out = x * mask
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, cfg):
        super(Encoder, self).__init__()
        # expand_dims = [128, 512]
        expand_dims = cfg['encoder']['expand_dims']
        kernel_size = cfg['encoder']['kernel_size']

        self.embed = nn.Sequential(
            PermuteBlock(),
            nn.Conv1d(in_channels=input_size, out_channels=expand_dims[0], kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(num_features=expand_dims[0]),
            nn.GELU(),
            nn.Conv1d(in_channels=expand_dims[0], out_channels=expand_dims[1], kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(num_features=expand_dims[1]),
            PermuteBlock()
        )
        # self.embed = ResNetEmbedding(
        #     input_dim=input_size,
        #     expand_dims=expand_dims,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )

        self.pBLSTMs = nn.Sequential(
            pBLSTM(input_size=expand_dims[1], hidden_size=hidden_size),
            # LockedDropout(cfg['pBLSTMs']['dropout_prob']),
            pBLSTM(input_size=2 * hidden_size, hidden_size=hidden_size),
            # LockedDropout(cfg['pBLSTMs']['dropout_prob']),
        )
        self.layer_norm_0 = nn.LayerNorm(hidden_size * 2)
        self.locked_dropout_0 = LockedDropout(cfg['pBLSTMs']['dropout_prob'])
        self.layer_norm_1 = nn.LayerNorm(hidden_size * 2)
        self.locked_dropout_1 = LockedDropout(cfg['pBLSTMs']['dropout_prob'])


    def forward(self, x, lens):
        x = self.embed(x)
        lens = lens.clamp(max=x.shape[1]).cpu()

        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x = self.pBLSTMs[0](x)
        x, lens = pad_packed_sequence(x, batch_first=True)
        x = self.layer_norm_0(x)
        x = self.locked_dropout_0(x)

        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x = self.pBLSTMs[1](x)
        x, lens = pad_packed_sequence(x, batch_first=True)
        x = self.layer_norm_1(x)
        x = self.locked_dropout_1(x)

        return x, lens