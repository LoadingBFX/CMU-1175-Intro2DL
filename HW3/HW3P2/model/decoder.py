#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 10:23 PM
# @Author  : Loading
import torch
from torch import nn

from model.permute_block import PermuteBlock


class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size=41):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            PermuteBlock(), torch.nn.BatchNorm1d(embed_size), PermuteBlock(),
            # TODO define your MLP arch. Refer HW1P2
            # Use Permute Block before and after BatchNorm1d() to match the size
            nn.Linear(embed_size, 2048),
            nn.GELU(),
            PermuteBlock(), torch.nn.BatchNorm1d(2048), PermuteBlock(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.GELU(),
            PermuteBlock(), torch.nn.BatchNorm1d(1024), PermuteBlock(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_size)
        )

        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        out = self.mlp(encoder_out)
        out = self.softmax(out)

        return out