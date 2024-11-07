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

        # self.mlp = torch.nn.Sequential(
        #     PermuteBlock(), torch.nn.BatchNorm1d(embed_size), PermuteBlock(),
        #     # Use Permute Block before and after BatchNorm1d() to match the size
        #     nn.Linear(embed_size, 2048),
        #     nn.GELU(),
        #     PermuteBlock(), torch.nn.BatchNorm1d(2048), PermuteBlock(),
        #     nn.Dropout(0.5),
        #     nn.Linear(2048, 1024),
        #     nn.GELU(),
        #     PermuteBlock(), torch.nn.BatchNorm1d(1024), PermuteBlock(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, output_size)
        # )

        self.mlp = torch.nn.Sequential(

            torch.nn.Linear(embed_size, 1024),
            PermuteBlock(), torch.nn.BatchNorm1d(1024), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(1024, 2048),
            PermuteBlock(), torch.nn.BatchNorm1d(2048), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.24),

            torch.nn.Linear(2048, 4096),
            PermuteBlock(), torch.nn.BatchNorm1d(4096), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.28),

            torch.nn.Linear(4096, 2048),
            PermuteBlock(), torch.nn.BatchNorm1d(2048), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.24),

            torch.nn.Linear(2048, 1024),
            PermuteBlock(), torch.nn.BatchNorm1d(1024), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 1024),
            PermuteBlock(), torch.nn.BatchNorm1d(1024), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 512),
            PermuteBlock(), torch.nn.BatchNorm1d(512), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),

            torch.nn.Linear(512, 256),
            PermuteBlock(), torch.nn.BatchNorm1d(256), PermuteBlock(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),


            torch.nn.Linear(256, output_size)
        )






        self.softmax = nn.LogSoftmax(dim=2)


    def forward(self, encoder_out):
        out = self.mlp(encoder_out)
        out = self.softmax(out)

        return out