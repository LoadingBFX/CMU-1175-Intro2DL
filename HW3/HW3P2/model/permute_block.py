#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 9:38 PM
# @Author  : Loading
import torch

class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)