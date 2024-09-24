#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/21/2024 11:52 PM
# @Author  : Loading

import torch
# TODO: Fill out the model definition below

class CNNNetwork(torch.nn.Module):

    def __init__(self, num_classes=8631):
        super().__init__()

        self.backbone = torch.nn.Sequential(
            # TODO
            )

        self.cls_layer = #TODO

    def forward(self, x):
            # TODO:
        return {"feats": feats, "out": out}

