#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/26/2024 10:04 PM
# @Author  : Loading
import torch

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)
        negative_distance = (anchor - negative).pow(2).sum(1)
        loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0.0))
        return loss
