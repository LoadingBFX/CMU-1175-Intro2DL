#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/24/2024 9:25 PM
# @Author  : Loading
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50, num_classes=8631, embedding_size=512):
        super(ArcFaceLoss, self).__init__()
        self.s = s  # scale factor
        self.m = m  # margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))  # cosine similarity

        # Add the margin to the cosine angle
        theta = torch.acos(cosine.clamp(-1.0, 1.0))
        target_logits = torch.cos(theta + self.m)

        # Create one-hot encoded labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply the margin to the target classes
        output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)

        # Rescale the logits using the scale factor
        output = output * self.s

        return output
