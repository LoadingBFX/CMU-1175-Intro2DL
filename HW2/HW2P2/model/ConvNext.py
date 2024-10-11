#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/5/2024 7:07 PM
# @Author  : Loading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LockDrop(nn.Module):
    def __init__(self, drop_prob=0.5):
        super(LockDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        # Implementing block-wise dropout. You can tweak the shape of the mask based on what structure you'd like to drop.
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.drop_prob)).to(x.device)
        return mask * x


class ConvNext(torch.nn.Module):
    def __init__(self, num_classes=8631, dropout_rate=0.5):
        super().__init__()

        # Load pre-trained ConvNeXt-T backbone
        convnext = models.convnext_tiny()

        # Remove the final fully connected layer from ConvNeXt-T
        self.backbone = torch.nn.Sequential(*list(convnext.children())[:-1])  # Remove only the last FC layer

        # Apply LockDrop in the middle of feature extraction to enhance robustness
        self.lockdrop = LockDrop(drop_prob=dropout_rate)

        # Classifier layer for custom number of classes
        self.cls_layer = torch.nn.Linear(convnext.classifier[2].in_features, num_classes)

    def forward(self, x):
        # Extract features from the first half of the backbone
        for i in range(len(self.backbone) // 2):
            x = self.backbone[i](x)

        # Apply LockDrop after some layers of the backbone
        x = self.lockdrop(x)

        # Continue extracting features
        for i in range(len(self.backbone) // 2, len(self.backbone)):
            x = self.backbone[i](x)

        feats = torch.flatten(x, 1)

        # Classification (not affected by LockDrop)
        out = self.cls_layer(feats)

        return {"feats": feats, "out": out}

    def get_feature_extractor_params(self):
        return self.backbone.parameters()

    def get_classifier_params(self):
        return self.cls_layer.parameters()
