#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/5/2024 7:07 PM
# @Author  : Loading
import torch
from torchvision import models

class ConvNext(torch.nn.Module):

    def __init__(self, num_classes=8631, dropout_rate=0.5):
        super().__init__()

        # Load pre-trained ConvNeXt-T backbone
        convnext = models.convnext_tiny()

        # Remove the final fully connected layer from ConvNeXt-T
        self.backbone = torch.nn.Sequential(*list(convnext.children())[:-1])  # Remove only the last FC layer

        # # Dropout layer
        # self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Classifier layer for custom number of classes
        self.cls_layer = torch.nn.Linear(convnext.classifier[2].in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)

        # Apply dropout
        # feats = self.dropout(feats)

        out = self.cls_layer(feats)

        return {"feats": feats, "out": out}

    def get_feature_extractor_params(self):
        return self.backbone.parameters()

    def get_classifier_params(self):
        return self.cls_layer.parameters()
