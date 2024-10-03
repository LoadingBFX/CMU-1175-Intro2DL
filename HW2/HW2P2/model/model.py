#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/21/2024 11:52 PM
# @Author  : Loading

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CNNNetwork(torch.nn.Module):

    def __init__(self, num_classes=8631):
        super().__init__()

        # Load pre-trained ResNet-50 backbone
        resnet = models.resnet50()

        # Remove the final fully connected layer from ResNet
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove only the last FC layer

        # Classifier layer for custom number of classes
        self.cls_layer = torch.nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        out = self.cls_layer(feats)

        return {"feats": feats, "out": out}

    def get_feature_extractor_params(self):
        return self.backbone.parameters()

    def get_classifier_params(self):
        return self.cls_layer.parameters()
