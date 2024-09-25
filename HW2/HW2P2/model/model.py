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

        # self.backbone = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),  # Output: 28x28
        #     torch.nn.BatchNorm2d(64),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 14x14
        #     torch.nn.BatchNorm2d(128),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 7x7
        #     torch.nn.BatchNorm2d(256),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output: 4x4
        #     torch.nn.BatchNorm2d(512),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # Output: 2x2
        #     torch.nn.BatchNorm2d(1024),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.AdaptiveAvgPool2d(1)  # Output: 1x1
        # )
        #
        # self.cls_layer = torch.nn.Linear(1024, num_classes)

        # Load pre-trained ResNet-50 backbone
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the final fully connected layer from ResNet
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove only the last FC layer

        # Classifier layer for custom number of classes
        self.cls_layer = torch.nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # TODO:
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        out = self.cls_layer(feats)

        return {"feats": feats, "out": out}

