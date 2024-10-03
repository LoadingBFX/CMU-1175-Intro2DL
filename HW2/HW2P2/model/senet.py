#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/29/2024 10:12 AM
# @Author  : Loading

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # 全局平均池化
        se = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        # 两层全连接
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(batch_size, channels, 1, 1)
        return x * se  # 重新加权输入特征图


# SE-ResNet
class SENetwork(nn.Module):
    def __init__(self, num_classes=8631):
        super(SENetwork, self).__init__()

        # 加载预训练的ResNet-50模型
        resnet = models.resnet50()

        # 修改ResNet中的每个Bottleneck block，插入SE模块
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后的全连接层

        # 将SE模块嵌入每个ResNet block的最后
        self.se1 = SEModule(256)
        self.se2 = SEModule(512)
        self.se3 = SEModule(1024)
        self.se4 = SEModule(2048)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 自定义分类器层
        self.cls_layer = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Backbone 提取特征
        x = self.backbone[0](x)  # 第一层卷积
        x = self.backbone[1](x)  # BatchNorm
        x = self.backbone[2](x)  # ReLU
        x = self.backbone[3](x)  # MaxPool

        # ResNet blocks with SE modules
        x = self.backbone[4](x)  # layer1
        x = self.se1(x)
        x = self.backbone[5](x)  # layer2
        x = self.se2(x)
        x = self.backbone[6](x)  # layer3
        x = self.se3(x)
        x = self.backbone[7](x)  # layer4
        x = self.se4(x)

        # 池化层
        feats = self.avgpool(x)
        feats = torch.flatten(feats, 1)

        # 分类器输出
        out = self.cls_layer(feats)

        return {"feats": feats, "out": out}
