#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/24/2024 9:25 PM
# @Author  : Loading
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class ArcFaceLoss(nn.Module):
#     def __init__(self, s=30.0, m=0.50, num_classes=8631, embedding_size=512):
#         super(ArcFaceLoss, self).__init__()
#         self.s = s  # scale factor
#         self.m = m  # margin
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, features, labels):
#         # Normalize features and weights
#         cosine = F.linear(F.normalize(features), F.normalize(self.weight))  # cosine similarity
#
#         # Add the margin to the cosine angle
#         theta = torch.acos(cosine.clamp(-1.0, 1.0))
#         target_logits = torch.cos(theta + self.m)
#
#         # Create one-hot encoded labels
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1, 1), 1)
#
#         # Apply the margin to the target classes
#         output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
#
#         # Rescale the logits using the scale factor
#         output = output * self.s
#
#         return output

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.s = s  # 缩放因子
        self.m = m  # 边距
        # 权重矩阵，形状为 (out_features, in_features)，即类别数和特征维度
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # 初始化权重

    def forward(self, input, label):
        # 将权重移动到与输入相同的设备
        self.weight = self.weight.to(input.device)

        # 计算归一化的输入特征和权重之间的余弦相似度
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # 计算夹角并添加边距
        theta = torch.acos(cosine.clamp(-1.0, 1.0))  # 夹角计算
        target_logit = torch.cos(theta + self.m)  # 加上边距

        # 构建 one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 使用 one-hot 确保只有正确类别的角度会加上边距
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)

        # 使用缩放因子 s 调整 logits 的大小
        output *= self.s
        return output

