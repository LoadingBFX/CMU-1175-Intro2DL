#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 9:15 PM
# @Author  : Loading
import torch
from torch import nn
from torch.nn import functional as F

class Network(nn.Module):
    def __init__(self, input_size=28, output_size=41, hidden_size=256, num_layers=1):
        super(Network, self).__init__()

        # 可选的卷积层用于特征提取
        self.embedding = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)

        # 双向 LSTM 层
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)

        # 分类层，将 LSTM 输出映射到音素类别
        self.classification = nn.Linear(hidden_size * 2, output_size)  # 乘以 2 是因为是双向的

        # Log Softmax 层，用于兼容 CTC 损失
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, lx=None):
        """
        网络的前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, input_size)。
            lx (torch.Tensor, optional): 批次中每个序列的长度。

        Returns:
            torch.Tensor: 每个时间步的对数概率，形状为 (batch_size, seq_length, output_size)。
        """
        # 应用嵌入层并转置
        x = self.embedding(x.transpose(1, 2)).transpose(1, 2)

        # 如果提供了 lx，则使用 Pack，将序列打包
        if lx is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            # 如果没有提供 lx，直接传递 x
            lstm_out, _ = self.lstm(x)

        # 分类层
        logits = self.classification(lstm_out)

        # 应用 log softmax
        log_probs = self.logSoftmax(logits)

        return log_probs