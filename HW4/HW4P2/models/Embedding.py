#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/18/2024 12:56 PM
# @Author  : Loading
# 2-Layer BiLSTM
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 2-Layer BiLSTM
class BiLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()
        self.bilstm = nn.LSTM(
            input_dim, output_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, x, x_len):
        """
        Args:
            x.    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        # Pack the padded sequence to avoid computing over padded tokens
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        # Pass through the BiLSTM
        packed_output, _ = self.bilstm(packed_input)
        # Unpack the sequence to restore the original padded shape
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output


### DO NOT MODIFY

class Conv2DSubsampling(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, time_stride=2, feature_stride=2):
        """
        Conv2dSubsampling module that can selectively apply downsampling
        for time and feature dimensions, and calculate cumulative downsampling factor.
        Args:
            time_stride (int): Stride along the time dimension for downsampling.
            feature_stride (int): Stride along the feature dimension for downsampling.
        """
        super(Conv2DSubsampling, self).__init__()

        # decompose to get effective stride across two layers
        tstride1, tstride2 = self.closest_factors(time_stride)
        fstride1, fstride2 = self.closest_factors(feature_stride)

        self.feature_stride = feature_stride
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, kernel_size=3, stride=(tstride1, fstride1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=(tstride2, fstride2)),
            torch.nn.ReLU(),
        )

        self.time_downsampling_factor = tstride1 * tstride2
        # Calculate output dimension for the linear layer
        conv_out_dim = (input_dim - (3 - 1) - 1) // fstride1 + 1
        conv_out_dim = (conv_out_dim - (3 - 1) - 1) // fstride2 + 1
        conv_out_dim = output_dim * conv_out_dim
        self.out = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, output_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            x_mask (torch.Tensor): Optional mask for the input tensor.

        Returns:
            torch.Tensor: Downsampled output of shape (batch_size, new_seq_len, output_dim).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv2D
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x

    def closest_factors(self, n):
        factor = int(n ** 0.5)
        while n % factor != 0:
            factor -= 1
        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)


class SpeechEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, time_stride, feature_stride, dropout):
        super(SpeechEmbedding, self).__init__()

        self.cnn = Conv2DSubsampling(input_dim, output_dim, dropout=dropout, time_stride=time_stride,
                                     feature_stride=feature_stride)
        self.blstm = BiLSTMEmbedding(output_dim, output_dim, dropout)
        self.time_downsampling_factor = self.cnn.time_downsampling_factor

    def forward(self, x, x_len, use_blstm: bool = False):
        """
        Args:
            x    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len // stride, output_dim)
        """
        # First, apply Conv2D subsampling
        x = self.cnn(x)
        # Adjust sequence length based on downsampling factor
        x_len = torch.ceil(x_len.float() / self.time_downsampling_factor).int()
        x_len = x_len.clamp(max=x.size(1))

        # Apply BiLSTM if requested
        if use_blstm:
            x = self.blstm(x, x_len)

        return x, x_len