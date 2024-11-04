#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 10:07 PM
# @Author  : Loading
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class pBLSTM(torch.nn.Module):
    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size=2 * input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True,
                             dropout=0.2, batch_first=True)

    def forward(self, x_packed):  # x_packed is a PackedSequence

        x, lengths = pad_packed_sequence(x_packed, batch_first=True)

        x, x_lens = self.trunc_reshape(x, lengths)

        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        x, h = self.blstm(x)

        return x

    def trunc_reshape(self, x, x_lens):
        if x.shape[1] % 2 != 0:
            x = x[:, :-1, :]

        x = x.reshape(x.shape[0], x.shape[1] // 2, x.shape[2] * 2)
        x_lens = x_lens // 2
        return x, x_lens