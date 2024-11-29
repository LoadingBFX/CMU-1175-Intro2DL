#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/18/2024 1:07 PM
# @Author  : Loading
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device

from models.PositionalEncoding import PositionalEncoding
from utils.mask import PadMask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )  # Hint: Linear layer - GELU - dropout - Linear layer
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        # Step 1: Apply pre-normalization
        x_norm = self.pre_norm(x)

        # Step 2: Self-attention with with dropout, and with residual connection
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=pad_mask)
        x = x + self.dropout(attn_output)

        # Step 3: Apply normalization
        x = self.norm1(x)

        # Step 4: Apply Feed-Forward Network (FFN) with dropout, and residual connection
        ffn_output = self.ffn1(x)
        x = x + self.dropout(ffn_output)

        # Step 5: Apply normalization after FFN
        x = self.norm2(x)

        return x, pad_mask


class Encoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 max_len,
                 target_vocab_size,
                 dropout=0.1):
        super(Encoder, self).__init__()

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.after_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, target_vocab_size)

    def forward(self, x, x_len):
        # Step 1: Create padding mask for inputs
        pad_mask = PadMask(x, input_lengths=x_len).bool()

        # Step 2: Apply positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Step 3: Apply dropout
        x = self.dropout(x)

        # Step 4: Add the residual connection (before passing through layers)
        # ''' TODO '''

        # Step 5: Pass through all encoder layers
        for layer in self.enc_layers:
            x, pad_mask = layer(x, pad_mask)

        # Step 6: Apply final normalization
        x = self.after_norm(x)

        # Step 7: Pass a branch through the CTC head
        x_ctc = self.ctc_head(x)

        return x, x_len, x_ctc.log_softmax(2).permute(1, 0, 2)