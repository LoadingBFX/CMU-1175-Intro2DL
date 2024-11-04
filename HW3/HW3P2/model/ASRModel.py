#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 10:27 PM
# @Author  : Loading
import torch

from model.decoder import Decoder
from model.encoder import Encoder


class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size=192, output_size=41):
        super().__init__()

        # self.augmentations  = torch.nn.Sequential(
        #     #TODO Add Time Masking/ Frequency Masking
        #     #Hint: See how to use PermuteBlock() function defined above
        #     PermuteBlock(),
        #     torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
        #     torchaudio.transforms.TimeMasking(time_mask_param=100),
        #     PermuteBlock(),
        # ) # did augmentation in the collate_fn
        self.encoder = Encoder(input_size, embed_size)  # TODO: Initialize Encoder
        self.decoder = Decoder(embed_size * 2, output_size)  # TODO: Initialize Decoder

    def forward(self, x, lengths_x):
        # if self.training:
        #     x = self.augmentations(x)

        encoder_out, encoder_lens = self.encoder(x, lengths_x)
        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_lens