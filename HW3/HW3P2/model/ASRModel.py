#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 10:27 PM
# @Author  : Loading
import torch
import torchaudio

from model.decoder import Decoder
from model.encoder import Encoder
from model.permute_block import PermuteBlock


class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size=192, output_size=41, cfg=None):
        super().__init__()
        if cfg is not None:
            self.augmentations = torch.nn.Sequential(
                PermuteBlock(),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=cfg['specaug']["freq_mask_param"]),
                torchaudio.transforms.TimeMasking(time_mask_param=cfg['specaug']["time_mask_param"]),
                PermuteBlock(),
            )
            print("Using SpecAugment")
            print(cfg['specaug'])

        self.encoder = Encoder(input_size, embed_size, cfg)
        self.decoder = Decoder(embed_size * 2, output_size, cfg)

    def forward(self, x, lengths_x):
        if self.training and hasattr(self, 'augmentations'):
            x = self.augmentations(x)

        encoder_out, encoder_lens = self.encoder(x, lengths_x)
        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_lens