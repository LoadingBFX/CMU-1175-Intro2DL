#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/18/2024 1:39 PM
# @Author  : Loading
import torch
import torch.nn as nn
import numpy as np
from typing import Literal

from models.Embedding import SpeechEmbedding
from models.myDecoder import Decoder
from models.myEncoder import Encoder


class Transformer(nn.Module):
    def __init__(self,
                 input_dim,
                 time_stride,
                 feature_stride,
                 embed_dropout,

                 d_model,
                 enc_num_layers,
                 enc_num_heads,
                 speech_max_len,
                 enc_dropout,

                 dec_num_layers,
                 dec_num_heads,
                 d_ff,
                 dec_dropout,

                 target_vocab_size,
                 trans_max_len):

        super(Transformer, self).__init__()

        self.embedding = SpeechEmbedding(input_dim, d_model, time_stride, feature_stride, embed_dropout)
        speech_max_len = int(np.ceil(speech_max_len / self.embedding.time_downsampling_factor))

        self.encoder = Encoder(enc_num_layers, d_model, enc_num_heads, d_ff, speech_max_len, target_vocab_size,
                               enc_dropout)

        self.decoder = Decoder(dec_num_layers, d_model, dec_num_heads, d_ff, dec_dropout, trans_max_len,
                               target_vocab_size)

    def forward(self, padded_input, input_lengths, padded_target, target_lengths,
                mode: Literal['full', 'dec_cond_lm', 'dec_lm'] = 'full'):
        '''DO NOT MODIFY'''
        if mode == 'full':  # Full transformer training
            encoder_output, encoder_lengths = self.embedding(padded_input, input_lengths, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        if mode == 'dec_cond_lm':  # Training Decoder as a conditional LM
            encoder_output, encoder_lengths = self.embedding(padded_input, input_lengths, use_blstm=True)
            ctc_out = None
        if mode == 'dec_lm':  # Training Decoder as an LM
            encoder_output, encoder_lengths, ctc_out = None, None, None

        # passing Encoder output through Decoder
        output, attention_weights = self.decoder(padded_target, target_lengths, encoder_output, encoder_lengths)
        return output, attention_weights, ctc_out

    def recognize(self, inp, inp_len, tokenizer, mode: Literal['full', 'dec_cond_lm', 'dec_lm'],
                  strategy: str = 'greedy'):
        """ sequence-to-sequence greedy search -- decoding one utterance at a time """
        '''DO NOT MODIFY'''
        if mode == 'full':
            encoder_output, encoder_lengths = self.embedding(inp, inp_len, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)

        if mode == 'dec_cond_lm':
            encoder_output, encoder_lengths, = self.embedding(inp, inp_len, use_blstm=True)
            ctc_out = None

        if mode == 'dec_lm':
            encoder_output, encoder_lengths, ctc_out = None, None, None

        if strategy == 'greedy':
            out = self.decoder.recognize_greedy_search(encoder_output, encoder_lengths, tokenizer=tokenizer)
        elif strategy == 'beam':
            out = self.decoder.recognize_beam_search(encoder_output, encoder_lengths, tokenizer=tokenizer, beam_width=5)
        return out