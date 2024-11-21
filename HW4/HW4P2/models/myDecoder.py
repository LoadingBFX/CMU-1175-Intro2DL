#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/18/2024 1:15 PM
# @Author  : Loading
import torch
import torch.nn as nn
import torch.nn.functional as F
from heapq import heappush, heappop

from models.PositionalEncoding import PositionalEncoding
from utils.mask import PadMask, CausalMask


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # @TODO: fill in the blanks appropriately (given the modules above)
        self.mha1       = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout,batch_first=True)
        self.mha2       = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout,batch_first=True)

        self.ffn        = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.identity   = nn.Identity()
        self.pre_norm   = nn.LayerNorm(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.dropout3   = nn.Dropout(dropout)

    def forward(self, padded_targets, enc_output, pad_mask_enc, pad_mask_dec, slf_attn_mask):

        #   Step 1: Self Attention
        #   (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization

        x = self.pre_norm(padded_targets)
        mha1_output, mha1_attn_weights = self.mha1(x, x, x, key_padding_mask=pad_mask_dec, attn_mask=slf_attn_mask)
        mha1_output = self.dropout1(mha1_output)
        x = x + mha1_output  # Residual connection
        x = self.layernorm1(x)


        #   Step 2: Cross Attention
        #   (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
              #  think about if key,value,query here are the same as the previous one?
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization

        if enc_output is None:
            # TODO: Implement this
            mha2_output       = self.identity(padded_targets)
            mha2_attn_weights = torch.zeros_like(mha1_attn_weights)
        else:
            mha2_output, mha2_attn_weights = self.mha2(x, enc_output, enc_output, key_padding_mask=pad_mask_enc)
            mha2_output = self.dropout2(mha2_output)
            x = x + mha2_output  # Residual connection
            x = self.layernorm2(x)

        #   Step 3: Feed Forward Network
        #   (1) pass through the FFN
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization
        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output)
        x = x + ffn_output  # Residual connection
        # TODO: Implement this
        x = self.layernorm3(x)

        return ffn_output, mha1_attn_weights, mha2_attn_weights


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff, dropout,
                 max_len,
                 target_vocab_size):

        super().__init__()

        self.max_len        = max_len
        self.num_layers     = num_layers
        self.num_heads      = num_heads

        # use torch.nn.ModuleList() with list comprehension looping through num_layers
        # @NOTE: think about what stays constant per each DecoderLayer (how to call DecoderLayer)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.target_embedding       = nn.Embedding(target_vocab_size, d_model)  # use torch.nn.Embedding
        self.positional_encoding    = PositionalEncoding(d_model, max_len)
        self.final_linear           = nn.Linear(d_model, target_vocab_size)
        self.dropout                = nn.Dropout(dropout)


    def forward(self, padded_targets, target_lengths, enc_output, enc_input_lengths):

        # Processing targets
        # create a padding mask for the padded_targets with <PAD_TOKEN>
        # creating an attention mask for the future subsequences (look-ahead mask)
        # computing embeddings for the target sequence
        # computing Positional Encodings with the embedded targets and apply dropout

        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_input=padded_targets, input_lengths=target_lengths).to(padded_targets.device)
        causal_mask = CausalMask(input_tensor=padded_targets).to(padded_targets.device)

        # Step1:  Apply the embedding
        x = self.target_embedding(padded_targets)

        # Step2:  Apply positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)


        # Step3:  Create attention mask to ignore padding positions in the input sequence during attention calculation
        pad_mask_enc = None
        if enc_output is not None:
            pad_mask_enc = PadMask(padded_input=enc_output, input_lengths=enc_input_lengths).to(enc_output.device)


        # Step4: Pass through decoder layers
        # @NOTE: store your mha1 and mha2 attention weights inside a dictionary
        # @NOTE: you will want to retrieve these later so store them with a useful name
        runnint_att = {}
        for i in range(self.num_layers):
            x, runnint_att['layer{}_dec_self'.format(i + 1)], runnint_att['layer{}_dec_cross'.format(i + 1)] = self.dec_layers[i](
                x, enc_output, pad_mask_enc, pad_mask_dec, causal_mask
            )


        # Step5: linear layer (Final Projection) for next character prediction
        seq_out = self.final_linear(x)

        return seq_out, runnint_att


    def recognize_greedy_search(self, enc_output, enc_input_lengths, tokenizer):
        ''' passes the encoder outputs and its corresponding lengths through autoregressive network
            @NOTE: You do not need to make changes to this method.
        '''
        # start with the <SOS> token for each sequence in the batch
        batch_size = enc_output.size(0)
        target_seq = torch.full((batch_size, 1), tokenizer.SOS_TOKEN, dtype=torch.long).to(enc_output.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_output.device)

        for _ in range(self.max_len):

            seq_out, runnint_att = self.forward(target_seq, None, enc_output, enc_input_lengths)
            logits = torch.nn.functional.log_softmax(seq_out[:, -1], dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            # appending the token to the sequence
            # checking if <EOS> token is generated
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            # end if all sequences have generated the EOS token
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            target_seq = torch.cat([target_seq, next_token], dim=-1)
            eos_mask = next_token.squeeze(-1) == tokenizer.EOS_TOKEN
            finished |= eos_mask
            if finished.all(): break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(target_seq,(0, self.max_len - max_length), value=tokenizer.PAD_TOKEN)

        return target_seq


    # def recognize_beam_search(self, enc_output, enc_input_lengths, tokenizer):
    #   # TODO Beam Decoding
    #   raise NotImplementedError

