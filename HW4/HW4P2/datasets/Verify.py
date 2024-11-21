#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/19/2024 5:28 PM
# @Author  : Loading
from tqdm import tqdm


def verify_dataset(dataloader, partition):
    '''Compute the Maximum MFCC and Transcript sequence length in a dataset'''
    print("Loaded Path: ", partition)
    max_len_feat = 0
    max_len_t    = 0  # To track the maximum length of transcripts

    # Iterate through the dataloader
    for batch in tqdm(dataloader, desc=f"Verifying {partition} Dataset"):
        try:
            x_pad, y_shifted_pad, y_golden_pad, x_len, y_len = batch

            # Update the maximum feat length
            len_x = x_pad.shape[1]
            if len_x > max_len_feat:
                max_len_feat = len_x

            # Update the maximum transcript length
            # transcript length is dim 1 of y_shifted_pad
            if y_shifted_pad is not None:
                len_y = y_shifted_pad.shape[1]
                if len_y > max_len_t:
                    max_len_t = len_y

        except Exception as e:
            # The text dataset has no transcripts
            y_shifted_pad, y_golden_pad, y_len = batch

            # Update the maximum transcript length
            # transcript length is dim 1 of y_shifted_pad
            len_y = y_shifted_pad.shape[1]
            if len_y > max_len_t:
                max_len_t = len_y


    print(f"Maximum Feat Length in Dataset       : {max_len_feat}")
    print(f"Maximum Transcript Length in Dataset : {max_len_t}")
    return max_len_feat, max_len_t