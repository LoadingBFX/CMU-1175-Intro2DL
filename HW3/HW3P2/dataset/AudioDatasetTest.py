#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/28/2024 12:19 AM
# @Author  : Loading
import os

import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root, partition="test-clean"):
        """
        Initializes the dataset.

        Args:
            root (str): Root directory containing the dataset.
            partition (str): Dataset partition to load ('test-clean', etc.).
        """
        # Set MFCC directory and check if it exists
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        if not os.path.exists(self.mfcc_dir):
            raise FileNotFoundError(f"MFCC directory not found at {self.mfcc_dir}")

        # Get and sort file list
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.length = len(self.mfcc_files)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.length

    def __getitem__(self, ind):
        """
        Loads and returns a single MFCC feature tensor.

        Args:
            ind (int): Index of the example.

        Returns:
            torch.FloatTensor: Normalized MFCC tensor.
        """
        # Load and normalize MFCC data
        mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[ind]))
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
        return torch.FloatTensor(mfcc)

    def collate_fn(self, batch):
        """
        Custom collate function for padding sequences in batch.

        Args:
            batch (list of torch.FloatTensor): List of MFCC tensors.

        Returns:
            tuple: Padded MFCC tensors and lengths of each sequence.
        """
        batch_mfcc = [mfcc for mfcc in batch]
        lengths_mfcc = torch.tensor([mfcc.size(0) for mfcc in batch_mfcc], dtype=torch.long)

        # Pad sequences
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)

        # Return padded features and actual lengths of features
        return batch_mfcc_pad, lengths_mfcc
