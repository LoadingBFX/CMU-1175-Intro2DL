#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/28/2024 12:19 AM
# @Author  : Loading
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root, partition="test-clean"):
        """
        Initializes the dataset.
        :param root:
        :param partition:
        """

        # MFCC directory - use partition to acces train/dev directories from kaggle data using root
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        # List files in sefl.mfcc_dir using os.listdir in sorted order
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))

        self.length = len(self.mfcc_files)

        self.mfccs, self.transcripts = [], []

        for i in range(len(self.mfcc_files)):
            #   Load a single mfcc
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

            self.mfccs.append(mfcc)

    def __len__(self):

        """
        Returns the length of the dataset
        """
        return self.length

    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]
        return torch.FloatTensor(mfcc)

    def collate_fn(self, batch):
        batch_mfcc = []
        lengths_mfcc = []

        for mfcc in batch:
            batch_mfcc.append(mfcc)
            lengths_mfcc.append(len(mfcc))

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)