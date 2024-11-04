#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/28/2024 12:18 AM
# @Author  : Loading
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root, phonemes, partition="train-clean-100", audio_transforms=None):
        """
        Initializes the dataset.

        Args:
            root (str): Root directory containing the dataset.
            phonemes (list): List of phonemes.
            partition (str): Dataset partition to load ('train-clean-100', 'test-clean', etc.).
        """
        self.PHONEMES = phonemes
        self.audio_transforms = audio_transforms

        # Load the directory and all files in them

        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.transcript_files = sorted(os.listdir(self.transcript_dir))


        # Making sure that we have the same no. of mfcc and transcripts
        assert len(self.mfcc_files) == len(self.transcript_files)

        self.length = len(self.mfcc_files)

        self.mfccs, self.transcripts = [], []

        for i in range(len(self.mfcc_files)):
            #   Load a single mfcc
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            #   Do Cepstral Normalization of mfcc (explained in writeup)
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
            #   Load the corresponding transcript
            transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[i]))
            if transcript[0] == "[SOS]" and transcript[-1] == "[EOS]":
                transcript = transcript[1:-1]
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
            #   Append each mfcc to self.mfcc, transcript to self.transcript
            transcript = np.vectorize(self.PHONEMES.index)(transcript)

            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

    def __len__(self):

        """
                Returns the length of the dataset.
        """
        return self.length

    def __getitem__(self, ind):
        """
        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        """

        mfcc = torch.FloatTensor(self.mfccs[ind])
        transcript = torch.tensor(self.transcripts[ind])
        return mfcc, transcript

    def collate_fn(self, batch):
        """
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        """

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc, batch_transcript = [], []
        lengths_mfcc, lengths_transcript = [], []
        for (m, t) in batch:
            batch_mfcc.append(m)
            lengths_mfcc.append(len(m))
            batch_transcript.append(t)
            lengths_transcript.append(len(t))

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)


        if self.audio_transforms is not None:
            batch_mfcc_pad = self.audio_transforms(batch_mfcc_pad)


        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)
