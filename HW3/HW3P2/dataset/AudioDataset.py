#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/28/2024 12:18 AM
# @Author  : Loading
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, phonemes, partition="train-clean-100"):
        """
        Initializes the dataset.

        Args:
            root (str): Root directory containing the dataset.
            phonemes (list): List of phonemes.
            partition (str): Dataset partition to load ('train-clean-100', 'test-clean', etc.).
            audio_transforms (callable, optional): Optional transform to be applied on the audio data.
        """
        self.PHONEMES = phonemes

        # Load directories and check if they exist
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        if not os.path.exists(self.mfcc_dir) or not os.path.exists(self.transcript_dir):
            raise FileNotFoundError("MFCC or transcript directory not found. Please check the file paths.")

        # Get and sort file lists
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.transcript_files = sorted(os.listdir(self.transcript_dir))

        # Ensure we have the same number of mfcc and transcript files
        assert len(self.mfcc_files) == len(self.transcript_files), "Mismatch in number of MFCC and transcript files."
        self.length = len(self.mfcc_files)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.length

    def __getitem__(self, ind):
        # Load MFCC data
        mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[ind]))

        # Normalize MFCC with epsilon to prevent division by zero
        epsilon = 1e-6
        std = np.std(mfcc, axis=0)
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / (std + epsilon)

        # Check for NaN or inf values in mfcc
        if np.isnan(mfcc).any() or np.isinf(mfcc).any():
            print(f"Warning: NaN or inf values detected in MFCC at index {ind}")
            # Optionally handle the case, e.g., by returning a default value or raising an error

        mfcc = torch.FloatTensor(mfcc)  # Convert to tensor

        # Load and preprocess transcript
        transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[ind]))
        if transcript[0] == "[SOS]" and transcript[-1] == "[EOS]":
            transcript = transcript[1:-1]  # Remove start and end symbols
        transcript = np.vectorize(self.PHONEMES.index)(transcript)  # Convert phonemes to indices
        transcript = torch.tensor(transcript, dtype=torch.long)  # Convert to tensor

        return mfcc, transcript

    def collate_fn(self, batch):
        """
        Custom collate function for padding sequences in batch.

        Args:
            batch (list of tuples): Each tuple contains (mfcc, transcript).

        Returns:
            tuple: Padded features, padded labels, lengths of features, lengths of labels.
        """
        # Separate features and labels
        batch_mfcc = [item[0] for item in batch]
        batch_transcript = [item[1] for item in batch]

        # Compute original lengths
        lengths_mfcc = torch.tensor([mfcc.size(0) for mfcc in batch_mfcc], dtype=torch.long)
        lengths_transcript = torch.tensor([transcript.size(0) for transcript in batch_transcript], dtype=torch.long)

        # Pad sequences
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)

        return batch_mfcc_pad, batch_transcript_pad, lengths_mfcc, lengths_transcript
