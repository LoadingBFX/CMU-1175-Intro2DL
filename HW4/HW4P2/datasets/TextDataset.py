#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/13/2024 11:35 PM
# @Author  : Loading
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import Literal, List, Optional, Any, Dict, Tuple
from utils.mytokenizer import GTokenizer


class TextDataset(Dataset):
    def __init__(self, partition: str, config:dict, tokenizer: GTokenizer):
        """
        Initializes the TextDataset class, which loads and tokenizes transcript files.

        Args:
            partition (str): Subdirectory under root that specifies the data partition (e.g., 'train', 'test').
            config (dict): Configuration dictionary for dataset settings.
            tokenizer (GTokenizer): Tokenizer instance for encoding transcripts into token sequences.
        """

        # General attributes
        self.root      = config['root']
        self.subset    = config['subset']
        self.partition = partition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN  # End of sequence token
        self.sos_token = tokenizer.SOS_TOKEN  # Start of sequence token
        self.pad_token = tokenizer.PAD_TOKEN  # Padding token

        # Paths and files
        self.text_dir = os.path.join(self.root, self.partition)  # Directory containing transcript files
        self.text_files = sorted(os.listdir(self.text_dir))  # Sorted list of transcript files

        # Limit to subset of files if specified
        subset = int(self.subset * len(self.text_files))
        self.text_files = self.text_files[:subset]
        self.length = len(self.text_files)

        # Storage for encoded transcripts
        self.transcripts_shifted, self.transcripts_golden = [], []

        # Load and encode transcripts
        for file in tqdm(self.text_files, desc=f"Loading transcript data for {partition}"):
            transcript = np.load(os.path.join(self.text_dir, file)).tolist()
            transcript = " ".join(transcript.split())  # Process text
            tokenized = self.tokenizer.encode(transcript)  # Tokenize transcript
            # Store shifted and golden versions of transcripts
            self.transcripts_shifted.append(np.array([self.eos_token] + tokenized))
            self.transcripts_golden.append(np.array(tokenized + [self.eos_token]))

    def __len__(self) -> int:
        """Returns the total number of transcripts in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Retrieves the shifted and golden version of the transcript at the specified index.

        Args:
            idx (int): Index of the transcript to retrieve.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: Shifted and golden version of the transcript.
        """
        shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
        golden_transcript = torch.LongTensor(self.transcripts_golden[idx])
        return shifted_transcript, golden_transcript

    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a batch of transcripts for model input, applying padding as needed.

        Args:
            batch (List[Tuple[torch.LongTensor, torch.LongTensor]]): Batch of (shifted, golden) transcripts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Padded shifted transcripts (batch_transcript_pad).
                - Padded golden transcripts (batch_golden_pad).
                - Lengths of shifted transcripts.
        """

        # Separate shifted and golden transcripts from batch
        batch_transcript = [i[0] for i in batch]  # B x T
        batch_golden = [i[1] for i in batch]  # B x T
        lengths_transcript = [len(i) for i in batch_transcript]

        # Pad sequences
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=self.pad_token)
        batch_golden_pad = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # Return padded sequences and lengths
        return batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_transcript)
