import sys

import torch
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
from typing import Literal
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct

from utils.mytokenizer import GTokenizer


class SpeechDataset(Dataset):

    def __init__(self,
                 partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
                 config: dict,
                 tokenizer: GTokenizer,
                 isTrainPartition: bool
                 ):
        """
        Initialize the SpeechDataset.

        Args:
            partition (str): Partition name
            config (dict): Configuration dictionary for dataset settings.
            tokenizer (GTokenizer): Tokenizer class for encoding and decoding text data.
            isTrainPartition (bool): Flag indicating if this partition is for training.
        """

        # general: Get config values
        self.config = config
        self.root = self.config['root']
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN
        self.sos_token = tokenizer.SOS_TOKEN
        self.pad_token = tokenizer.PAD_TOKEN
        self.subset = self.config['subset']
        self.feat_type = self.config['feat_type']
        self.num_feats = self.config['num_feats']
        self.norm = self.config['norm']

        # paths | files
        self.fbank_dir = os.path.join(self.root, self.partition, "fbank")
        self.fbank_files = sorted(os.listdir(self.fbank_dir))
        subset = int(self.subset * len(self.fbank_files))
        self.fbank_files = sorted(os.listdir(self.fbank_dir))[:subset]

        if self.partition != 'test-clean':
            self.text_dir = os.path.join(self.root, self.partition, "text")
            self.text_files = sorted(os.listdir(self.text_dir))
            self.text_files = sorted(os.listdir(self.text_dir))[:subset]
            assert len(self.fbank_files) == len(
                self.text_files), "Number of fbank files and text files must be the same"

        self.length = len(self.fbank_files)
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []

        for i in tqdm(range(len(self.fbank_files)), desc=f"Loading fbank and transcript data for {self.partition}"):
            # load features
            feats = np.load(os.path.join(self.fbank_dir, self.fbank_files[i]))
            # print(f"Original feats shape for file {self.fbank_files[i]}: {feats.shape}")
            if self.feat_type == 'mfcc':
                feats = self.fbank_to_mfcc(feats)

            if self.config['norm'] == 'cepstral':
                feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1E-8)

            # print(f"Feats shape after processing for file {self.fbank_files[i]}: {feats.shape}")
            self.feats.append(feats[:self.num_feats, :])

            # load and encode transcripts
            # Why do we have two different types of targets?
            # How do we want our decoder to know the start of sequence <SOS> and end of sequence <EOS>?

            if self.partition != 'test-clean':
                # Note: You dont have access to transcripts in dev_clean
                transcript = np.load(os.path.join(self.text_dir, self.text_files[i])).tolist()
                transcript = "".join(transcript)
                # Invoke our tokenizer to tokenize the string
                tokenized = self.tokenizer.encode(transcript)
                self.transcripts_shifted.append(np.array([self.sos_token] + tokenized))
                self.transcripts_golden.append(np.array(tokenized + [self.eos_token]))

        if self.partition != 'test-clean':
            assert len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)

        # precompute global stats for global mean and variance normalization
        self.global_mean, self.global_std = None, None
        if self.config['norm'] == 'global_mvn':
            self.global_mean, self.global_std = self.compute_global_stats()

        # Torch Audio Transforms
        # time masking
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=self.config["specaug_conf"]["time_mask_width_range"])

        # frequency masking
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.config["specaug_conf"]["freq_mask_width_range"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])
        shifted_transcript, golden_transcript = None, None
        if self.partition != 'test-clean':
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript = torch.LongTensor(self.transcripts_golden[idx])
        # Apply global mean and variance normalization if enabled
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch):
        # @NOTE: batch corresponds to output from __getitem__ for a minibatch

        '''
        1.  Extract the features and labels from 'batch'.
        2.  We will additionally need to pad both features and labels,
            look at PyTorch's documentation for pad_sequence.
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lengths of features, and lengths of labels.

        '''
        # Prepare batch (features)
        batch_feats = [item[0].transpose(0, 1) for item in
                       batch]  # TODO: position of feats do you return from get_item + transpose B x T x F
        lengths_feats = [feat.shape[0] for feat in batch_feats]  # Lengths of each T x F sequence
        batch_feats_pad = pad_sequence(batch_feats, batch_first=True, padding_value=0)  # Pad sequence

        if self.partition != 'test-clean':
            batch_transcript = [item[1] for item in batch]  # TODO: # B x T
            batch_golden = [item[2] for item in batch]  # TODO # B x T
            lengths_transcript = [len(i) for i in batch_transcript]  # Lengths of each T
            batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=self.pad_token)
            batch_golden_pad = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # print(f"Before SpecAugment: batch_feats_pad.shape = {batch_feats_pad.shape}")
        # TODO: do specaugment transforms
        if self.config["specaug"] and self.isTrainPartition:

            # transpose back to F x T to apply transforms
            batch_feats_pad = batch_feats_pad.transpose(1, 2)

            # shape should be B x num_feats x T
            assert batch_feats_pad.shape[1] == self.num_feats

            # freq_mask
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    batch_feats_pad = self.freq_mask(batch_feats_pad)
            # print(f"After Frequency Mask: batch_feats_pad.shape = {batch_feats_pad.shape}")

            # time mask
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    batch_feats_pad = self.time_mask(batch_feats_pad)
            # print(f"After Time Mask: batch_feats_pad.shape = {batch_feats_pad.shape}")
            # transpose back to T x F
            batch_feats_pad = batch_feats_pad.transpose(1, 2)
            # shape should be B x T x num_feats
            assert batch_feats_pad.shape[2] == self.num_feats

        # Return the following values:
        # padded features, padded shifted labels, padded golden labels, actual length of features, actual length of the shifted label
        if self.partition != 'test-clean':
            return batch_feats_pad, batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_feats), torch.tensor(
                lengths_transcript)
        else:
            return batch_feats_pad, None, None, torch.tensor(lengths_feats), None

    def fbank_to_mfcc(self, fbank):
        # Helper function that applies the dct to the filterbank features to concert them to mfccs
        mfcc = dct(fbank.T, type=2, axis=1, norm='ortho')
        return mfcc.T

    # Will be discussed in bootcamp
    # def compute_global_stats(self):
    #     raise NotImplementedError

    # Will be discussed in bootcamp
    def compute_global_stats(self):
        print("Computing global mean and variance for normalization")
        all_feats = np.concatenate([feat for feat in self.feats], axis=1)  # concatenate along time axis (F x T)
        global_mean = np.mean(all_feats, axis=1)  # Mean across time(per frequency bin)
        global_std = np.std(all_feats, axis=1) + 1e-20  # Standard deviation across time(per
        print(f"computed global mean: {global_mean.shape}, global variance: {global_std.shape}")
        # return torch.FloatTensor(global_mean), torch.floatTensor(global_std)
        return torch.FloatTensor(global_mean), torch.FloatTensor(global_std)
