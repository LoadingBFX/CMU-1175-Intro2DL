import os
import numpy as np
import torch

### PHONEME LIST
PHONEMES = [
            '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]']


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, root, size, phonemes = PHONEMES, context=0, partition= "train-clean-100"): # Feel free to add more arguments

        self.context    = context
        self.phonemes   = phonemes

        self.mfcc_dir       = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        mfcc_names          = sorted(os.listdir(self.mfcc_dir))[:size]
        transcript_names    = sorted(os.listdir(self.transcript_dir))[:size]

        # Making sure that we have the same no. of mfcc and transcripts
        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []

        # TODO: Iterate through mfccs and transcripts
        for i in range(len(mfcc_names)):
        #   Load a single mfcc
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_names[i])
            mfcc        = np.load(mfcc_path)
        #   Do Cepstral Normalization of mfcc (explained in writeup)
            mfcc_mean = np.mean(mfcc, axis=0)
            mfcc_std = np.std(mfcc, axis=0)
            mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
        #   Load the corresponding transcript
            transcript_path  = os.path.join(self.transcript_dir, transcript_names[i])
            transcript = np.load(transcript_path)
            # Remove [SOS] and [EOS] from the transcript
            if len(transcript) > 0 and transcript[0] == '[SOS]':
              transcript = transcript[1:]
            if len(transcript) > 0 and transcript[-1] == '[EOS]':
              transcript = transcript[:-1]
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.

        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        # NOTE:
        # Each mfcc is of shape T1 x 28, T2 x 28, ...
        # Each transcript is of shape (T1+2), (T2+2) before removing [SOS] and [EOS]

        # TODO: Concatenate all mfccs in self.mfccs such that
        # the final shape is T x 28 (Where T = T1 + T2 + ...)
        self.mfccs          = np.concatenate(self.mfccs, axis=0)

        # TODO: Concatenate all transcripts in self.transcripts such that
        # the final shape is (T,) meaning, each time step has one phoneme output
        self.transcripts    = np.concatenate(self.transcripts)
        # Hint: Use numpy to concatenate

        # Length of the datasets is now the length of concatenated mfccs/transcripts
        self.length = len(self.mfccs)

        # Take some time to think about what we have done.
        # self.mfcc is an array of the format (Frames x Features).
        # Our goal is to recognize phonemes of each frame
        # We can introduce context by padding zeros on top and bottom of self.mfcc
        self.mfccs = np.pad(self.mfccs, ((self.context, self.context), (0, 0)), 'constant', constant_values=0) # TODO

        # The available phonemes in the transcript are of string datasets type
        # But the neural network cannot predict strings as such.
        # Hence, we map these phonemes to integers

        # TODO: Map the phonemes to their corresponding list indexes in self.phonemes
        self.transcripts = np.array([self.phonemes.index(p) for p in self.transcripts])
        # Now, if an element in self.transcript is 0, it means that it is 'SIL' (as per the above example)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
        frames = self.mfccs[ind:ind + 2 * self.context + 1]
        # After slicing, you get an array of shape 2*context+1 x 28. But our MLP needs 1d datasets and not 2d.
        frames = frames.flatten() # TODO: Flatten to get 1d datasets

        frames      = torch.FloatTensor(frames) # Convert to tensors
        phonemes    = torch.tensor(self.transcripts[ind])

        return frames, phonemes