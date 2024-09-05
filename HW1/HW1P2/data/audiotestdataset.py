import os
import numpy as np
import torch


class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, context=0, partition="test-clean"):
        self.context = context
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.mfccs = []

        for mfcc_name in mfcc_names:
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_name)
            mfcc = np.load(mfcc_path)
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
            self.mfccs.append(mfcc)

        self.mfccs = np.concatenate(self.mfccs, axis=0)
        self.length = len(self.mfccs)
        self.mfccs = np.pad(self.mfccs, ((context, context), (0, 0)), 'constant', constant_values=0)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        frames = self.mfccs[ind:ind + 2 * self.context + 1]
        frames = frames.flatten()
        frames = torch.FloatTensor(frames)
        return frames