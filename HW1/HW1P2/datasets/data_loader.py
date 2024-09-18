import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

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

        # the final shape is T x 28 (Where T = T1 + T2 + ...)
        self.mfccs          = np.concatenate(self.mfccs, axis=0)
        # pca_test(self.mfccs)

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

        self.transcripts = np.array([self.phonemes.index(p) for p in self.transcripts])
        # Now, if an element in self.transcript is 0, it means that it is 'SIL' (as per the above example)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        frames = self.mfccs[ind:ind + 2 * self.context + 1]

        # After slicing, you get an array of shape 2*context+1 x 28. But our MLP needs 1d datasets and not 2d.
        frames = frames.flatten()

        frames      = torch.FloatTensor(frames) # Convert to tensors
        phonemes    = torch.tensor(self.transcripts[ind])

        return frames, phonemes


if __name__ == "__main__":
    DATASET_PATH = "../data/11785-f24-hw1p2"
    PHONEMES = [
        '[SIL]', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
        'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
        'V', 'W', 'Y', 'Z', 'ZH', '[SOS]', '[EOS]'
    ]

    batch_size = 5
    context = 2
    train_size = 5  # 使用5个样本进行小规模测试
    val_size = 2

    # 测试数据集的长度
    dataset = AudioDataset(DATASET_PATH, train_size, PHONEMES, context, 'train-clean-100', training=True)
    # print(f"Dataset Length: {len(dataset)}")
    # assert len(dataset) == train_size * 100  # 5个样本，每个样本100帧

    # 检查某个样本的形状
    # frames, phonemes = dataset[10]
    # print("Sample Frame Shape:", frames.shape)
    # assert frames.shape == (5 * 28,)  # 2*context + 1 帧，展平后长度是 5*28
    # assert isinstance(phonemes, torch.Tensor)
    #
    # # 测试上下文填充
    # frames, phonemes = dataset[0]
    # print("First Sample Frames (Padding Check):", frames[:28])  # 打印前28个值
    # assert (frames[:28] == 0).all()  # 检查最前面的 2 个上下文帧是否被 padding 为 0
    # assert frames.shape == (5 * 28,)  # 2*context + 1 帧

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 打印测试样本
    for i, (frames, phonemes) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print("Frames shape:", frames.shape)
        print("Phonemes:", phonemes)
        print()

        # 可视化帧的第一个样本
        plt.imshow(frames[0].view(2 * context + 1, 28).numpy(), cmap='hot', interpolation='nearest')
        plt.title(f"Sample {i + 1} - MFCC Features")
        plt.colorbar()
        plt.show()

        if i == 4:  # 仅可视化前两个 batch
            break