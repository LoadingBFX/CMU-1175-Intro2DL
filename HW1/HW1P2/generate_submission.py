#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/5/2024 10:45 PM
# @Author  : Loading

import torch

from HW1.HW1P2.datasets.test_data_loader import AudioTestDataset
from HW1.HW1P2.models.model import Network
from HW1.HW1P2.src.utils import generate_submission

PHONEMES = [
    '[SIL]', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
    'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH', '[SOS]', '[EOS]']

if __name__ == '__main__':
    MODEL_SAVE_PATH = "checkpoints/best_model - 0.85686.pth"
    DATASET_PATH = "./data/11785-f24-hw1p2"

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"*_ Using device: {device}\n")

    config = {
        'epochs'        : 50,
        'batch_size'    : 4096,
        'context'       : 35,
        'init_lr'       : 1e-4,
    }

    # Calculate input size
    INPUT_SIZE = (2 * config['context'] + 1) * 28
    print(f"*_ Input size calculated: {INPUT_SIZE}\n")

    model = Network(INPUT_SIZE, len(PHONEMES)).to(device)

    test_data = AudioTestDataset(DATASET_PATH, config['context'], 'test-clean')
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=2,
        batch_size=config['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    generate_submission(MODEL_SAVE_PATH, test_loader, model, device, PHONEMES)