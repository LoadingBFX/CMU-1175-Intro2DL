#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/6/2024 7:45 PM
# @Author  : Loading
import os

import torch
import pandas as pd
from tqdm import tqdm
from ctcdecode import CTCBeamDecoder
from utils import decode_prediction, load_config, set_seed, load_model
from dataset.AudioDatasetTest import AudioDatasetTest
from model.ASRModel import ASRModel

# Define constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = ('./ckpt/epoch_19.pth')
CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())


PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]

TEST_BEAM_WIDTH = 100


def main():
    set_seed(42)
    cfg = load_config("./config/config.yaml")
    best_model_path = os.path.join(cfg['save_model_folder'], 'best_model.pth')

    torch.cuda.empty_cache()
    model = ASRModel(
        input_size=cfg['model']['input_size'],
        embed_size=cfg['model']['embed_size'],
        output_size=len(PHONEMES)
    ).to(DEVICE)

    loaded = load_model(MODEL_PATH, model) # [model, optimizer, scheduler, epoch, metric]
    model = loaded[0]
    model.eval()

    # Initialize the decoder
    test_decoder = CTCBeamDecoder(LABELS, beam_width=TEST_BEAM_WIDTH, log_probs_input=True)

    # Load the test data
    test_data = AudioDatasetTest(root=cfg['data_folder'],
                                 partition="test-clean")

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=2,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=test_data.collate_fn
    )

    # Perform inference
    results = []
    print("Testing")
    for data in tqdm(test_loader):
        x, lx = data
        x = x.to(DEVICE)

        with torch.no_grad():
            h, lh = model(x, lx)

        prediction_string = decode_prediction(h, lh, test_decoder, LABELS)
        results.extend(prediction_string)

        del x, lx, h, lh
        torch.cuda.empty_cache()

    # Save results to CSV
    df = pd.DataFrame({'index': range(len(results)), 'label': results})
    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()