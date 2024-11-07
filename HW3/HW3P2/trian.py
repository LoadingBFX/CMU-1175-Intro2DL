#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/28/2024 12:17 AM
# @Author  : Loading

import torchaudio
from s3prl.schedulers import get_linear_schedule_with_warmup

import wandb
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import gc

import pandas as pd
from tqdm import tqdm
import os
import datetime

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder
import gc
import warnings

from dataset.AudioDataset import AudioDataset
from dataset.AudioDatasetTest import AudioDatasetTest
from model.ASRModel import ASRModel
from model.basic import Network
from model.permute_block import PermuteBlock
from utils import load_config, set_seed, validate_model, train_model, save_model, decode_prediction

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

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
print("Phonemes: ", len(PHONEMES))

# audio_transforms = nn.Sequential(
#     PermuteBlock(),
#     torchaudio.transforms.FrequencyMasking(freq_mask_param=5),
#     torchaudio.transforms.TimeMasking(time_mask_param=100),
#     PermuteBlock()
# )


def main():
    # Set the random seed for reproducibility
    set_seed(42)
    import wandb

    # parse the config file from config.yaml
    cfg = load_config("./config/config.yaml")

    wandb.login(key="46b9373c96fe8f8327255e7da8a4046da7ffeef6")
    run = wandb.init(
        name=f"{cfg['train']['epochs']}ep_{cfg['model']['input_size']}in_{cfg['model']['embed_size']}em_{cfg['specaug']['freq_mask_param']}fmp_{cfg['specaug']['time_mask_param']}tmp_{cfg['decode']['beam_width']}bw_{cfg['encoder']['kernel_size']}ks_{cfg['pBLSTMs']['dropout_prob']}pblstm_{cfg['decoder']['dropout_prob']}ddp",
        project="hw3p2-ablations",  ### Project should be created in your wandb account
        config=cfg  ### Wandb Config for your run
    )


    # audio_transforms = nn.Sequential(
    #     PermuteBlock(),
    #     torchaudio.transforms.FrequencyMasking(freq_mask_param= cfg['specaug']["freq_mask_param"]),
    #     torchaudio.transforms.TimeMasking(time_mask_param=cfg['specaug']["time_mask_param"]),
    #     PermuteBlock()
    # )

    last_epoch_completed = 0
    start = last_epoch_completed
    end = cfg['train']["epochs"]
    best_lev_dist = float("inf")  # if you're restarting from some checkpoint, use what you saw there.

    best_model_path =  os.path.join(cfg['save_model_folder'], 'best_model.pth')


    gc.collect()

    # Create objects for the dataset class
    train_data = AudioDataset(root=cfg['data_folder'],
                              phonemes=PHONEMES,
                              partition="train-clean-100")

    val_data = AudioDataset(root=cfg['data_folder'],
                            phonemes=PHONEMES,
                            partition="dev-clean")

    test_data = AudioDatasetTest(root=cfg['data_folder'],
                                 partition="test-clean")


    # Do NOT forget to pass in the collate function as parameter while creating the dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=4,
        batch_size=cfg['train']['batch_size'],
        pin_memory=True,
        shuffle=True,
        collate_fn=train_data.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=2,
        batch_size=cfg['train']['batch_size'],
        pin_memory=True,
        shuffle=False,
        collate_fn=val_data.collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=2,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=test_data.collate_fn
    )

    print("Batch size: ", cfg['train']['batch_size'])
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))


    torch.cuda.empty_cache()
    model = ASRModel(
        cfg=cfg,
        input_size=cfg['model']['input_size'],
        embed_size=cfg['model']['embed_size'],
        output_size=len(PHONEMES)
    ).to(device)
    print(model)

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean',
                                 zero_infinity=False)  # Define CTC loss as the criterion. How would the losses be reduced?



    # Declare the decoder. Use the CTC Beam Decoder to decode phonemes
    # CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
    decoder = CTCBeamDecoder(LABELS, beam_width=cfg['decode']['beam_width'], log_probs_input=True)


    # optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']))  # What goes in here?
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['lr']))

    # optimizer = optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=cfg['train']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg['scheduler']['patience'])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler']['T_max'], eta_min=1e-9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Mixed Precision, if you need it
    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(0, cfg['train']['epochs']):

        print("\nEpoch: {}/{}".format(epoch+1, cfg['train']['epochs']))

        curr_lr =  optimizer.param_groups[0]['lr']

        train_loss = train_model(model, train_loader, criterion, optimizer, device, scaler)
        valid_loss, valid_dist = validate_model(model, val_loader, decoder, LABELS, criterion, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(valid_dist)

        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
        print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))


        wandb.log({
            'train_loss': train_loss,
            'valid_dist': valid_dist,
            'valid_loss': valid_loss,
            'lr'        : curr_lr
        })

        if (epoch + 1) % cfg['train']['save_interval'] == 0 or (epoch + 1) == cfg['train']['epochs']:
            epoch_model_path = os.path.join(cfg['save_model_folder'], 'epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
            wandb.save(epoch_model_path)
            print("Saved epoch model")

        if valid_dist <= best_lev_dist:
            best_lev_dist = valid_dist
            save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
            wandb.save(best_model_path)
            print("Saved best model")
          # You may find it interesting to exlplore Wandb Artifcats to version your models
    run.finish()


    TEST_BEAM_WIDTH = 41

    test_decoder = CTCBeamDecoder(LABELS, beam_width=TEST_BEAM_WIDTH, log_probs_input=True)
    results = []

    model.eval()
    print("Testing")
    for data in tqdm(test_loader):
        x, lx = data
        x = x.to(device)

        with torch.no_grad():
            h, lh = model(x, lx)

        prediction_string = decode_prediction(h, lh, test_decoder, LABELS)

        results.extend(prediction_string)

        del x, lx, h, lh
        torch.cuda.empty_cache()

    df = pd.DataFrame({'index': range(len(results)), 'label': results})

    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()