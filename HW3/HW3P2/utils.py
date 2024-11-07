#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/2024 9:00 PM
# @Author  : Loading
import random
import numpy as np
import torch
import yaml
import Levenshtein
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_prediction(output, output_lens, decoder, PHONEME_MAP):
    # TODO: look at docs for CTC.decoder and find out what is returned here. Check the shape of output and expected shape in decode.
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(output, seq_lens = output_lens)  # lengths - list of lengths

    pred_strings = []
    # print(beam_results.shape)
    # print(beam_results)
    for i in range(output_lens.shape[0]):
        # TODO: Create the prediction from the output of decoder.decode. Don't forget to map it using PHONEMES_MAP.
        pred_strings.append(''.join([PHONEME_MAP[n] for n in beam_results[i][0][:out_seq_len[i][0]]]))
    # print(pred_strings)

    return pred_strings


def calculate_levenshtein(output, label, output_lens, label_lens, decoder,
                          PHONEME_MAP):  # y - sequence of integers

    dist = 0
    batch_size = label.shape[0]

    # Apply temperature scaling
    # output = output / 0.5

    pred_strings = decode_prediction(output, output_lens, decoder, PHONEME_MAP)
    # print(batch_size)
    for i in range(batch_size):
        pred_string = pred_strings[i]
        # print('pred',pred_string)
        label_string = ''.join([PHONEME_MAP[n] for n in label[i][:label_lens[i]]])
        # print('label',label_string)
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size

    return dist

def train_model(model, train_loader, criterion, optimizer, device='cpu', scaler=torch.cuda.amp.GradScaler()):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=True, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


def validate_model(model, val_loader, decoder, phoneme_map, criterion, device='cpu'):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=True, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += float(loss)
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist

def save_model(model, optimizer, scheduler, metric, epoch,path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
        path
    )

def load_model(path, model, metric= 'valid_dist', optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]


