#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/18/2024 5:27 PM
# @Author  : Loading
from typing import Tuple, Dict, Any, Literal
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as tat
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader
import gc
import os
from transformers import AutoTokenizer
import yaml
import math
from typing import Literal, List, Optional, Any, Dict, Tuple
import random
import zipfile
import datetime
from torchinfo import summary
import glob
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fftpack import dct
import seaborn as sns
import matplotlib.pyplot as plt

from datasets.SpeechDataset import SpeechDataset
from datasets.TextDataset import TextDataset
from datasets.Verify import verify_dataset
from models.myTransformer import Transformer
from utils.metrics import calculateBatchMetrics
from utils.mytokenizer import GTokenizer

''' Imports for decoding and distance calculation. '''
import json
import warnings
import shutil
warnings.filterwarnings("ignore")

import torchaudio

def train_step(
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        ctc_loss: nn.CTCLoss,
        ctc_weight: float,
        optimizer,
        scheduler,
        scaler,
        device: str,
        train_loader: DataLoader,
        tokenizer: Any,
        mode: Literal['full', 'dec_cond_lm', 'dec_lm']
) -> Tuple[float, float, torch.Tensor]:
    """
    Trains a model for one epoch based on the specified training mode.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.CrossEntropyLoss): The loss function for cross-entropy.
        ctc_loss (nn.CTCLoss): The loss function for CTC.
        ctc_weight (float): Weight of the CTC loss in the total loss calculation.
        optimizer (Optimizer): The optimizer to update model parameters.
        scheduler (_LRScheduler): The learning rate scheduler.
        scaler (GradScaler): For mixed-precision training.
        device (str): The device to run training on, e.g., 'cuda' or 'cpu'.
        train_loader (DataLoader): The training data loader.
        tokenizer (Any): Tokenizer with PAD_TOKEN attribute.
        mode (Literal): Specifies the training objective.

    Returns:
        Tuple[float, float, torch.Tensor]: The average training loss, perplexity, and attention weights.
    """
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=True, position=0, desc=f"[Train mode: {mode}]")

    running_loss = 0.0
    running_perplexity = 0.0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Separate inputs and targets based on the mode
        if mode != 'dec_lm':
            inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths = batch
            inputs = inputs.to(device)
        else:
            inputs, inputs_lengths = None, None
            targets_shifted, targets_golden, targets_lengths = batch

        targets_shifted = targets_shifted.to(device)
        targets_golden = targets_golden.to(device)

        # Forward pass with mixed-precision
        with torch.autocast(device_type=device, dtype=torch.float16):
            raw_predictions, attention_weights, ctc_out = model(inputs, inputs_lengths, targets_shifted,
                                                                targets_lengths, mode=mode)
            padding_mask = torch.logical_not(torch.eq(targets_shifted, tokenizer.PAD_TOKEN))

            if torch.isnan(raw_predictions).any() or torch.isinf(raw_predictions).any():
                print("NaN or Inf detected in the output tensor.")

            # Calculate cross-entropy loss
            ce_loss = criterion(raw_predictions.transpose(1, 2), targets_golden) * padding_mask
            loss = ce_loss.sum() / padding_mask.sum()

            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN or Inf, skipping backward pass.")
                print(torch.isnan(loss))
                print(torch.isinf(loss))
                continue  # Skip this iteration if the loss is invalid

            # Optionally optimize a weighted sum of ce and ctc_loss from the encoder outputs
            # Only available during full transformer training, a ctc_loss must be passed in
            if mode == 'full' and ctc_loss and ctc_out is not None:
                inputs_lengths = torch.ceil(inputs_lengths.float() / model.embedding.time_downsampling_factor).int()
                inputs_lengths = inputs_lengths.clamp(max=ctc_out.size(0))
                loss += ctc_weight * ctc_loss(ctc_out, targets_golden, inputs_lengths, targets_lengths)

        # --- Check for NaN/Inf in gradients ---
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN or Inf gradients found in {name}")
                    break  # Or continue with debugging steps

        # Backward pass and optimization with mixed-precision
        scaler.scale(loss).backward()

        # Clip gradients (if needed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss and perplexity for monitoring
        running_loss += float(loss.item())
        perplexity = torch.exp(loss)
        running_perplexity += perplexity.item()

        # Update the progress bar
        batch_bar.set_postfix(
            loss=f"{running_loss / (i + 1):.4f}",
            perplexity=f"{running_perplexity / (i + 1):.4f}"
        )
        batch_bar.update()

        # Clean up to save memory
        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

    # Compute average loss and perplexity
    avg_loss = running_loss / len(train_loader)
    avg_perplexity = running_perplexity / len(train_loader)
    batch_bar.close()

    return avg_loss, avg_perplexity, attention_weights


def validate_step(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        tokenizer: Any,
        device: str,
        mode: Literal['full', 'dec_cond_lm', 'dec_lm'],
        threshold: int = 5
) -> Tuple[float, Dict[int, Dict[str, str]], float, float]:
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        tokenizer (Any): Tokenizer with a method to handle special tokens.
        device (str): The device to run validation on, e.g., 'cuda' or 'cpu'.
        mode (Literal): Specifies the validation objective.
        threshold (int, optional): Max number of batches to validate on (for early stopping).

    Returns:
        Tuple[float, Dict[int, Dict[str, str]], float, float]: The average distance, JSON output with inputs/outputs,
                                                               average WER, and average CER.
    """
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=True, position=0, desc="Val")

    running_distance = 0.0
    running_wer = 0.0
    running_cer = 0.0
    json_output = {}

    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            # Separate inputs and targets based on the mode
            if mode != 'dec_lm':
                inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths = batch
                inputs = inputs.to(device)
            else:
                inputs, inputs_lengths = None, None
                _, targets_shifted, targets_golden, _, targets_lengths = batch

            # targets_shifted = targets_shifted.to(device)
            targets_golden = targets_golden.to(device)

            # Perform recognition and calculate metrics
            greedy_predictions = model.recognize(inputs, inputs_lengths, tokenizer=tokenizer, mode=mode)
            dist, wer, cer, y_string, pred_string = calculateBatchMetrics(greedy_predictions, targets_golden,
                                                                          targets_lengths, tokenizer)

            # Accumulate metrics
            running_distance += dist
            running_wer += wer
            running_cer += cer
            json_output[i] = {"Input": y_string, "Output": pred_string}

            # Update progress bar
            batch_bar.set_postfix(
                running_distance=f"{running_distance / (i + 1):.4f}",
                WER=f"{running_wer / (i + 1):.4f}",
                CER=f"{running_cer / (i + 1):.4f}"
            )
            batch_bar.update()

            # Early stopping for thresholded validation
            if threshold and i == threshold:
                break

            del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
            torch.cuda.empty_cache()

    # Compute averages
    num_batches = threshold + 1 if threshold else len(val_loader)
    avg_distance = running_distance / num_batches
    avg_wer = running_wer / num_batches
    avg_cer = running_cer / num_batches
    batch_bar.close()

    return avg_distance, json_output, avg_wer, avg_cer


def test_step(model, test_loader, tokenizer, device):
    model.eval()
    # progress bar
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=True, position=0, desc="Test", ncols=5)

    predictions = []

    ## Iterate through batches
    for i, batch in enumerate(test_loader):

        inputs, _, _, inputs_lengths, _ = batch
        inputs = inputs.to(device)

        with torch.inference_mode():  # TODO():
            greedy_predictions = model.recognize(inputs, inputs_lengths, tokenizer=tokenizer, mode='full')

        # @NOTE: modify the print_example to print more or less validation examples
        batch_size, _ = greedy_predictions.shape
        batch_pred = []

        ## TODO decode each sequence in the batch
        for batch_idx in range(batch_size):
            # trim predictons upto the EOS_TOKEN
            # Convert tensor predictions into tokens using the tokenizer

            pad_indices = torch.where(greedy_predictions[batch_idx] == tokenizer.EOS_TOKEN)[0]
            lowest_pad_idx = pad_indices.min().item() if pad_indices.numel() > 0 else len(greedy_predictions[batch_idx])
            pred_trimmed = greedy_predictions[batch_idx, :lowest_pad_idx]

            # Decode the predicted token IDs to string
            pred_string = tokenizer.decode(pred_trimmed)

            batch_pred.append(pred_string)

        predictions.extend(batch_pred)

        batch_bar.update()

        del inputs, inputs_lengths
        torch.cuda.empty_cache()

    return predictions