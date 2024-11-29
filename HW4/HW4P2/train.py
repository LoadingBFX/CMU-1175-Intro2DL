import time

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
from utils.misc import get_optimizer, get_scheduler, save_attention_plot, save_model, load_checkpoint
from utils.mytokenizer import GTokenizer
from utils.train_val import validate_step, train_step, test_step

''' Imports for decoding and distance calculation. '''
import json
import warnings
import shutil
warnings.filterwarnings("ignore")

import torchaudio
if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    ##Config##
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    print("Config: ", config)

    ##Tokenizer##
    Tokenizer = GTokenizer(config['token_type'])

    ##Data##
    ###Datasets###
    start_time = time.time()
    print("*****Loading Datasets")
    # @NOTE: use the config file to specify PARTITION and CEPSTRAL
    train_dataset = SpeechDataset(
        partition=config['train_partition'],
        config=config,
        tokenizer=Tokenizer,
        isTrainPartition=True,
    )
    print(f"Train Dataset Loaded in {time.time() - start_time:.2f} seconds")

    val_dataset = SpeechDataset(
        partition=config['val_partition'],
        config=config,
        tokenizer=Tokenizer,
        isTrainPartition=False,
    )
    print(f"Val Dataset Loaded in {time.time() - start_time:.2f} seconds")


    test_dataset = SpeechDataset(
        partition=config['test_partition'],
        config=config,
        tokenizer=Tokenizer,
        isTrainPartition=False,
    )
    print(f"Test Dataset Loaded in {time.time() - start_time:.2f} seconds")


    ###DataLoaders###
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    print(f"Train Loader Loaded in {time.time() - start_time:.2f} seconds")

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    print(f"Val Loader Loaded in {time.time() - start_time:.2f} seconds")


    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )
    print(f"Test Loader Loaded in {time.time() - start_time:.2f} seconds")
    print(f"*****Data Loaded in {time.time() - start_time:.2f} seconds")


    if config['DEBUG']:
        print('')
        print("Paired Data Stats: ")
        print(f"No. of Train Feats   : {train_dataset.__len__()}")
        print(f"Batch Size           : {config['batch_size']}")
        print(f"Train Batches        : {train_loader.__len__()}")
        print(f"Val Batches          : {val_loader.__len__()}")
        # print(f"Test Batches         : {test_loader.__len__()}")
        print('')
        print("Checking the Shapes of the Data --\n")
        for batch in train_loader:
            x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch
            print(f"x_pad shape:\t\t{x_pad.shape}")
            print(f"x_len shape:\t\t{x_len.shape}")

            if y_shifted_pad is not None and y_golden_pad is not None and y_len is not None:
                print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
                print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
                print(f"y_len shape:\t\t{y_len.shape}\n")
                # convert one transcript to text
                transcript = train_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
                print(f"Transcript Shifted: {transcript}")
                transcript = train_dataset.tokenizer.decode(y_golden_pad[0].tolist())
                print(f"Transcript Golden: {transcript}")
            break
        print('')

        # UNCOMMENT if pretraining decoder as LM
        # print("Unpaired Data Stats: ")
        # print(f"No. of text          : {text_dataset.__len__()}")
        # print(f"Batch Size           : {config['batch_size']}")
        # print(f"Train Batches        : {text_loader.__len__()}")
        # print('')
        # print("Checking the Shapes of the Data --\n")
        # for batch in text_loader:
        #      y_shifted_pad, y_golden_pad, y_len, = batch
        #      print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
        #      print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
        #      print(f"y_len shape:\t\t{y_len.shape}\n")
        #
        #      # convert one transcript to text
        #      transcript = text_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
        #      print(f"Transcript Shifted: {transcript}")
        #      transcript = text_dataset.tokenizer.decode(y_golden_pad[0].tolist())
        #      print(f"Transcript Golden: {transcript}")
        #      break
        print('')
        plt.figure(figsize=(10, 6))
        plt.imshow(x_pad[0].numpy().T, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('Features')
        plt.title('Feature Representation')
        plt.show()

    print("*****Verifying Datasets")
    start_time = time.time()
    max_train_feat, max_train_transcript = verify_dataset(train_loader, config['train_partition'])
    max_val_feat, max_val_transcript = verify_dataset(val_loader, config['val_partition'])
    max_test_feat, max_test_transcript = verify_dataset(test_loader, config['test_partition'])
    # _, max_text_transcript               = verify_dataset(text_loader,  config['unpaired_text_partition'])
    #Maximum Feat Length in Dataset       : 3066 Maximum Transcript Length in Dataset : 399
    # Maximum Feat Length in Dataset       : 4081
    # Maximum Transcript Length in Dataset : 517
    # Maximum Feat Length in Dataset       : 4370
    # Maximum Transcript Length in Dataset : 0
    # Maximum Feat. Length in Entire Dataset      : 4370
    # Maximum Transcript Length in Entire Dataset : 517

    MAX_SPEECH_LEN = max(max_train_feat, max_val_feat, max_test_feat)
    MAX_TRANS_LEN = max(max_train_transcript, max_val_transcript)
    # MAX_TRANS_LEN = max(max_train_transcript, max_val_transcript,max_text_transcript)
    # MAX_TRANS_LEN = 525
    print(f"Maximum Feat. Length in Entire Dataset      : {MAX_SPEECH_LEN}")
    print(f"Maximum Transcript Length in Entire Dataset : {MAX_TRANS_LEN}")
    print(f"Dataset Verification Completed in {time.time() - start_time:.2f} seconds")
    print('')
    gc.collect()



    ##Model##
    model = Transformer(
        input_dim=x_pad.shape[-1],
        time_stride=config['time_stride'],
        feature_stride=config['feature_stride'],
        embed_dropout=config['embed_dropout'],
        d_model=config['d_model'],
        enc_num_layers=config['enc_num_layers'],
        enc_num_heads=config['enc_num_heads'],
        speech_max_len=MAX_SPEECH_LEN,
        enc_dropout=config['enc_dropout'],
        dec_num_layers=config['dec_num_layers'],
        dec_num_heads=config['dec_num_heads'],
        d_ff=config['d_ff'],
        dec_dropout=config['dec_dropout'],
        target_vocab_size=Tokenizer.VOCAB_SIZE,
        trans_max_len=MAX_TRANS_LEN
    )

    # model = torch.compile(model)
    summary(model.to(device), input_data=[x_pad.to(device), x_len.to(device), y_shifted_pad.to(device), y_len.to(device)])

    gc.collect()
    torch.cuda.empty_cache()

    loss_func = nn.CrossEntropyLoss(ignore_index=Tokenizer.PAD_TOKEN)
    ctc_loss_fn = None
    if config['use_ctc']:
        ctc_loss_fn = nn.CTCLoss(blank=Tokenizer.PAD_TOKEN)
    scaler = torch.cuda.amp.GradScaler()

    torch.cuda.empty_cache()
    gc.collect()

    # using WandB? resume training?
    USE_WANDB = config['use_wandb']
    RESUME_LOGGING = False



    ## Pretraining the model
    gc.collect()
    torch.cuda.empty_cache()
    # creating your WandB run
    e = 0
    best_loss = 100000.0
    best_perplexity = 100000.0
    best_dist = 10000
    best_cer = 10000
    RESUME_LOGGING = False
    run_name = "{}_{}_Transformer_ENC-{}-{}_DEC-{}-{}_{}_{}_{}_{}_token_{}".format(
        config["Name"],
        config['feat_type'],
        config["enc_num_layers"],
        config["enc_num_heads"],
        config["dec_num_layers"],
        config["dec_num_heads"],
        config["d_model"],
        config["d_ff"],
        config["optimizer"],
        config["scheduler"],
        config["token_type"],
    )
    expt_root = os.path.join(os.getcwd(), run_name)
    os.makedirs(expt_root, exist_ok=True)
    shutil.copy(os.path.join(os.getcwd(), 'config.yaml'), os.path.join(expt_root, 'config.yaml'))
    checkpoint_root = os.path.join(expt_root, 'checkpoints')
    text_root = os.path.join(expt_root, 'out_text')
    attn_img_root = os.path.join(expt_root, 'attention_imgs')
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(attn_img_root, exist_ok=True)
    os.makedirs(text_root, exist_ok=True)

    checkpoint_best_loss_model_filename = 'checkpoint-best-loss-modelfull.pth'
    checkpoint_last_epoch_filename = 'checkpoint-epochfull-'
    best_loss_model_path = os.path.join(checkpoint_root, checkpoint_best_loss_model_filename)
    best_cer_model_path = os.path.join(checkpoint_root, 'checkpoint-best-cer-modelfull.pth')

    if USE_WANDB:
        wandb.login(key="46b9373c96fe8f8327255e7da8a4046da7ffeef6", relogin=True)

    if config['pretrain']:

        expt_root = os.path.join(os.getcwd(), "pre_train_" + run_name)
        os.makedirs(expt_root, exist_ok=True)

        ### Create a local directory with all the checkpoints
        shutil.copy(os.path.join(os.getcwd(), 'config.yaml'), os.path.join(expt_root, 'config.yaml'))

        checkpoint_root = os.path.join(expt_root, 'checkpoints')
        text_root = os.path.join(expt_root, 'out_text')
        attn_img_root = os.path.join(expt_root, 'attention_imgs')
        os.makedirs(checkpoint_root, exist_ok=True)
        os.makedirs(attn_img_root, exist_ok=True)
        os.makedirs(text_root, exist_ok=True)
        pretrain_model_checkpoint_last_epoch_filename = 'pretrain_model_checkpoint-epochfull-'
        best_loss_pretrain_model_path = os.path.join(checkpoint_root, 'pre_checkpoint-best-loss-modelfull.pth')
        best_cer_pretrain_model_path = os.path.join(checkpoint_root, 'pre_checkpoint-best-cer-modelfull.pth')

        run = wandb.init(
            name="pre_train_" + run_name,
            ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit=True,  ### Allows reinitalizing runs when you re-run this cell
            project="HW4P2-Fall",  ### Project should be created in your wandb account
            config=config  ### Wandb Config for your run
        )

        # if USE_WANDB:
        #     wandb.watch(model, log="all")
        #
        # ### Save your model architecture as a string with str(model)
        # model_arch = str(model)
        # ### Save it in a txt file
        # model_path = os.path.join(expt_root, "model_arch.txt")
        # arch_file = open(model_path, "w")
        # file_write = arch_file.write(model_arch)
        # arch_file.close()

    #
    #     print("Pretrain Approach 2: Decoder Conditional LM w/ SpeechEmbeddings")
    #     optimizer = get_optimizer(model, config)
    #     assert optimizer is not None
    #     scheduler = get_scheduler(optimizer, config)
    #     assert scheduler is not None
    #
    #     # set your epochs for this approach
    #     epochs = config['pre_epochs']
    #     for epoch in range(e, epochs):
    #
    #         print("\nEpoch {}/{}".format(epoch + 1, epochs))
    #
    #         curr_lr = float(optimizer.param_groups[0]["lr"])
    #
    #         train_loss, train_perplexity, attention_weights = train_step(
    #             model,
    #             criterion=loss_func,
    #             ctc_loss=None,
    #             ctc_weight=0.,
    #             optimizer=optimizer,
    #             scheduler=scheduler,
    #             scaler=scaler,
    #             device=device,
    #             train_loader=train_loader,
    #             tokenizer=Tokenizer,
    #             mode='dec_cond_lm'
    #         )
    #
    #         print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
    #             epoch + 1, epochs, train_loss, train_perplexity, curr_lr))
    #
    #         levenshtein_distance, json_out, wer, cer = validate_step(
    #             model,
    #             val_loader=val_loader,
    #             tokenizer=Tokenizer,
    #             device=device,
    #             mode='dec_cond_lm',
    #             threshold=5
    #         )
    #
    #         fpath = os.path.join(text_root, f'dec_cond_lm_{epoch + 1}_out.json')
    #         with open(fpath, "w") as f:
    #             json.dump(json_out, f, indent=4)
    #
    #         print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    #         print("WER                  : {:.04f}".format(wer))
    #         print("CER                  : {:.04f}".format(cer))
    #
    #         attention_keys = list(attention_weights.keys())
    #         attention_weights_decoder_self = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    #         attention_weights_decoder_cross = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
    #
    #         if USE_WANDB:
    #             wandb.log({
    #                 "train_loss": train_loss,
    #                 "train_perplexity": train_perplexity,
    #                 "learning_rate": curr_lr,
    #                 "lev_dist": levenshtein_distance,
    #                 "WER": wer,
    #                 "CER": cer
    #             })
    #
    #         save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch, mode='dec_cond_lm')
    #         save_attention_plot(str(attn_img_root), attention_weights_decoder_self, epoch + 100, mode='dec_cond_lm')
    #         if config["scheduler"] == "ReduceLR":
    #             scheduler.step(levenshtein_distance)
    #         else:
    #             scheduler.step()
    #
    #         ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    #         epoch_model_path = os.path.join(checkpoint_root, (pretrain_model_checkpoint_last_epoch_filename + '.pth'))
    #         save_model(model, optimizer, scheduler, ('CER', cer), epoch, epoch_model_path)
    #         print("Saved last checkpoint model")
    #         if best_dist >= levenshtein_distance:
    #             best_loss = train_loss
    #             best_dist = levenshtein_distance
    #             save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_loss_pretrain_model_path)
    #             print("Saved best dist model")
    #         if best_cer >= cer:
    #             best_loss = train_loss
    #             best_cer = cer
    #             save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_cer_pretrain_model_path)
    #             print("Saved best CER model")
    #     ### Finish your wandb run
    #     if USE_WANDB:
    #         run.finish()

    ## Training the model
    epochs = config['epochs']
    # encoder_pretrain_epochs = 5

    # creating your WandB run



    if config['pretrain']:
        print("Loading pre model: ", best_cer_pretrain_model_path)
        # Load the model, optimizer, and scheduler states
        model, optimizer, scheduler = load_checkpoint(
            checkpoint_path=best_cer_pretrain_model_path,
            model=model,
            embedding_load=True,
            encoder_load=True,
            decoder_load=True,
        )


    loss_func = nn.CrossEntropyLoss(ignore_index=Tokenizer.PAD_TOKEN, label_smoothing=0.1)
    ctc_loss_fn = None
    if config['use_ctc']:
        ctc_loss_fn = nn.CTCLoss(blank=Tokenizer.PAD_TOKEN)
    scaler = torch.cuda.amp.GradScaler()

    optimizer = get_optimizer(model, config)
    assert optimizer is not None

    scheduler = get_scheduler(optimizer, config)
    assert scheduler is not None


    if USE_WANDB:
        run = wandb.init(
            name=run_name,
            ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit=True,  ### Allows reinitalizing runs when you re-run this cell
            project="HW4P2-Fall",  ### Project should be created in your wandb account
            config=config  ### Wandb Config for your run
        )
        wandb.watch(model, log="all")

    ### Save your model architecture as a string with str(model)
    model_arch = str(model)
    ### Save it in a txt file
    model_path = os.path.join(expt_root, "model_arch.txt")
    arch_file = open(model_path, "w")
    file_write = arch_file.write(model_arch)
    arch_file.close()


    for epoch in range(e, epochs):

        print("\nEpoch {}/{}".format(epoch + 1, epochs))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        # if epoch < encoder_pretrain_epochs:
        #     print("Training encoder only.")
        #     set_requires_grad(model, "embedding", False)
        #     set_requires_grad(model, "decoder", False)
        #     set_requires_grad(model, "encoder", True)
        # else:
        #     print("Training full Transformer.")
        #     set_requires_grad(model, "embedding", True)
        #     set_requires_grad(model, "decoder", True)
        #     set_requires_grad(model, "encoder", True)

        train_loss, train_perplexity, attention_weights = train_step(
            model,
            criterion=loss_func,
            ctc_loss=ctc_loss_fn,
            ctc_weight=config['ctc_weight'],
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            train_loader=train_loader,
            tokenizer=Tokenizer,
            mode='full'
        )

        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr))

        levenshtein_distance, json_out, wer, cer = validate_step(
            model,
            val_loader=val_loader,
            tokenizer=Tokenizer,
            device=device,
            mode='full',
            threshold=5
        )

        fpath = os.path.join(text_root, f'full_{epoch + 1}_out.json')
        with open(fpath, "w") as f:
            json.dump(json_out, f, indent=4)

        print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
        print("WER                  : {:.04f}".format(wer))
        print("CER                  : {:.04f}".format(cer))

        attention_keys = list(attention_weights.keys())
        attention_weights_decoder_self = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
        attention_weights_decoder_cross = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

        if USE_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "learning_rate": curr_lr,
                "lev_dist": levenshtein_distance,
                "WER": wer,
                "CER": cer
            })

        save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch, mode='full')
        save_attention_plot(str(attn_img_root), attention_weights_decoder_self, epoch + 100, mode='full')
        if config["scheduler"] == "ReduceLR":
            scheduler.step(levenshtein_distance)
        else:
            scheduler.step()

        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '-2' + '.pth'))
        save_model(model, optimizer, scheduler, ('CER', cer), epoch, epoch_model_path)

        if best_dist >= levenshtein_distance:
            best_loss = train_loss
            best_dist = levenshtein_distance
            save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_loss_model_path)
            print("Saved best dist model")
        if best_cer >= cer:
            best_loss = train_loss
            best_cer = cer
            save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_cer_model_path)
            print("Saved best CER model")
    print("best_loss_model_path: ", best_loss_model_path)
    print("best_cer: ", best_cer,"best_loss: ", best_loss, "best_dist: ", best_dist, "best_perplexity: ", best_perplexity)
    ### Finish your wandb run
    if USE_WANDB:
        run.finish()

    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode='full',
        threshold=None
    )

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    fpath = os.path.join(os.getcwd(), f'final_out_{run_name}.json')
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    predictions = test_step(
        model,
        test_loader=test_loader,
        tokenizer=Tokenizer,
        device=device
    )

    import csv

    # Sample list

    # Specify the CSV file path
    csv_file_path = "submission.csv"

    # Write the list to the CSV with index as the first column
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Index", "Labels"])
        # Write list items with index
        for idx, item in enumerate(predictions):
            writer.writerow([idx, item])

    print(f"CSV file saved to {csv_file_path}")