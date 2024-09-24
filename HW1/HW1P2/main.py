#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Loading

import os
import time
from sched import scheduler
import torch


import wandb

from HW1.HW1P2.models.model import Network
from HW1.HW1P2.src.eval import evaluate
from HW1.HW1P2.src.test import test
from HW1.HW1P2.src.train import train
from datasets.data_loader import AudioDataset
from datasets.test_data_loader import  AudioTestDataset
from torchsummaryX import summary
import gc


def setup_wandb(api_key, project_name, configs):
    """
    Set up Weights and Biases (wandb) for tracking experiments.

    Parameters:
        api_key (str): Your WandB API key for login.
        project_name (str): Name of the project in WandB.
        configs (dict): Configuration parameters for the run.

    Returns:
        wandb.run: The initialized wandb run object.
    """
    # Login to wandb
    wandb.login(key=api_key)

    # Initialize a new wandb run
    run = wandb.init(
        name=f"{configs['model_name']}_{configs['epochs']}_{configs['batch_size']}_{configs['init_lr']}",  # Recommend providing meaningful names
        reinit=True,  # Allows reinitializing runs
        project=project_name,  # Project name in wandb account
        config=configs  # Configuration for the run
    )
    ### Save your model architecture as a string with str(model)
    model_arch = str(model)

    ### Save it in a txt file
    arch_file   = open("../model_arch.txt", "w")
    arch_file.write(model_arch)
    arch_file.close()

    # Save the model architecture
    wandb.save('model_arch.txt')  # Log the model architecture in wandb

    print("WandB setup complete. Run ID:", run.id)
    return run

def audio_dataloader(dataset_path, phonemes, context, batch_size, train_data_size=-1, val_data_size=-1):
    """
    Create audio datasets and data loaders for training, validation, and testing.

    Parameters:
        dataset_path (str): The path to the dataset.
        phonemes (list): The list of phonemes.
        context (int): The context size.
        batch_size (int): The batch size for the data loaders.
        train_data_size (int): The size of the training data. Default is -1 (use all).
        val_data_size (int): The size of the validation data. Default is -1 (use all).

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    print("\n#### Loading dataset...")
    # Create datasets
    train_data = AudioDataset(dataset_path, train_data_size, phonemes, context, 'train-clean-100')
    # tricky
    # train_data = AudioDataset(dataset_path, train_data_size, phonemes, context, 'dev-clean')
    val_data = AudioDataset(dataset_path, val_data_size, phonemes, context, 'dev-clean')
    test_data = AudioTestDataset(dataset_path, context, 'test-clean')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

    print("Train datasets samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Validation datasets samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test datasets samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, model_save_path, resume=False):
    """
    Train the model over a specified number of epochs and save the best model.
    Optionally, resume training from a saved checkpoint.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).
        epochs (int): Number of epochs to train.
        model_save_path (str): Path to save the best model.
        resume (bool): Whether to resume training from a checkpoint.
    """
    best_val_loss = float('inf')
    best_val_acc = 0.8688
    start_epoch = 0

    # Resume training from a checkpoint
    if resume and os.path.isfile(model_save_path):
        print(f"Resuming training from {model_save_path}...")
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        epochs += start_epoch
        best_val_loss = checkpoint['val_loss']  # Restore best validation loss
        print(start_epoch, best_val_loss)

        for param_group in optimizer.param_groups:
            param_group['lr'] = config['init_lr']
            print("lr：", param_group['lr'])
        #     param_group['weight_decay'] = config['weight_decay']
        #     print("New weight_decay:", param_group['weight_decay'])

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"\tVal Acc: {val_acc * 100:.04f}%\tVal Loss: {val_loss:.04f}")

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        curr_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Log training and validation metrics
        print(f"\tTrain Acc: {train_acc * 100:.04f}%\tTrain Loss: {train_loss:.04f}\tLearning Rate: {curr_lr:.07f}")
        print(f"\tVal Acc: {val_acc * 100:.04f}%\tVal Loss: {val_loss:.04f}")

        # Save the best model
        if val_loss < best_val_loss or val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
            }, model_save_path)
            print(f"New best model saved to {model_save_path} with validation loss {val_loss:.04f}")

        # Log metrics to WandB
        wandb.log({
            'train_acc': train_acc * 100,
            'train_loss': train_loss,
            'val_acc': val_acc * 100,
            'val_loss': val_loss,
            'lr': curr_lr
        })

        # scheduler.step(val_loss)
        scheduler.step()

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        print("\nTime cost: ", time.time() - start_time)



if __name__ == '__main__':

    """# Configuration Setting """
    ## wandb api
    wandb_api_key = "46b9373c96fe8f8327255e7da8a4046da7ffeef6"
    ## Paths
    DATASET_PATH = "./data/11785-f24-hw1p2"
    MODEL_SAVE_DIR = "./checkpoints"




    ### PHONEME LIST
    PHONEMES = [
        '[SIL]', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
        'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
        'V', 'W', 'Y', 'Z', 'ZH', '[SOS]', '[EOS]']

    ## Model Parameters
    TRAIN_DATASIZE = -1
    VAL_DATASIZE = -1
    time_stamp = int(time.time())
    config = {
        'model_name'    : time_stamp,
        'epochs'        : 200,
        'batch_size'    : 4096,
        'context'       : 30,
        'init_lr'       : 0.001,
        'weight_decay' : 0.01,
    }


    resume_training = True  # Set to True to resume training
    resume_from = "best_model_1726848272_test.pth"

    if resume_training:
        BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, resume_from)
    else:
        BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"best_model_{time_stamp}_test.pth")
    print(f"\n[IMPORTANT] - resume flag is : {resume_training}\n")

    # Ensure Save Directory Exists
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created directory: {MODEL_SAVE_DIR}")

    if resume_training and not os.path.isfile(BEST_MODEL_PATH):

        raise Exception(f"No best model found! - {BEST_MODEL_PATH}")



    """# Load Data """
    train_loader, val_loader, test_loader = audio_dataloader(
        DATASET_PATH,
        PHONEMES,
        config['context'],
        config['batch_size'],
        TRAIN_DATASIZE,
        VAL_DATASIZE
    )

    """# Create Model """

    print("\n##### Creating Model #####\n")
    # For debug: Get a batch of data
    frames, phoneme = next(iter(train_loader))
    print(f"*_ Frames shape: {frames.shape}, Phoneme shape: {phoneme.shape}\n")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"*_ Using device: {device}\n")

    # Calculate input size
    INPUT_SIZE = (2 * config['context'] + 1) * 28
    print(f"*_ Input size calculated: {INPUT_SIZE}\n")

    # Initialize model
    model = Network(INPUT_SIZE, len(PHONEMES)).to(device)
    print("*_ Model initialized.\n")

    # Display model summary
    print("##### Model Summary #####")
    summary(model, frames.to(device))



    # # 先前计算得到的标准化权重
    # normalized_weights = {
    #     '[SIL]': 0.0010195278239303483,
    #     'CH': 0.027262550061861155,
    #     'AE': 0.0067245591181908934,
    #     'P': 0.010025373566600252,
    #     'T': 0.00349931752405353,
    #     'ER': 0.00645628884112817,
    #     'W': 0.008772433835102017,
    #     'AH': 0.0028058526550210627,
    #     'N': 0.0035802071571017697,
    #     'M': 0.007422871221713976,
    #     'IH': 0.004521105137211673,
    #     'S': 0.0033461244887151615,
    #     'Z': 0.00633730993488654,
    #     'R': 0.005619078784083,
    #     'EY': 0.009346859654934477,
    #     'L': 0.005358756729683189,
    #     'D': 0.00547898976560746,
    #     'AY': 0.00686518673889761,
    #     'V': 0.012964230101785348,
    #     'JH': 0.03732634839282497,
    #     'EH': 0.00716064845345569,
    #     'DH': 0.009344122854929575,
    #     'IY': 0.0050510569932791915,
    #     'OW': 0.010879836040777692,
    #     'AW': 0.018807655441894396,
    #     'UW': 0.012031580302845556,
    #     'HH': 0.009420916838887785,
    #     'AA': 0.01104404848581983,
    #     'F': 0.009204785079653907,
    #     'B': 0.014295731797740048,
    #     'UH': 0.055647403099402935,
    #     'K': 0.007087897274718058,
    #     'AO': 0.011985529883666404,
    #     'TH': 0.03786628746182458,
    #     'Y': 0.032112626094812205,
    #     'NG': 0.017824124416237416,
    #     'G': 0.025944402196734426,
    #     'SH': 0.01832560825674828,
    #     'OY': 0.09279744190122104,
    #     'ZH': 0.4184353255920183
    # }
    # #
    # # # 创建权重张量，并根据标准化权重更新
    # weights = torch.tensor([normalized_weights.get(phoneme, 1.0) for phoneme in PHONEMES]).to(device)
    #
    # # Defining Loss function.
    # criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # # 为每个类别设置默认权重为1
    # weights = torch.ones(len(PHONEMES)).to(device)
    # # 比如想降低SIL的权重为0.5
    # weights[0] = 0.1

    # Defining Loss function.
    # criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion = torch.nn.CrossEntropyLoss()

    # Defining Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['init_lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5,  verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])


    """# wandb setup """
    torch.cuda.empty_cache()
    gc.collect()

    run = setup_wandb(wandb_api_key, "hw1p2_test", config)

    torch.cuda.empty_cache()
    gc.collect()
    wandb.watch(model, log="all")


    print("== Config ==")
    print("Epochs         : ", config['epochs'])
    print("Batch size     : ", config['batch_size'])
    print("Context        : ", config['context'])
    print("init_lr        : ", config['init_lr'])
    print("Input size     : ", (2 * config['context'] + 1) * 28)
    print("Output symbols : ", len(PHONEMES))

    print("##### Model Summary #####")
    summary(model, frames.to(device))
    print("criterion: ", criterion.weight)
    print("optimizer: ", optimizer)
    print('scheduler: ', scheduler)

    """#
    Training Process
    """
    train_model(model, train_loader, val_loader,
                optimizer, criterion, device, config['epochs'],
                BEST_MODEL_PATH,
                resume=resume_training)


    """#
    Testing Process
    """
    predictions = test(model, test_loader, device)

    ### Create CSV file with predictions
    with open("./submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(predictions)):
            predicted_phoneme = PHONEMES[predictions[i]]
            f.write("{},{}\n".format(i, predicted_phoneme))
        print("./submission.csv saved")

    torch.cuda.empty_cache()
    gc.collect()
    ### Finish your wandb run
    # run.finish()

### Submit to kaggle competition using kaggle API (Uncomment below to use)
# !kaggle competitions submit -c 11785-hw1p2-f24 -f ./submission.csv -m "Test Submission"