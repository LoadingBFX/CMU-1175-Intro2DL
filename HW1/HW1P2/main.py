#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Loading

import os
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
        name=f"{configs['epochs']}_{configs['batch_size']}_{configs['init_lr']}",  # Recommend providing meaningful names
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"\tVal Acc: {val_acc * 100:.04f}%\tVal Loss: {val_loss:.04f}")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        curr_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Log training and validation metrics
        print(f"\tTrain Acc: {train_acc * 100:.04f}%\tTrain Loss: {train_loss:.04f}\tLearning Rate: {curr_lr:.07f}")
        print(f"\tVal Acc: {val_acc * 100:.04f}%\tVal Loss: {val_loss:.04f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
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

        scheduler.step(val_loss)

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == '__main__':

    """# Configuration Setting """
    ## wandb api
    wandb_api_key = "46b9373c96fe8f8327255e7da8a4046da7ffeef6"
    ## Paths
    DATASET_PATH = "./data/11785-f24-hw1p2"
    MODEL_SAVE_DIR = "./checkpoints"
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
    resume_training = True  # Set to True to resume training
    print(f"\n[IMPORTANT] - resume flag is : {resume_training}\n")



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
    config = {
        'epochs'        : 10,
        'batch_size'    : 8192,
        'context'       : 35,
        'init_lr'       : 1e-4,
    }
    print("== Config ==")
    print("Epochs         : ", config['epochs'])
    print("Batch size     : ", config['batch_size'])
    print("Context        : ", config['context'])
    print("init_lr        : ", config['init_lr'])
    print("Input size     : ", (2 * config['context'] + 1) * 28)
    print("Output symbols : ", len(PHONEMES))

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

    # Defining Loss function.
    criterion = torch.nn.CrossEntropyLoss()

    # Defining Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['init_lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=0.01)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    """# wandb setup """
    torch.cuda.empty_cache()
    gc.collect()

    run = setup_wandb(wandb_api_key, "hw1p2", config)

    torch.cuda.empty_cache()
    gc.collect()
    wandb.watch(model, log="all")

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