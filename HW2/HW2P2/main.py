#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/21/2024 10:41 PM
# @Author  : Loading
import time

import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import os
import gc

from torchsummaryX import summary

from dataset.ImagePairDataset import ImagePairDataset
from dataset.TestImagePairDataset import TestImagePairDataset
from model.model import CNNNetwork
from test import test_epoch_ver
from train import train_epoch
from utils import save_model
from val import valid_epoch_cls
from ver import valid_epoch_ver


def create_dataloader(cfg):
    data_dir = cfg['data_dir']

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'dev')

    # train transforms
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),  # Why are we resizing the Image?
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])

    # val transforms
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])

    # get datasets
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)

    train_dataloader = DataLoader(train_dataset,
                               batch_size=cfg["batch_size"],
                               shuffle=True,
                               pin_memory=True,
                               num_workers=8,
                               sampler=None)
    val_dataloader = DataLoader(val_dataset,
                             batch_size=cfg["batch_size"],
                             shuffle=False,
                             num_workers=4)

    # TODO: Add your validation pair txt file
    pair_dataset = ImagePairDataset(data_dir, csv_file=cfg['val_pairs_file'], transform=val_transforms)
    pair_dataloader = torch.utils.data.DataLoader(pair_dataset,
                                                  batch_size=cfg["batch_size"],
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4)

    # TODO: Add your validation pair txt file
    test_pair_dataset = TestImagePairDataset(data_dir, csv_file=cfg['test_pairs_file'], transform=val_transforms)
    test_pair_dataloader = torch.utils.data.DataLoader(test_pair_dataset,
                                                       batch_size=cfg["batch_size"],
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       num_workers=4)

    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", cfg['batch_size'])
    print("Train batches        : ", train_dataloader.__len__())
    print("Val batches          : ", val_dataloader.__len__())

    # r, c = [5, 5]
    # fig, ax = plt.subplots(r, c, figsize=(15, 15))
    #
    # k = 0
    # dtl = DataLoader(
    #     dataset=torchvision.datasets.ImageFolder(train_dir, transform=train_transforms),
    #     # dont wanna see the images with transforms
    #     batch_size=cfg['batch_size'],
    #     shuffle=True)
    #
    # for data in dtl:
    #     x, y = data
    #
    #     for i in range(r):
    #         for j in range(c):
    #             img = x[k].numpy().transpose(1, 2, 0)
    #             ax[i, j].imshow(img)
    #             ax[i, j].axis('off')
    #             k += 1
    #     break
    #
    # del dtl
    # fig.show()

    return train_dataloader, val_dataloader, pair_dataloader, test_pair_dataloader


if __name__ == '__main__':
    wandb_api_key = "46b9373c96fe8f8327255e7da8a4046da7ffeef6"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", DEVICE)

    config = {
        'batch_size': 64,
        'lr': 0.1,
        'epochs': 20,
        'weight_decay': 0.01,
        'data_dir': "./11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/cls_data",
        'data_ver_dir': "./11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/ver_data",
        'val_pairs_file': "./11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/val_pairs.txt",
        'test_pairs_file':'./11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/test_pairs.txt',
        'checkpoint_dir': "./checkpoint"
        # Include other parameters as needed.
    }

    # Create Data loader
    print("==Create Dataloaders for Image Recognition==")
    train_loader, val_loader, pair_loader, test_pair_loader = create_dataloader(config)

    # Initialize model
    print("==Create Model==")
    model = CNNNetwork().to(DEVICE)
    x = torch.randn(1, 3, 112, 112).to(DEVICE)
    summary(model, x)

    # Defining Loss function
    criterion = torch.nn.CrossEntropyLoss() # TODO: What loss do you need for a multi class classification problem and would label smoothing be beneficial here?

    # Defining Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) # TODO: Feel free to pick a optimizer

    # Defining Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])  # TODO: Use a good scheduler such as ReduceLRonPlateau, StepLR, MultistepLR, CosineAnnealing, etc.

    # Initialising mixed-precision training. # Good news. We've already implemented FP16 (Mixed precision training) for you
    # It is useful only in the case of compatible GPUs such as T4/V100
    scaler = torch.cuda.amp.GradScaler()

    wandb.login(key=wandb_api_key)
    # Create your wandb run
    run = wandb.init(
        name="early-submission",  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Allows reinitalizing runs when you re-run this cell
        # run_id = int(time.time()),### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw2p2-ablations",  ### Project should be created in your wandb account
        config=config  ### Wandb Config for your run
    )

    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    ## Experiments
    e = 0
    best_valid_cls_acc = 0.0
    eval_cls = True
    best_valid_ret_acc = 0.0
    for epoch in range(e, config['epochs']):
        # epoch
        print("\nEpoch {}/{}".format(epoch + 1, config['epochs']))

        # train
        train_cls_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, config)
        curr_lr = float(optimizer.param_groups[0]['lr'])
        print("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1, config['epochs'], train_cls_acc, train_loss, curr_lr))
        metrics = {
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
        }
        # classification validation
        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, DEVICE, config)
            print("Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics.update({
                'valid_cls_acc': valid_cls_acc,
                'valid_loss': valid_loss,
            })

        # retrieval validation
        valid_ret_acc = valid_epoch_ver(model, pair_loader, DEVICE, config)
        print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
        metrics.update({
            'valid_ret_acc': valid_ret_acc
        })

        # save model
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
        print("Saved epoch model")

        # save best model
        if eval_cls:
            if valid_cls_acc >= best_valid_cls_acc:
                best_valid_cls_acc = valid_cls_acc
                save_model(model, optimizer, scheduler, metrics, epoch,
                           os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
                wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
                print("Saved best classification model")

        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(model, optimizer, scheduler, metrics, epoch,
                       os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            print("Saved best retrieval model")

        # log to tracker
        if run is not None:
            run.log(metrics)

    scores = test_epoch_ver(model, test_pair_loader, config)
    with open("verification_early_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(scores)):
            f.write("{},{}\n".format(i, scores[i]))




