#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/21/2024 10:41 PM
# @Author  : Loading
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import os

from torchsummaryX import summary

from HW2.HW2P2.model.model import CNNNetwork


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

    return train_dataloader, val_dataloader


if __name__ == '__main__':

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", DEVICE)

    config = {
        'batch_size': 64,
        'lr': 0.1,
        'epochs': 20,
        'data_dir': "./11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/cls_data",
        'data_ver_dir': "./11785-hw-2-p-2-face-verification-fall-2024/11-785-f24-hw2p2-verification/ver_data",
        'checkpoint_dir': "./checkpoint"  # TODO
        # Include other parameters as needed.
    }

    # Create Data loader
    print("==Create Dataloaders for Image Recognition==")
    train_loader, val_loader = create_dataloader(config)

    # Initialize model
    print("==Create Model==")
    model = CNNNetwork().to(DEVICE)
    summary(model, (3, 112, 112))

    # Defining Loss function
    criterion =  # TODO: What loss do you need for a multi class classification problem and would label smoothing be beneficial here?

    # Defining Optimizer
    optimizer =  # TODO: Feel free to pick a optimizer

    # Defining Scheduler
    scheduler = None  # TODO: Use a good scheduler such as ReduceLRonPlateau, StepLR, MultistepLR, CosineAnnealing, etc.

    # Initialising mixed-precision training. # Good news. We've already implemented FP16 (Mixed precision training) for you
    # It is useful only in the case of compatible GPUs such as T4/V100
    scaler = torch.cuda.amp.GradScaler()




