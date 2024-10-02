#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/21/2024 10:41 PM
# @Author  : Loading
import time

import torch
import torchvision
from tqdm import tqdm

import wandb
from torch.utils.data import DataLoader
import os
import gc

from torchsummaryX import summary

from dataset.ImagePairDataset import ImagePairDataset
from dataset.TestImagePairDataset import TestImagePairDataset

from dataset.TripletImageDataset import TripletImageDataset
from loss.Triplet import TripletLoss
from model.model import CNNNetwork
from model.senet import SENetwork
from test import test_epoch_ver
from train import train_epoch, train_epoch_triplet
from utils import save_model, load_model
from val import valid_epoch_cls
from ver import valid_epoch_ver
import yaml

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def initialize_model_optimizer_scheduler(model, config):
    # Defining Optimizer
    optimizer = None
    # Defining Scheduler
    scheduler = None

    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),  # Optional for SGD
            weight_decay=optimizer_config['weight_decay']
        )

    # 根据 config 选择调度器
    scheduler_config = config['scheduler']
    if scheduler_config['type'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config['T_0'],
            T_mult=scheduler_config['T_mult'],
            eta_min=scheduler_config['eta_min']
        )
    elif scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
        )

    print(f"optimizer: {optimizer} \n scheduler: {scheduler}")

    return optimizer, scheduler


def create_dataloader(cfg):
    data_dir = cfg['data_dir']

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'dev')
    # https: // pytorch.org / vision / stable / transforms.html

    # train transforms
    # train_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(112),  # Why are we resizing the Image?
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                      std=[0.5, 0.5, 0.5])])

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.1),
        torchvision.transforms.RandomRotation(18),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]),

        torchvision.transforms.RandomErasing(p=0.3, scale=(0.05, 0.1)),
    ])

    # train_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(0.5),  # Keep this as it is
    #     torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1),  # Increase brightness
    #     torchvision.transforms.RandomRotation(10),  # Reduce rotation to prevent excessive face angle changes
    #     torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    #     # Slightly reduce translation and scaling
    #     torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.2),  # Reduce distortion scale
    #
    #     torchvision.transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
    #     # Reduce the blur kernel size and sigma for less aggressive blurring
    #     torchvision.transforms.Resize(112),  # Resize is necessary to ensure uniform input size for the network
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                      std=[0.5, 0.5, 0.5]),
    #
    #     torchvision.transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Keep erasing as is
    # ])

    # val transforms
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])

    # get datasets
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)

    # 使用自定义数据集
    # train_dataset = TripletImageDataset(train_dataset, transform=train_transforms)

    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)

    train_dataloader = DataLoader(train_dataset,
                               batch_size=cfg["batch_size"],
                               shuffle=True,
                               pin_memory=True,
                               num_workers=16,
                               sampler=None)
    val_dataloader = DataLoader(val_dataset,
                             batch_size=cfg["batch_size"],
                             shuffle=False,
                             num_workers=4)


    pair_dataset = ImagePairDataset(cfg['data_ver_dir'], csv_file=cfg['val_pairs_file'], transform=val_transforms)
    pair_dataloader = torch.utils.data.DataLoader(pair_dataset,
                                                  batch_size=cfg["batch_size"],
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4)


    test_pair_dataset = TestImagePairDataset(cfg['data_ver_dir'], csv_file=cfg['test_pairs_file'], transform=val_transforms)
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

def train_and_val(model, optimizer, scheduler, criterion, scaler, train_loader, val_loader, pair_loader, config, DEVICE, run):
    e = config['e']
    best_valid_cls_acc = 0.0
    eval_cls = True
    best_valid_ret_acc = 0.0
    for epoch in range(e, config['epochs']):
        # # epoch
        tqdm.write("\nEpoch {}/{}".format(epoch + 1, config['epochs']))

        # train
        train_cls_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE,
                                                config)

        # train_cls_acc, train_loss = train_epoch_triplet(model, train_loader, criterion[0], criterion[1],optimizer, scheduler, scaler, DEVICE,
        #                                         config)

        curr_lr = float(optimizer.param_groups[0]['lr'])
        tqdm.write("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1, config['epochs'], train_cls_acc, train_loss, curr_lr))
        metrics = {
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
            'learning_rate': curr_lr
        }
        # classification validation
        if eval_cls:
            # valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, criterion[0], DEVICE, config)
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, criterion, DEVICE, config)
            tqdm.write("\nVal Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics.update({
                'valid_cls_acc': valid_cls_acc,
                'valid_loss': valid_loss,
            })

        # retrieval validation
        valid_ret_acc, valid_ret_eer = valid_epoch_ver(model, pair_loader, DEVICE, config)
        tqdm.write("\nVal Ret. Acc {:.04f}% \t Val Ret. EER {:.04f}%".format(valid_ret_acc, valid_ret_eer))
        metrics.update({
            'valid_ret_acc': valid_ret_acc,
            'valid_ret_eer': valid_ret_eer,
        })

        # save model
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
        tqdm.write("Saved last epoch model")

        # save best model
        if eval_cls:
            if valid_cls_acc >= best_valid_cls_acc:
                best_valid_cls_acc = valid_cls_acc
                save_model(model, optimizer, scheduler, metrics, epoch,
                           os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
                wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
                tqdm.write(f"\n######### Saved best classification model best_valid_cls_acc: {best_valid_cls_acc}#########")

        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(model, optimizer, scheduler, metrics, epoch,
                       os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            tqdm.write(f"\n****** Saved best retrieval model best_valid_ret_acc: {best_valid_ret_acc} ******")

        # log to tracker
        if run is not None:
            run.log(metrics)


def setup_wandb(model, cfg):
    wandb_config = cfg['wandb']
    wandb.login(key=wandb_config['wandb_api_key'])
    resume = cfg['resume']['resume']
    _id = str(int(time.time()))
    # if resume:
    #     _id = cfg['resume']['id']

    # Create your wandb run
    run = wandb.init(
        id=_id,
        name=wandb_config.get('name', None),
        project=wandb_config['project'],
        resume= resume,
        config = {key: value for key, value in cfg.items() if key != 'wandb'},
    )

    x = torch.randn(1, 3, 112, 112).to(cfg['device'])
    summary(model, x)

    return run



def main():
    config = load_config("config.yaml")
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = config['device']

    # Initialize model
    print("==Create Model==")
    model = CNNNetwork().to(DEVICE)
    # model = SENetwork().to(DEVICE)

    # Defining Loss function
    # criterion = ArcFaceLoss(num_classes=8631, embedding_size=8631)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion =    torch.nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=1.0)

    optimizer, scheduler = initialize_model_optimizer_scheduler(model, config)

    if config['resume']['resume']:
        model, optimizer, scheduler, epoch, metrics = load_model(model, config, optimizer, scheduler, './checkpoints/last.pth')
        config['e'] = epoch + 1
        config['epochs'] += epoch
        print(f"!!!- [Resuming from {'./checkpoints/last.pth'}]  "
              f"\n epoch_from {config['e']} "
              f"\n optimizer : {optimizer} "
              f"\n scheduler: {scheduler} "
              f"\n metrics: {metrics}")

    run = setup_wandb(model, config)

    # Create Data loader
    print("==Create Dataloaders for Image Recognition==")
    train_loader, val_loader, pair_loader, test_pair_loader = create_dataloader(config)


    # Initialising mixed-precision training.
    scaler = torch.cuda.amp.GradScaler()


    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    ## Experiments
    train_and_val(model, optimizer, scheduler, criterion, scaler, train_loader, val_loader,
                  pair_loader, config, DEVICE,
                  run)
    # train_and_val(model, optimizer, scheduler, [criterion, criterion_triplet], scaler, train_loader, val_loader, pair_loader, config, DEVICE,
    #               run)

    scores = test_epoch_ver(model, test_pair_loader, config)
    with open("verification_early_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(scores)):
            f.write("{},{}\n".format(i, scores[i]))
    print("verification_early_submission.csv saved")


    # kaggle competitions submit -c 11785-hw-2-p-2-face-verification-fall-2024 -f submission.csv -m "Message"

if __name__ == '__main__':
    main()




