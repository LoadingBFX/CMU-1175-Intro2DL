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
from loss.ArcFace import ArcFaceLoss
from loss.Triplet import TripletLoss
from model.model import CNNNetwork
from model.senet import SENetwork
from test import test_epoch_ver
from train import train_epoch, train_epoch_triplet, train_epoch_combined, train_epoch_arcface
from utils import save_model, load_model
from val import valid_epoch_cls
from ver import valid_epoch_ver
import yaml

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    # 转换学习率和 eta_min 为浮点数
    config['optimizer']['lr_feature_extraction'] = float(config['optimizer']['lr_feature_extraction'])
    config['optimizer']['lr_classification'] = float(config['optimizer']['lr_classification'])
    config['scheduler']['feature']['eta_min'] = float(config['scheduler']['feature']['eta_min'])
    config['scheduler']['classifier']['eta_min'] = float(config['scheduler']['classifier']['eta_min'])

    print("Feature Learning Rate:", config['optimizer']['lr_feature_extraction'])
    print("Classifier Learning Rate:", config['optimizer']['lr_classification'])
    print("Feature Scheduler Eta Min:", config['scheduler']['feature']['eta_min'])
    print("Classifier Scheduler Eta Min:", config['scheduler']['classifier']['eta_min'])

    return config


def print_loss_configuration(criterion_ce, criterion_triplet, criterion_arcface, config):
    print("\n=== Loss Function Configuration ===")
    print("{:<20} {:<10} {}".format("Loss Function", "Status", "Parameters"))
    print("=" * 50)

    # Print CrossEntropy Loss
    print("{:<20} {:<10} {}".format(
        "CrossEntropy Loss",
        "Enabled" if criterion_ce is not None else "Disabled",
        "Smoothing: {}".format(config['loss']['cross_entropy']['smoothing'])
    ))

    # Print Triplet Loss
    if criterion_triplet is not None:
        print("{:<20} {:<10} {}".format(
            "Triplet Margin Loss",
            "Enabled",
            "Margin: {}".format(config['loss']['triplet'].get('margin', '-'))
        ))
    else:
        print("{:<20} {:<10} {}".format(
            "Triplet Margin Loss",
            "Disabled",
            "-"
        ))

    # Print ArcFace Loss
    if criterion_arcface is not None:
        print("{:<20} {:<10} {}".format(
            "ArcFace Loss",
            "Enabled",
            "s: {}, m: {}".format(config['loss']['arcface'].get('s', '-'), config['loss']['arcface'].get('m', '-'))
        ))
    else:
        print("{:<20} {:<10} {}".format(
            "ArcFace Loss",
            "Disabled",
            "-"
        ))

    print("=" * 50)




def initialize_criterion(config):
    criterion = []

    for loss_name, loss_config in config['loss_functions'].items():
        if loss_config['enabled']:
            if loss_name == 'cross_entropy':
                criterion.append(torch.nn.CrossEntropyLoss(label_smoothing=loss_config['smoothing']))
            elif loss_name == 'triplet':
                criterion.append(TripletLoss(margin=loss_config['margin']))
            elif loss_name == 'arcface':
                criterion.append(ArcFaceLoss(s=loss_config['s'], m=loss_config['m']))

    return criterion


def initialize_optimizer_scheduler(model, config):
    if config.get('use_mixed_loss', False):  # 判断是否使用混合loss
        # 分别为特征提取和分类部分设置不同的学习率
        optimizer = torch.optim.AdamW([
            {'params': model.get_feature_extractor_params(), 'lr': config['optimizer']['lr_feature_extraction']},
            {'params': model.get_classifier_params(), 'lr': config['optimizer']['lr_classification']}
        ], weight_decay=config['optimizer']['weight_decay'])

        # 如果需要为每部分分别使用scheduler，也可以这样设置
        scheduler_feature = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['scheduler']['feature']['T_max'],
            eta_min =config['scheduler']['feature']['eta_min'],
            last_epoch=config['e'] - 1  # 恢复训练时的起始epoch
        )
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['scheduler']['classifier']['T_max'],
            eta_min =config['scheduler']['classifier']['eta_min'],
            last_epoch=config['e'] - 1
        )
        return optimizer, (scheduler_feature, scheduler_classifier)

    else:
        # 不使用混合loss，直接统一设置
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr_classification'],
                                      weight_decay=config['optimizer']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'],
                                                               last_epoch=config['e'] - 1)
        return optimizer, scheduler



# def initialize_model_optimizer_scheduler(model, config):
#     # Defining Optimizer
#     optimizer = None
#     # Defining Scheduler
#     scheduler = None
#
#     optimizer_config = config['optimizer']
#     if optimizer_config['type'] == 'AdamW':
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=optimizer_config['lr'],
#             weight_decay=optimizer_config['weight_decay']
#         )
#     elif optimizer_config['type'] == 'SGD':
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=optimizer_config['lr'],
#             momentum=optimizer_config.get('momentum', 0.9),  # Optional for SGD
#             weight_decay=optimizer_config['weight_decay']
#         )
#
#     # 根据 config 选择调度器
#     scheduler_config = config['scheduler']
#     if scheduler_config['type'] == 'CosineAnnealingWarmRestarts':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=scheduler_config['T_0'],
#             T_mult=scheduler_config['T_mult'],
#             eta_min=scheduler_config['eta_min']
#         )
#     elif scheduler_config['type'] == 'CosineAnnealingLR':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=scheduler_config['T_max'],
#         )
#
#     print(f"optimizer: {optimizer} \n scheduler: {scheduler}")
#
#     return optimizer, scheduler


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
    if cfg['loss']['triplet']['enabled']:
        train_dataset = TripletImageDataset(train_dataset, transform=train_transforms)

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

    return train_dataloader, val_dataloader, pair_dataloader, test_pair_dataloader




def train_and_val(model, optimizer, scheduler, scaler, train_loader, val_loader, pair_loader, config, DEVICE, run):
    e = config['e']
    best_valid_cls_acc = 0.0
    eval_cls = True
    best_valid_ret_acc = 0.0

    # 初始化损失函数
    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=config['loss']['cross_entropy']['smoothing'])  # 交叉熵损失
    criterion_triplet = None
    criterion_arcface = None

    # 根据配置决定是否启用三元组损失或ArcFace损失
    if config['loss']['triplet']['enabled']:
        # criterion_triplet = torch.nn.TripletMarginLoss(margin=config['loss']['triplet_margin'])
        criterion_triplet = TripletLoss(margin=config['loss']['triplet']['margin']) # 三元组损失

    if config['loss']['arcface']['enabled']:
        criterion_arcface = ArcFaceLoss(s=config['loss']['arcface']['s'], m=config['loss']['arcface']['m'])  # 假设你已经实现了ArcFace损失

    # In the train_and_val function
    print_loss_configuration(criterion_ce, criterion_triplet, criterion_arcface, config)

    for epoch in range(e, config['epochs']):
        tqdm.write("\nEpoch {}/{}".format(epoch + 1, config['epochs']))

        # 根据启用的损失函数选择训练函数
        if criterion_triplet is not None and criterion_arcface is not None:
            train_cls_acc, train_loss = train_epoch_combined(
                model, train_loader, criterion_ce, criterion_triplet, criterion_arcface, optimizer, scheduler, scaler, DEVICE, config)
        elif criterion_triplet is not None:
            train_cls_acc, train_loss = train_epoch_triplet(
                model, train_loader, criterion_ce, criterion_triplet, optimizer, scheduler, scaler, DEVICE, config)
        elif criterion_arcface is not None:
            train_cls_acc, train_loss = train_epoch_arcface(
                model, train_loader, criterion_ce, criterion_arcface, optimizer, scheduler, scaler, DEVICE, config)
        else:
            train_cls_acc, train_loss = train_epoch(
                model, train_loader, criterion_ce, optimizer, scheduler, scaler, DEVICE, config)

        # 获取特征提取和分类器的学习率
        feature_lr = float(optimizer.param_groups[0]['lr'])
        classifier_lr = float(optimizer.param_groups[1]['lr'])

        tqdm.write("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Feature Learning Rate {:.04f}\t Classifier Learning Rate {:.04f}".format(
            epoch + 1, config['epochs'], train_cls_acc, train_loss, feature_lr, classifier_lr))

        metrics = {
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
            'feature_learning_rate': feature_lr,
            'classifier_learning_rate': classifier_lr
        }

        # 分类验证
        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, criterion_ce, DEVICE, config)
            tqdm.write("\nVal Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics.update({
                'valid_cls_acc': valid_cls_acc,
                'valid_loss': valid_loss,
            })

        # 检索验证
        valid_ret_acc, valid_ret_eer = valid_epoch_ver(model, pair_loader, DEVICE, config)
        tqdm.write("\nVal Ret. Acc {:.04f}% \t Val Ret. EER {:.04f}%".format(valid_ret_acc, valid_ret_eer))
        metrics.update({
            'valid_ret_acc': valid_ret_acc,
            'valid_ret_eer': valid_ret_eer,
        })

        # 保存模型
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
        tqdm.write("Saved last epoch model")

        # 保存最佳分类模型
        if eval_cls and valid_cls_acc >= best_valid_cls_acc:
            best_valid_cls_acc = valid_cls_acc
            save_model(model, optimizer, scheduler, metrics, epoch,
                       os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
            tqdm.write(f"\n######### Saved best classification model best_valid_cls_acc: {best_valid_cls_acc} #########")

        # 保存最佳检索模型
        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(model, optimizer, scheduler, metrics, epoch,
                       os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            tqdm.write(f"\n****** Saved best retrieval model best_valid_ret_acc: {best_valid_ret_acc} ******")

        # 记录到 wandb
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
    run = setup_wandb(model, config)

    # # Defining Loss function
    # # criterion = ArcFaceLoss(num_classes=8631, embedding_size=8631)
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # # criterion =    torch.nn.CrossEntropyLoss()
    # criterion_triplet = TripletLoss(margin=1.0)
    #
    # optimizer, scheduler = initialize_model_optimizer_scheduler(model, config)

    # 初始化优化器和学习率调度器
    optimizer, scheduler = initialize_optimizer_scheduler(model, config)

    if config['resume']['resume']:
        model, optimizer, scheduler, epoch, metrics = load_model(model, config, optimizer, scheduler, './checkpoints/last.pth')
        config['e'] = epoch + 1
        config['epochs'] += epoch
        print(f"!!!- [Resuming from {'./checkpoints/last.pth'}]  "
              f"\n epoch_from {config['e']} "
              f"\n optimizer : {optimizer} "
              f"\n scheduler: {scheduler} "
              f"\n metrics: {metrics}")



    # Create Data loader
    print("==Create Dataloaders for Image Recognition==")
    train_loader, val_loader, pair_loader, test_pair_loader = create_dataloader(config)


    # Initialising mixed-precision training.
    scaler = torch.cuda.amp.GradScaler()


    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    ## Experiments
    train_and_val(model, optimizer, scheduler, scaler, train_loader, val_loader,
                  pair_loader, config, DEVICE,
                  run)


    scores = test_epoch_ver(model, test_pair_loader, config)
    with open("verification_early_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(scores)):
            f.write("{},{}\n".format(i, scores[i]))
    print("verification_early_submission.csv saved")


    # kaggle competitions submit -c 11785-hw-2-p-2-face-verification-fall-2024 -f submission.csv -m "Message"

if __name__ == '__main__':
    main()




