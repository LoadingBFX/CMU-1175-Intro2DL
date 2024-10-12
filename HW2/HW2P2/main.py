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
from model.ConvNext import ConvNext
from model.ResNet18 import ResNet18
from model.model import CNNNetwork
from model.senet import SENetwork
from test import test_epoch_ver
from train import train_epoch, train_epoch_triplet, train_epoch_combined, train_epoch_arcface
from utils import save_model, load_model
from val import valid_epoch_cls
from ver import valid_epoch_ver
import yaml
from pytorch_metric_learning import losses

# import random
# import numpy as np
# import torch
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
#     torch.backends.cudnn.benchmark = False     # 保证实验可复现




def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    # 转换学习率和 eta_min 为浮点数
    # config['optimizer']['lr_feature_extraction'] = float(config['optimizer']['lr_feature_extraction'])
    # config['optimizer']['lr_classification'] = float(config['optimizer']['lr_classification'])
    # config['scheduler']['feature']['eta_min'] = float(config['scheduler']['feature']['eta_min'])
    # config['scheduler']['classifier']['eta_min'] = float(config['scheduler']['classifier']['eta_min'])
    #
    # print("Feature Learning Rate:", config['optimizer']['lr_feature_extraction'])
    # print("Classifier Learning Rate:", config['optimizer']['lr_classification'])
    # print("Feature Scheduler Eta Min:", config['scheduler']['feature']['eta_min'])
    # print("Classifier Scheduler Eta Min:", config['scheduler']['classifier']['eta_min'])

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
    # 初始化损失函数
    criterion_ce = None  # 交叉熵损失
    criterion_triplet = None
    criterion_arcface = None

    if config['loss']['cross_entropy']['enabled']:
        criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=config['loss']['cross_entropy']['smoothing'])
    # 根据配置决定是否启用三元组损失或ArcFace损失
    if config['loss']['triplet']['enabled']:
        # criterion_triplet = torch.nn.TripletMarginLoss(margin=config['loss']['triplet_margin'])
        # criterion_triplet = TripletLoss(margin=config['loss']['triplet']['margin'])  # 三元组损失
        criterion_triplet = torch.nn.TripletMarginLoss(margin=config['loss']['triplet']['margin'],
                                                       swap=config['loss']['triplet']['swap'])

    if config['loss']['arcface']['enabled']:
        criterion_arcface = losses.ArcFaceLoss(num_classes=8631, embedding_size=2048,
                                               margin=config['loss']['arcface']['m'],
                                               scale=config['loss']['arcface']['s'])  # 假设你已经实现了ArcFace损失

    # In the train_and_val function
    print_loss_configuration(criterion_ce, criterion_triplet, criterion_arcface, config)

    return criterion_ce, criterion_triplet, criterion_arcface


def get_scheduler(optimizer, sched_config):
    if sched_config['type'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max'],
            eta_min=float(sched_config['eta_min']),
        )
    elif sched_config['type'] == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=sched_config['T_0'],
            T_mult=sched_config['T_mult'],
            eta_min=float(sched_config['eta_min']),
        )
    elif sched_config['type'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_config['step_size'],
            gamma=sched_config['gamma'],
        )
    elif sched_config['type'] == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_config.get('mode', 'min'),
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            threshold=sched_config.get('threshold', 1e-4),
            min_lr=sched_config.get('min_lr', 0),
            verbose=sched_config.get('verbose', False)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_config['type']}")


def initialize_optimizer_scheduler(model, config):
    if config.get('use_mixed_loss', False):  # 使用混合loss时，分开设置特征提取和分类的scheduler
        # optimizer = torch.optim.AdamW([
        #     {'params': model.get_feature_extractor_params(), 'lr': config['optimizer']['lr_feature_extraction']},
        #     {'params': model.get_classifier_params(), 'lr': config['optimizer']['lr_classification']}
        # ], weight_decay=config['optimizer']['weight_decay'])

        optimizer_feature = torch.optim.AdamW(model.get_feature_extractor_params(),
                                              lr=float(config['optimizer']['lr_feature_extraction']),
                                              weight_decay=float(config['optimizer']['weight_decay']))  # 为 ArcFace 部分
        optimizer_classifier = torch.optim.AdamW(model.get_classifier_params(),
                                                 lr=float(config['optimizer']['lr_classification']),
                                                 weight_decay=float(config['optimizer']['weight_decay']))  # 为分类部

        scheduler_feature = get_scheduler(optimizer_feature, config['scheduler']['feature'])
        scheduler_classifier = get_scheduler(optimizer_classifier, config['scheduler']['classifier'])

        # 打印优化器和调度器的类型及参数
        print(f"optimizer_feature: {optimizer_feature} \n"
              f"optimizer_classifier: {optimizer_classifier} \n"
              f"scheduler_feature type: {scheduler_feature.__class__.__name__} \n"
              f"scheduler_feature parameters: {scheduler_feature.state_dict()} \n"
              f"scheduler_classifier type: {scheduler_classifier.__class__.__name__} \n"
              f"scheduler_classifier parameters: {scheduler_classifier.state_dict()}"
              )

        return (optimizer_feature, optimizer_classifier), (scheduler_feature, scheduler_classifier)

    else:  # 不使用混合loss时，使用统一的scheduler

        optimizer_config = config['optimizer']

        optimizer_type = optimizer_config.get('type')
        lr = float(optimizer_config.get('lr'))
        weight_decay = float(optimizer_config.get('weight_decay', 0))

        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'SGD':
            momentum = optimizer_config.get('momentum', 0.9)  # Optional for SGD
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: '{optimizer_type}'. Supported types: 'AdamW', 'SGD'.")

        scheduler = get_scheduler(optimizer, config['scheduler'])

        # 打印优化器和调度器的类型及参数
        print(f"Optimizer: {optimizer} \n"
              f"Scheduler type: {scheduler.__class__.__name__} \n"
              f"Scheduler parameters: {scheduler.state_dict()}")

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

    # train_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(112),
    #     torchvision.transforms.RandomHorizontalFlip(0.5),
    #     torchvision.transforms.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.1),
    #     torchvision.transforms.RandomRotation(20),
    #     torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    #     torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                      std=[0.5, 0.5, 0.5]),
    #
    #     torchvision.transforms.RandomErasing(p=0.3, scale=(0.05, 0.1)),
    # ])

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        # torchvision.transforms.RandomHorizontalFlip(0.5),
        # torchvision.transforms.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.1),
        torchvision.transforms.RandomRotation(18),
        # torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),

        # torchvision.transforms.RandomErasing(p=0.3, scale=(0.1, 0.2)),
    ])

    # val transforms
    # val_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(112),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225]),])
    #
    # ver_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.CenterCrop(112),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225]),])

    # # val transforms
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    ver_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])


    # # val transforms
    # val_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(250),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])])
    #
    # ver_transforms = val_transforms



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
                               num_workers=8,
                               sampler=None)
    val_dataloader = DataLoader(val_dataset,
                             batch_size=cfg["batch_size"],
                             shuffle=False,
                             num_workers=4)


    pair_dataset = ImagePairDataset(cfg['data_ver_dir'], csv_file=cfg['val_pairs_file'], transform=ver_transforms)
    pair_dataloader = torch.utils.data.DataLoader(pair_dataset,
                                                  batch_size=cfg["batch_size"],
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4)


    test_pair_dataset = TestImagePairDataset(cfg['data_ver_dir'], csv_file=cfg['test_pairs_file'], transform=ver_transforms)
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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['loss']['cross_entropy']['smoothing'])

    criterion_ce, criterion_triplet, criterion_arcface = initialize_criterion(config)
    # 获取启用的损失函数列表
    enabled_criteria = [(criterion_ce, "ce"), (criterion_triplet, "triplet"), (criterion_arcface, "arcface")]
    enabled_criteria = [(criterion, name) for criterion, name in enabled_criteria if criterion is not None]

    initial_triplet_alpha = config['loss']['triplet']['alpha']
    initial_arcface_alpha = config['loss']['arcface']['alpha']

    for epoch in range(e, config['epochs']):
        tqdm.write("\nEpoch {}/{}".format(epoch + 1, config['epochs']))
        if epoch < 1000:
            # 动态调整loss比例
            config['loss']['triplet']['alpha'] = 0.001
            config['loss']['arcface']['alpha'] =  0.001
        else:
            # 动态调整loss比例
            config['loss']['triplet']['alpha'] = initial_triplet_alpha * ((epoch + 1) / config['epochs'])
            config['loss']['arcface']['alpha'] = initial_arcface_alpha * ((epoch + 1) / config['epochs'])

        # 根据启用的损失函数数量选择对应的训练函数
        if len(enabled_criteria) == 3:
            # 三个损失函数都启用
            train_cls_acc, train_loss = train_epoch_combined(
                model, train_loader, criterion_ce, criterion_triplet, criterion_arcface, optimizer, scheduler, scaler,
                DEVICE, config)
        elif len(enabled_criteria) == 2:
            # 启用交叉熵 + 另一个损失函数
            _, second_loss_name = enabled_criteria[1]
            if second_loss_name == "triplet":
                train_cls_acc, train_loss = train_epoch_triplet(
                    model, train_loader, criterion_ce, criterion_triplet, optimizer, scheduler, scaler, DEVICE, config)
            elif second_loss_name == "arcface":
                train_cls_acc, train_loss = train_epoch_arcface(
                    model, train_loader, criterion_ce, criterion_arcface, optimizer, scheduler, scaler, DEVICE, config)
        elif len(enabled_criteria) == 1:
            # 只启用一个损失函数
            loss_name = enabled_criteria[0][1]
            if loss_name == "ce":
                train_cls_acc, train_loss = train_epoch(
                    model, train_loader, criterion_ce, optimizer, scheduler, scaler, DEVICE, config)
            elif loss_name == "triplet":
                train_cls_acc, train_loss = train_epoch_triplet(
                    model, train_loader, None, criterion_triplet, optimizer, scheduler, scaler, DEVICE, config)
            elif loss_name == "arcface":
                train_cls_acc, train_loss = train_epoch_arcface(
                    model, train_loader, None, criterion_arcface, optimizer, scheduler, scaler, DEVICE, config)

        if config.get('use_mixed_loss', False):
            # 获取特征提取和分类器的学习率
            feature_lr = float(optimizer[0].param_groups[0]['lr'])
            classifier_lr = float(optimizer[1].param_groups[0]['lr'])

            tqdm.write(
                "\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Feature Learning Rate {:.04f}\t Classifier Learning Rate {:.04f}".format(
                    epoch + 1, config['epochs'], train_cls_acc, train_loss, feature_lr, classifier_lr))

            metrics = {
                'train_cls_acc': train_cls_acc,
                'train_loss': train_loss,
                'feature_learning_rate': feature_lr,
                'classifier_learning_rate': classifier_lr,
                'alpha' : config['loss']['arcface']['alpha'],
            }

        else:
            curr_lr = float(optimizer.param_groups[0]['lr'])
            tqdm.write("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1, config['epochs'], train_cls_acc, train_loss, curr_lr))

            metrics = {
                'train_cls_acc': train_cls_acc,
                'train_loss': train_loss,
                'lr': curr_lr
            }


        # 分类验证
        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, criterion, DEVICE, config)
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

def val(model, optimizer, scheduler, scaler, train_loader, val_loader, pair_loader, config, DEVICE, run):
    # 检索验证
    valid_ret_acc, valid_ret_eer = valid_epoch_ver(model, pair_loader, DEVICE, config)
    tqdm.write("\nVal Ret. Acc {:.04f}% \t Val Ret. EER {:.04f}%".format(valid_ret_acc, valid_ret_eer))



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
    if config['model']['type'] == 'ResNet50':
        model = CNNNetwork().to(DEVICE)
    elif config['model']['type'] == 'ResNet18':
        model = ResNet18().to(DEVICE)
    elif config['model']['type'] == 'SENet':
        model = SENetwork().to(DEVICE)
    elif config['model']['type'] == 'ConvNext':
        model = ConvNext().to(DEVICE)

    # 拼接 wandb name
    enabled_losses = []
    if config['loss']['cross_entropy']['enabled']:
        enabled_losses.append('CE')
    if config['loss']['triplet']['enabled']:
        enabled_losses.append('Triplet')
    if config['loss']['arcface']['enabled']:
        enabled_losses.append('ArcFace')

    loss_str = '_'.join(enabled_losses) if enabled_losses else 'None'

    # 根据 use_mixed_loss 判断优化器和调度器
    if config['use_mixed_loss']:
        # Mixed loss 情况下，拼接特征提取和分类的学习率和调度器
        lr_feature_extraction = config['optimizer']['lr_feature_extraction']
        lr_classification = config['optimizer']['lr_classification']

        feature_scheduler = config['scheduler']['feature']['T_max']
        classifier_scheduler = config['scheduler']['classifier']['T_max']

        config['wandb'][
            'name'] = f"{config['model']['type']}_{loss_str}_lrF{lr_feature_extraction}_lrC{lr_classification}_schF{feature_scheduler}_schC{classifier_scheduler}_bs{config['batch_size']}"
    else:
        # 非 mixed loss 情况下，拼接统一的学习率和调度器
        lr = config['optimizer']['lr']
        scheduler_type = config['scheduler']['T_max']

        config['wandb'][
            'name'] = f"{config['model']['type']}_{loss_str}_lr{lr}_sch{scheduler_type}_bs{config['batch_size']}"


    run = setup_wandb(model, config)

    # 初始化优化器和学习率调度器
    optimizer, scheduler = initialize_optimizer_scheduler(model, config)

    if config['resume']['resume']:
        model, optimizer, scheduler, epoch, metrics = load_model(model, config, optimizer, scheduler, './checkpoints/best.pth')
        config['e'] = epoch + 1
        config['epochs'] += epoch
        # print(f"!!!- [Resuming from {'./checkpoints/best_ret.pth'}]  "
        #       f"\n epoch_from {config['e']} \n "
        #       f"Optimizer: {optimizer} \n"
        #       f"Scheduler type: {scheduler.__class__.__name__} \n"
        #       f"Scheduler parameters: {scheduler.state_dict()}"
        #       f"\n metrics: {metrics}")



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
    # val(model, optimizer, scheduler, scaler, train_loader, val_loader, pair_loader, config, DEVICE, run)


    scores = test_epoch_ver(model, test_pair_loader, config)
    with open("verification_early_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(scores)):
            f.write("{},{}\n".format(i, scores[i]))
    print("verification_early_submission.csv saved")


    # kaggle competitions submit -c 11785-hw-2-p-2-face-verification-fall-2024 -f submission.csv -m "Message"

if __name__ == '__main__':
    # set_seed(42)
    main()




