"""
@author: bfx
@version: 1.0.0
@file: train.py
@time: 9/24/24 12:59
"""
import torch
from tqdm import tqdm

from loss.Triplet import TripletLoss
from metrics.AverageMeter import AverageMeter
from utils import accuracy


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, scaler, device, config):

    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train', ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
            outputs = model(images)

            # Use the type of output depending on the loss function you want to use
            loss = criterion(outputs['out'], labels)

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update()
        # metrics
        loss_m.update(loss.item())
        if 'feats' in outputs:
            acc = accuracy(outputs['out'], labels)[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            # acc         = "{:.04f}%".format(100*accuracy),
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss        = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

    # You may want to call some schedulers inside the train function. What are these?
    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()

    return acc_m.avg, loss_m.avg


def train_epoch_triplet(model, dataloader, criterion_ce, criterion_triplet, optimizer, lr_schedulers, scaler, device, config):
    model.train()

    # Metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train', ncols=5)

    for i, (anchor_img, positive_img, negative_img, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        # Send to device
        anchor_img = anchor_img.to(device, non_blocking=True)
        positive_img = positive_img.to(device, non_blocking=True)
        negative_img = negative_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(anchor_img)
            anchor_feats = outputs['feats']
            positive_feats = model(positive_img)['feats']
            negative_feats = model(negative_img)['feats']

            # 计算 triplet loss
            triplet_loss = criterion_triplet(anchor_feats, positive_feats, negative_feats)

            # 计算交叉熵损失
            ce_loss = criterion_ce(outputs['out'], labels)

            # 总损失
            loss = 0.7 * triplet_loss + 0.3 * ce_loss

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        loss_m.update(loss.item())
        acc = accuracy(outputs['out'], labels)[0].item()
        acc_m.update(acc)

        # Update progress bar
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update()

    # 更新学习率调度器
    if isinstance(lr_schedulers, tuple):  # 如果有两个scheduler
        lr_schedulers[0].step()  # 调整特征提取部分的学习率
        lr_schedulers[1].step()  # 调整分类部分的学习率
    elif lr_schedulers is not None:
        lr_schedulers.step()  # 调整统一的学习率

    batch_bar.close()
    return acc_m.avg, loss_m.avg

def train_epoch_arcface(model, dataloader, criterion_ce, criterion_arcface, optimizer, lr_schedulers, scaler, device,
                        config):
    model.train()

    # Metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train', ncols=5)

    for i, (anchor_img, positive_img, negative_img, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        # Send to device
        anchor_img = anchor_img.to(device, non_blocking=True)
        positive_img = positive_img.to(device, non_blocking=True)
        negative_img = negative_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(anchor_img)

            # 计算 triplet loss
            arcface_loss = criterion_arcface(outputs['feats'], labels)

            # 计算交叉熵损失
            ce_loss = criterion_ce(outputs['out'], labels)

            # 总损失
            loss = 0.7 * arcface_loss + 0.3 * ce_loss

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        loss_m.update(loss.item())
        acc = accuracy(outputs['out'], labels)[0].item()
        acc_m.update(acc)

        # Update progress bar
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update()

    # 更新学习率调度器
    if isinstance(lr_schedulers, tuple):  # 如果有两个scheduler
        lr_schedulers[0].step()  # 调整特征提取部分的学习率
        lr_schedulers[1].step()  # 调整分类部分的学习率
    elif lr_schedulers is not None:
        lr_schedulers.step()  # 调整统一的学习率

    batch_bar.close()
    return acc_m.avg, loss_m.avg


def train_epoch_combined(model, train_loader, criterion_ce, criterion_triplet, criterion_arcface, optimizer, scheduler,
                         scaler, DEVICE, config):
    pass



