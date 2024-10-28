"""
@author: bfx
@version: 1.0.0
@file: train.py
@time: 9/24/24 12:59
"""
import torch
from pyexpat import features
from tqdm import tqdm

from loss.Triplet import TripletLoss
from metrics.AverageMeter import AverageMeter
from utils import accuracy
import torch.nn.functional as F


#
# def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, scaler, device, config):
#
#     model.train()
#
#     # metric meters
#     loss_m = AverageMeter()
#     acc_m = AverageMeter()
#
#     # Progress Bar
#     batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train', ncols=5)
#
#     for i, (images, labels) in enumerate(dataloader):
#
#         optimizer.zero_grad() # Zero gradients
#
#         # send to cuda
#         images = images.to(device, non_blocking=True)
#         if isinstance(labels, (tuple, list)):
#             targets1, targets2, lam = labels
#             labels = (targets1.to(device), targets2.to(device), lam)
#         else:
#             labels = labels.to(device, non_blocking=True)
#
#         # forward
#         with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
#             outputs = model(images)
#
#             # Use the type of output depending on the loss function you want to use
#             loss = criterion(outputs['out'], labels)
#
#         scaler.scale(loss).backward() # This is a replacement for loss.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer) # This is a replacement for optimizer.step()
#         scaler.update()
#         # metrics
#         loss_m.update(loss.item())
#         if 'feats' in outputs:
#             acc = accuracy(outputs['out'], labels)[0].item()
#         else:
#             acc = 0.0
#         acc_m.update(acc)
#
#         # tqdm lets you add some details so you can monitor training as you train.
#         batch_bar.set_postfix(
#             # acc         = "{:.04f}%".format(100*accuracy),
#             acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
#             loss        = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
#             lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
#
#         batch_bar.update() # Update tqdm bar
#
#     # You may want to call some schedulers inside the train function. What are these?
#     if lr_scheduler is not None:
#         lr_scheduler.step()
#
#     batch_bar.close()
#
#     return acc_m.avg, loss_m.avg


# R - Drop
def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, scaler, device, config, lambda_rdrop=0.2):

    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    arcface_criterion = config['arcface_loss']

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train', ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():  # This implements mixed precision. That's it!
            # First forward pass
            outputs1 = model(images)
            # Second forward pass for R-Drop
            outputs2 = model(images)

            # Compute the original loss using the criterion
            loss = criterion(outputs1['out'], labels)
            arcface_loss = arcface_criterion(outputs1['feats'], labels)

            # Compute the KL divergence between the two outputs
            kl_loss = F.kl_div(F.log_softmax(outputs1['out'], dim=-1), F.softmax(outputs2['out'], dim=-1), reduction='batchmean')
            kl_loss += F.kl_div(F.log_softmax(outputs2['out'], dim=-1), F.softmax(outputs1['out'], dim=-1), reduction='batchmean')

            # Combine original loss and R-Drop loss
            total_loss = loss + kl_loss * lambda_rdrop + arcface_loss * 0.5

        # Backpropagation and optimization
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        loss_m.update(total_loss.item())
        if 'feats' in outputs1:
            acc = accuracy(outputs1['out'], labels)[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)

        # Update tqdm bar
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(total_loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update()

    # Learning rate scheduling
    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()

    return acc_m.avg, loss_m.avg



def train_epoch_triplet(model, dataloader, criterion_ce, criterion_triplet, optimizer, lr_schedulers, scaler, device, config):
    model.train()

    # Metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    alpha = config['loss']['triplet']['alpha']  # Triplet 损失权重
    beta = config['loss']['triplet']['beta']  # 交叉熵损失权重

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    for i, (anchor_img, positive_img, negative_img, labels) in enumerate(dataloader):

        if isinstance(optimizer, tuple):  # 如果有两个 scheduler
            optimizer[0].zero_grad()  # 调整特征提取部分的学习率
            optimizer[1].zero_grad()  # 调整分类部分的学习率
        elif lr_schedulers is not None:
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

            ce_loss = 0
            if criterion_ce:
                # 计算交叉熵损失
                ce_loss = criterion_ce(outputs['out'], labels)

            # 总损失
            loss = alpha * triplet_loss + beta * ce_loss

            # Backward pass
            scaler.scale(loss).backward()

            # 梯度裁剪
            if isinstance(optimizer, tuple):
                for opt in optimizer:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可以调整max_norm
                    scaler.step(opt)
                    # Update progress bar
                scaler.update()

                # Metrics
                loss_m.update(loss.item())
                acc = accuracy(outputs['out'], labels)[0].item()
                acc_m.update(acc)

                batch_bar.set_postfix(
                    mode="Tri_Mix_{:.04f}_{:.04f}".format(alpha, beta),
                    acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
                    loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
                    features_lr="{:.04f}".format(float(optimizer[0].param_groups[0]['lr'])),
                    cls_lr="{:.04f}".format(float(optimizer[1].param_groups[0]['lr']))
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可以调整max_norm
                scaler.step(optimizer)
                scaler.update()

                # Metrics
                loss_m.update(loss.item())
                acc = accuracy(outputs['out'], labels)[0].item()
                acc_m.update(acc)

                # Update progress bar
                batch_bar.set_postfix(
                    mode="Tri_{:.04f}_{:.04f}".format(alpha, beta),
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

    alpha = config['loss']['arcface']['alpha']  # ArcFace 损失权重
    beta = config['loss']['arcface']['beta']  # 交叉熵损失权重

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        if isinstance(optimizer, tuple):  # 如果有两个scheduler
            optimizer[0].zero_grad()  # 调整特征提取部分的学习率
            optimizer[1].zero_grad()  # 调整分类部分的学习率
        elif lr_schedulers is not None:
            optimizer.zero_grad() # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)

            # # First forward pass
            # outputs1 = model(images)
            # # Second forward pass for R-Drop
            # outputs2 = model(images)
            #
            # # Compute the KL divergence between the two outputs
            # kl_loss = F.kl_div(F.log_softmax(outputs1['out'], dim=-1), F.softmax(outputs2['out'], dim=-1),
            #                    reduction='batchmean')
            # kl_loss += F.kl_div(F.log_softmax(outputs2['out'], dim=-1), F.softmax(outputs1['out'], dim=-1),
            #                     reduction='batchmean')


            arcface_loss = criterion_arcface(outputs['feats'], labels)

            ce_loss = 0
            if criterion_ce:
                # 计算交叉熵损失
                ce_loss = criterion_ce(outputs['out'], labels)

            # 总损失
            loss = alpha * arcface_loss + beta * ce_loss

        # Backward pass
        scaler.scale(loss).backward()



        # 梯度裁剪
        if isinstance(optimizer, tuple):
            for opt in optimizer:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可以调整max_norm
                scaler.step(opt)
                # Update progress bar
            scaler.update()

            # Metrics
            loss_m.update(loss.item())
            acc = accuracy(outputs['out'], labels)[0].item()
            acc_m.update(acc)

            batch_bar.set_postfix(
                mode="arcface_{:.04f}_{:.04f}".format(alpha, beta),
                acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
                loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
                features_lr="{:.04f}".format(float(optimizer[0].param_groups[0]['lr'])),
                cls_lr = "{:.04f}".format(float(optimizer[1].param_groups[0]['lr']))
            )
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可以调整max_norm
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            loss_m.update(loss.item())
            acc = accuracy(outputs['out'], labels)[0].item()
            acc_m.update(acc)

            # Update progress bar
            batch_bar.set_postfix(
                mode="arcface_{:.04f}_{:.04f}".format(alpha, beta),
                acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
                loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
            )


        # scaler.update()

        # # Metrics
        # loss_m.update(loss.item())
        # acc = accuracy(outputs['out'], labels)[0].item()
        # acc_m.update(acc)

        # Update progress bar
        # batch_bar.set_postfix(
        #     mode= f"arcface_{alpha}_{beta}",
        #     acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
        #     loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
        #     lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        # )
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



