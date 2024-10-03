"""
@author: bfx
@version: 1.0.0
@file: utils.py
@time: 9/24/24 13:02
"""
import torch
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_ver_metrics(labels, scores, FPRs):
    # eer and auc
    fpr, tpr, _ = mt.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x: 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * mt.auc(fpr, tpr)

    # get acc
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    # TPR @ FPR
    if isinstance(FPRs, list):
        TPRs = [
            ('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR)))
            for FPR in FPRs
        ]
    else:
        TPRs = []

    return {
        'ACC': ACC,
        'EER': EER,
        'AUC': AUC,
        'TPRs': TPRs,
    }


def save_model(model, optimizer, schedulers, metrics, epoch, path):
    # 如果调度器是元组，分别获取每个调度器的状态
    if isinstance(schedulers, tuple):
        scheduler_state_dicts = [scheduler.state_dict() for scheduler in schedulers]
    else:
        scheduler_state_dicts = [schedulers.state_dict()]

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dicts': scheduler_state_dicts,  # 保存调度器状态字典
        'metric': metrics,
        'epoch': epoch
    }, path)


def load_model(model, cfg, optimizer=None, schedulers=None, path='./checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['optimizer']['lr']
    else:
        optimizer = None

    if schedulers is not None:
        for i, scheduler in enumerate(schedulers):
            scheduler.load_state_dict(checkpoint['scheduler_state_dicts'][i])  # 从保存的状态中加载每个调度器
    else:
        schedulers = None

    epoch = checkpoint['epoch']
    metrics = checkpoint['metric']

    return model, optimizer, schedulers, epoch, metrics
