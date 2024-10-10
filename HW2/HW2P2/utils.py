"""
@author: bfx
@version: 1.0.0
@file: utils.py
@time: 9/24/24 13:02
"""
import torch
import torchvision
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
    state_dict = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'epoch': epoch
    }

    # 判断是否有多个优化器
    if isinstance(optimizer, (tuple, list)):
        state_dict['optimizer_state_dict'] = [opt.state_dict() for opt in optimizer]
    else:
        state_dict['optimizer_state_dict'] = optimizer.state_dict()

    # 判断是否有多个调度器
    if isinstance(schedulers, (tuple, list)):
        state_dict['scheduler_state_dict'] = [sched.state_dict() for sched in schedulers]
    else:
        state_dict['scheduler_state_dict'] = schedulers.state_dict()

    torch.save(state_dict, path)


def load_model(model, cfg, optimizer=None, schedulers=None, path='./checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.cls_layer.parameters():
        param.requires_grad = False
    print("Last fully connected layer parameters frozen.")

    # # 加载优化器状态
    # if optimizer is not None:
    #     if isinstance(optimizer, (tuple, list)):
    #         for opt, opt_state in zip(optimizer, checkpoint['optimizer_state_dict']):
    #             opt.load_state_dict(opt_state)
    #             for param_group in opt.param_groups:
    #                 param_group['lr'] = cfg['optimizer']['lr']
    #     else:
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = cfg['optimizer']['lr']

    # # 加载调度器状态
    # if schedulers is not None:
    #     if isinstance(schedulers, (tuple, list)):
    #         for sched, sched_state in zip(schedulers, checkpoint['scheduler_state_dict']):
    #             sched.load_state_dict(sched_state)
    #     else:
    #         schedulers.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint['metric']

    print(f"Checkpoint loaded. Last epoch: {epoch}, Metrics: {metrics}")

    return model, optimizer, schedulers, epoch, metrics

import torch
from torchvision import datasets, transforms
def cal_mean_std():
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.1),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        torchvision.transforms.ToTensor()
    ])

    # Load the dataset and transform images to tensor
    dataset = datasets.ImageFolder('./data/11-785-f24-hw2p2-verification/cls_data/dev', transform=train_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

    mean = torch.zeros(3)  # Assuming RGB images
    std = torch.zeros(3)
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten image pixels
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    print(f"Mean: {mean}, Std: {std}")

if __name__ == '__main__':
    cal_mean_std()
