"""
@author: bfx
@version: 1.0.0
@file: ver.py
@time: 9/24/24 13:01
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils import get_ver_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from pprint import pprint


def print_metrics(metric_dict):
    print("Verification Metrics:")
    print(f"{'ACC':<15}: {metric_dict['ACC']:.2f}%")
    print(f"{'EER':<15}: {metric_dict['EER']:.4f}%")
    print(f"{'AUC':<15}: {metric_dict['AUC']:.2f}%")

    print("\nTrue Positive Rates at Specific False Positive Rates:")
    for tpr, value in metric_dict['TPRs']:
        print(f"{tpr:<20}: {value:.4f}%")

def valid_epoch_ver(model, pair_data_loader, device, config):

    model.eval()
    scores = []
    match_labels = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc='Val Veri.')
    for i, (images1, images2, labels) in enumerate(pair_data_loader):

        # match_labels = match_labels.to(device)
        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs['feats'], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())
        batch_bar.update()

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)

    FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']
    metric_dict = get_ver_metrics(match_labels.tolist(), scores.tolist(), FPRs)
    # print(metric_dict)
    print_metrics(metric_dict)

    return metric_dict['ACC'], metric_dict['EER']