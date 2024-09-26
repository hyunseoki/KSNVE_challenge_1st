import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(labels_npy, logits_npy):
    fault_types = ["normal", "inner", "outer", "ball"]
    roc_auc_scores = {'roc_auc' : 0}

    if len(np.unique(labels_npy)) < 4: return roc_auc_scores ## avoiding pl sanity check

    for idx in [1, 2, 3]:
        mask = (labels_npy == 0) | (labels_npy == idx)
        masked_labels = labels_npy[mask]
        masked_scores = logits_npy[mask]
        masked_labels = (masked_labels==idx).astype(int)
        roc_auc_scores[fault_types[idx]] = roc_auc_score(masked_labels.tolist(), masked_scores.tolist())

    # roc_auc_scores['roc_auc'] = sum(list(roc_auc_scores.values())) / 3
    label_ova = labels_npy.copy()
    label_ova[label_ova > 0] = 1
    roc_auc_scores['roc_auc'] = roc_auc_score(label_ova.tolist(), logits_npy.tolist())
    return roc_auc_scores