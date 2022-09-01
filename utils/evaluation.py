
import faiss 
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_neighbor_accuracy(fvecs, targets, k=20):
    index = faiss.IndexFlatIP(fvecs.shape[1])
    index.add(fvecs.astype(np.float32))
    _, neighbor_idx = index.search(fvecs, k+1)

    anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
    neighbor_targets = np.take(targets, neighbor_idx[:, 1:], axis=0)
    accuracy = np.mean(anchor_targets == neighbor_targets)
    return accuracy

def hungarian_match(pred, targets, pred_k, targets_k):
    num_samples = targets.shape[0]
    num_correct = np.zeros((pred_k, pred_k))

    for c1 in range(pred_k):
        for c2 in range(pred_k):
            votes = int(((pred == c1) * (targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))
    cls_map = {out_c: gt_c for out_c, gt_c in match}
    return cls_map