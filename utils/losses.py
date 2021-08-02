
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SimclrLoss(nn.Module):
    
    def __init__(self, normalize=False, temperature=1.0):
        super(SimclrLoss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, zi, zj):
        bs = zi.shape[0]
        labels = torch.zeros(2 * bs).long().to(zi.device)
        mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

        if self.normalize:
            zi_norm = F.normalize(zi, p=2, dim=-1)
            zj_norm = F.normalize(zj, p=2, dim=-1)
        else:
            zi_norm = zi
            zj_norm = zj

        logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temperature
        logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temperature
        logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temperature
        logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temperature

        logits_ij_pos = logits_ij[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)                                             # Shape (N, N-1)

        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)                         # Shape (2N, 1)
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                                    # Shape (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                                    # Shape (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                                      # Shape (2N, 2N-2)

        logits = torch.cat((pos, neg), dim=1)                                                       # Shape (2N, 2N-1)
        loss = F.cross_entropy(logits, labels)
        return loss


class InfoNCELoss(nn.Module):

    def __init__(self, normalize=True, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.normalize = normalize 
        self.temperature = temperature

    def forward(self, query, keys, memory_vectors):
        bs = query.shape[0]
        labels = torch.zeros((bs,)).long().to(query.device)
        mask = torch.zeros((bs, bs), dtype=bool).fill_diagonal_(1)

        if self.normalize:
            q_norm = F.normalize(query, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
        else:
            q_norm = query
            k_norm = keys

        pos_logits = torch.mm(q_norm, k_norm.t())[mask].unsqueeze(-1) / self.temperature            # Shape (N, 1)
        neg_logits = torch.mm(q_norm, memory_vectors.t()) / self.temperature                        # Shape (N, K)
        logits = torch.cat((pos_logits, neg_logits), dim=1)                                         # Shape (N, K+1)
        loss = F.cross_entropy(logits, labels)
        return loss