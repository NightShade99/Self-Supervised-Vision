
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
        labels = torch.zeros((2*bs,)).long().to(zi.device)
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


class MocoLoss(nn.Module):

    def __init__(self, normalize=True, temperature=1.0):
        super(MocoLoss, self).__init__()
        self.normalize = normalize 
        self.temperature = temperature

    def forward(self, query, keys, memory_vectors):
        bs = query.shape[0]
        labels = torch.zeros((bs,)).long().to(query.device)
        mask = torch.zeros((bs, bs), dtype=bool, device=query.device).fill_diagonal_(1)

        if self.normalize:
            q_norm = F.normalize(query, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
        else:
            q_norm = query
            k_norm = keys

        pos_logits = torch.mm(q_norm, k_norm.t())[mask].unsqueeze(-1) / self.temperature                # Shape (N, 1)
        neg_logits = torch.mm(q_norm, memory_vectors.t()) / self.temperature                            # Shape (N, K)
        logits = torch.cat((pos_logits, neg_logits), dim=1)                                             # Shape (N, K+1)
        loss = F.cross_entropy(logits, labels)
        return loss


class DinoLoss(nn.Module):

    def __init__(self):
        super(DinoLoss, self).__init__()

    def forward(self, teacher_fvecs, student_fvecs, temp_s, temp_t, center):
        # teacher_fvecs has size (bs, 2, K) with teacher_global features
        # student_fvecs has size (bs, 2+V, K) with (student_global, student_local) features stacked
        targets_1 = teacher_fvecs[:, 0, :].unsqueeze(1).repeat(1, student_fvecs.size(1), 1)             # (bs, 2+V, K) of first global view features 
        targets_2 = teacher_fvecs[:, 1, :].unsqueeze(1).repeat(1, student_fvecs.size(1), 1)             # (bs, 2+V, K) of second global view features
        targets_1 = F.softmax((targets_1 - center) / temp_t, -1)
        targets_2 = F.softmax((targets_2 - center) / temp_t, -1)
        loss_1 = -(targets_1 * F.log_softmax(student_fvecs / temp_s, -1)).sum(-1).mean()         
        loss_2 = -(targets_2 * F.log_softmax(student_fvecs / temp_s, -1)).sum(-1).mean()
        return loss_1 + loss_2


class PirlLoss(nn.Module):

    def __init__(self, normalize=True, temperature=1.0, loss_weight=0.5):
        super(PirlLoss, self).__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize 
        self.temp = temperature

    def forward(self, img_features, patch_features, memory_pos_features, memory_neg_features):
        if self.normalize:
            v_img = F.normalize(img_features, p=2, dim=-1)
            v_patch = F.normalize(patch_features, p=2, dim=-1)
        else:
            v_img = img_features
            v_patch = patch_features 

        bs = img_features.size(0)
        mask = torch.zeros(bs, bs).fill_diagonal_(1).bool().to(img_features.device)

        pos_logits_1 = (torch.mm(memory_pos_features, v_patch.t()) / self.temp)[mask].view(bs, 1)                       # (bs, 1)
        pos_logits_2 = (torch.mm(memory_pos_features, v_img.t()) / self.temp)[mask].view(bs, 1)                         # (bs, 1)                                    
        neg_logits = (torch.mm(memory_pos_features, memory_neg_features.t()) / self.temp).view(bs, -1)                  # (bs, K)
        logits_1, logits_2 = torch.cat((pos_logits_1, neg_logits), 1), torch.cat((pos_logits_2, neg_logits), 1)         # (bs, K+1)
        labels = torch.zeros(bs).long().to(img_features.device)
        loss_1, loss_2 = F.cross_entropy(logits_1, labels), F.cross_entropy(logits_2, labels)
        return self.loss_weight * loss_1 + (1.0 - self.loss_weight) * loss_2
        

class BarlowLoss(nn.Module):

    def __init__(self, normalize=True, off_diagonal_weight=0.005):
        super(BarlowLoss, self).__init__()
        self.normalize = normalize
        self.lmbda = off_diagonal_weight

    def forward(self, z_i, z_j):
        if self.normalize:
            zi_norm = F.normalize(z_i, p=2, dim=-1)
            zj_norm = F.normalize(z_j, p=2, dim=-1)
        else:
            zi_norm = z_i
            zj_norm = z_j
        
        bs, fsize = zi_norm.size()
        zi_norm = (zi_norm - zi_norm.mean(0)) / zi_norm.std(0)
        zj_norm = (zj_norm - zj_norm.mean(0)) / zj_norm.std(0)
        corr_mat = torch.mm(zi_norm.t(), zj_norm) / bs
        loss = F.mse_loss(corr_mat, torch.eye(fsize).to(zi_norm.device), reduction="none")
        off_diag_factor = (torch.ones(fsize, fsize) * self.lmbda).fill_diagonal_(1.0).to(zi_norm.device)
        loss = loss * off_diag_factor
        return loss.sum()
