"""
Custom loss functions used across different methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    """
    Loss prediction loss from Learning Loss paper
    """
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        raise NotImplementedError()

    return loss


def entropic_bc_loss(out_open, label, pareto_alpha, num_classes, query, weight):
    """
    Entropic loss for binary classifier (EOAL method)
    """
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                        out_open.size(2)+1)).cuda()  
    label_range = torch.arange(0, out_open.size(0))  
    label_p[label_range, label] = 1  
    label_n = 1 - label_p
    if query > 0:
        label_p[label==num_classes,:] = pareto_alpha/num_classes
        label_n[label==num_classes,:] = pareto_alpha/num_classes
    label_p = label_p[:,:-1]
    label_n = label_n[:,:-1]
    if (query > 0) and (weight!=0):
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[label<num_classes, 1, :]
                                                        + 1e-8) * (1 - pareto_alpha) * label_p[label<num_classes], 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[label<num_classes, 0, :] +
                                                    1e-8) * (1 - pareto_alpha) * label_n[label<num_classes], 1)[0]) ##### take max negative alone
        open_loss_pos_ood = torch.mean(torch.sum(-torch.log(out_open[label==num_classes, 1, :] +
                                                    1e-8) * label_p[label==num_classes], 1))
        open_loss_neg_ood = torch.mean(torch.sum(-torch.log(out_open[label==num_classes, 0, :] +
                                                    1e-8) * label_n[label==num_classes], 1))
        
        return open_loss_pos, open_loss_neg, open_loss_neg_ood, open_loss_pos_ood
    else:
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * (1 - 0) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                    1e-8) * (1 - 0) * label_n, 1)[0]) ##### take max negative alone
        return open_loss_pos, open_loss_neg, 0, 0


def reg_loss(features, labels, cluster_centers, cluster_labels, num_classes):
    """
    Regularization loss function for EOAL
    """
    features_k, _ = features[labels<num_classes], labels[labels<num_classes]
    features_u, _ = features[labels==num_classes], labels[labels==num_classes]
    k_dists = torch.cdist(features_k, cluster_centers)
    uk_dists = torch.cdist(features_u, cluster_centers)
    pk = torch.softmax(-k_dists, dim=1)
    pu = torch.softmax(-uk_dists, dim=1)

    k_ent = -torch.sum(pk*torch.log(pk+1e-20), 1)
    u_ent = -torch.sum(pu*torch.log(pu+1e-20), 1)
    true = torch.gather(uk_dists, 1, cluster_labels.long().view(-1, 1)).view(-1)

    non_gt = torch.tensor([[i for i in range(len(cluster_centers)) if cluster_labels[x] != i] for x in range(len(uk_dists))]).long().cuda()
    others = torch.gather(uk_dists, 1, non_gt)
    intra_loss = torch.mean(true)
    inter_loss = torch.exp(-others+true.unsqueeze(1))
    inter_loss = torch.mean(torch.log(1+torch.sum(inter_loss, dim = 1)))
    loss = 0.1*intra_loss + 1*inter_loss
    return loss, k_ent.sum(), u_ent.sum()


def ova_loss(logits_open, label):
    """
    One-vs-All loss for PAL method
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    """
    One-vs-All entropy loss for PAL method
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    L_c = torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1)
    return Le, L_c
