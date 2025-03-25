"""
Training functionality for PAL (Probabilistic Active Learning) method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from tqdm import tqdm
from utils.general_utils import AverageMeter
from utils.loss_functions import ova_loss, ova_ent
from utils.ema import ModelEMA


def train_pal_cls_epoch(args, models, optimizers, dataloaders, ema_model):
    """
    Training epoch for the classifier part of PAL.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        ema_model: EMA model for model averaging
    """
    losses = AverageMeter("losses")
    losses_x = AverageMeter("losses_x")
    models['backbone'].train()
    
    for train_data in dataloaders['train']:
        # Data loading
        feature_id, targets_x, index_x = train_data

        b_size = feature_id.shape[0]
        input_l = feature_id.to(args.device)
        targets_x = targets_x.to(args.device)

        # Feed data
        logits, logits_open = models['backbone'](input_l)
        
        # Loss for labeled samples
        Lx = F.cross_entropy(logits[:b_size], targets_x, reduction='mean')
        Lo = ova_loss(logits_open[:b_size], targets_x)
        loss = Lx + Lo
        loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())

        # Optimization step
        optimizers['backbone'].step()
        
        if args.use_ema:
            ema_model.update(models['backbone'])
        
        models['backbone'].zero_grad()


def train_pal_meta_epoch(args, models, optimizers, dataloaders, coef, wnet, optimizer_wnet):
    """
    Training epoch for the meta-learning part of PAL.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        coef: coefficient for weighting losses
        wnet: weight network
        optimizer_wnet: optimizer for weight network
    """
    losses_hat = AverageMeter("losses_hat")
    losses_wet = AverageMeter("losses_wet")
    losses_ova = AverageMeter("losses_ova")
    meta_model = models['ood_detection']
    meta_model.train()

    # 将迭代器初始化移到循环外，避免每次循环都重新初始化
    labeled_iter = iter(dataloaders['train'])
    unlabeled_iter = iter(dataloaders['unlabeled'])

    for _ in range(args.meta_step):
        try:
            # 获取下一个 labeled 数据
            feature_id, targets_x, _ = next(labeled_iter)
        except StopIteration:
            # 重新初始化迭代器并继续
            labeled_iter = iter(dataloaders['train'])
            feature_id, targets_x, _ = next(labeled_iter)
        try:
            # 获取下一个 unlabeled 数据
            feature_al, _, _ = next(unlabeled_iter)
        except StopIteration:
            # 重新初始化迭代器并继续
            unlabeled_iter = iter(dataloaders['unlabeled'])
            feature_al, _, _ = next(unlabeled_iter)
        b_size = feature_id.shape[0]

        # 先将数据移动到设备，再进行拼接，避免在CPU上进行不必要的操作
        feature_id = feature_id.to(args.device)
        feature_al = feature_al.to(args.device)
        inputs = torch.cat([feature_id, feature_al], 0)
        input_l = feature_id  # 已经在设备上，无需再次调用 to()
        targets_x = targets_x.to(args.device)

        logits, logits_open = meta_model(inputs, method='PAL')
        logits_open_w = logits_open[b_size:]
        weight = wnet(logits_open_w)

        # 加上一个很小的 epsilon，避免除以零，并消除不必要的 if 语句
        norm = torch.sum(weight) + 1e-8
        Lx = F.cross_entropy(logits[:b_size], targets_x, reduction='mean')
        Lo = ova_loss(logits_open[:b_size], targets_x)
        losses_ova.update(Lo.item())
        L_o_u1, cost_w = ova_ent(logits_open_w)
        cost_w = cost_w.view(-1, 1)  # 使用 view 代替 reshape，提高效率

        loss_hat = Lx + coef * (torch.sum(weight * cost_w) / norm + Lo)

        meta_model.zero_grad()
        loss_hat.backward()
        optimizers['ood_detection'].step()

        losses_hat.update(loss_hat.item())
        y_l_hat, _ = meta_model(input_l, method='PAL')
        L_cls = F.cross_entropy(y_l_hat, targets_x, reduction='mean')

        # 计算上层目标
        optimizer_wnet.zero_grad()
        L_cls.backward()
        optimizer_wnet.step()
        losses_wet.update(L_cls.item())