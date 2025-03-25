"""
Training functionality for MQNet (Meta Query Network) method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.loss_functions import LossPredLoss
from methods.methods_utils.mqnet_util import *


def init_mqnet(args, nets, models, optimizers, schedulers):
    """
    Initialize MQNet model, optimizer and scheduler.
    
    Args:
        args: arguments object with training parameters
        nets: network module containing the QueryNet class
        models: dictionary of models
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        
    Returns:
        Tuple of (models, optimizers, schedulers) with MQNet added
    """
    models['mqnet'] = nets.__dict__['QueryNet'](input_size=2, inter_dim=64).to(args.device)

    optim_mqnet = torch.optim.SGD(models['mqnet'].parameters(), lr=args.lr_mqnet)
    sched_mqnet = torch.optim.lr_scheduler.MultiStepLR(
        optim_mqnet, 
        milestones=[int(args.epochs_mqnet / 2)]
    )

    optimizers['mqnet'] = optim_mqnet
    schedulers['mqnet'] = sched_mqnet
    return models, optimizers, schedulers


def mqnet_train_epoch(args, models, optimizers, criterion, delta_loader, meta_input_dict):
    """
    Training epoch for MQNet.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        optimizers: dictionary of optimizers
        criterion: loss function
        delta_loader: data loader for delta set
        meta_input_dict: dictionary of meta inputs
    """
    models['mqnet'].train()
    models['backbone'].eval()

    batch_idx = 0
    while (batch_idx < args.steps_per_epoch):
        for data in delta_loader:
            optimizers['mqnet'].zero_grad()
            inputs, labels, indexs = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

            # Get pred_scores through MQNet
            meta_inputs = torch.tensor([]).to(args.device)
            in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(args.device)
            for idx in indexs:
                meta_inputs = torch.cat((meta_inputs, meta_input_dict[idx.item()][0].reshape((-1, 2))), 0)
                in_ood_masks = torch.cat((in_ood_masks, meta_input_dict[idx.item()][1]), 0)

            pred_scores = models['mqnet'](meta_inputs)

            # Get target loss
            mask_labels = labels * in_ood_masks  # Make the label of OOD points to 0 (to calculate loss)

            out, features = models['backbone'](inputs)
            true_loss = criterion(out, mask_labels)  # Ground truth loss
            mask_true_loss = true_loss * in_ood_masks  # Make the true_loss of OOD points to 0

            loss = LossPredLoss(pred_scores, mask_true_loss.reshape((-1, 1)), margin=1)

            loss.backward()
            optimizers['mqnet'].step()

            batch_idx += 1
            if batch_idx >= args.steps_per_epoch:
                break


def mqnet_train(args, models, optimizers, schedulers, criterion, delta_loader, meta_input_dict):
    """
    Training loop for MQNet.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        criterion: loss function
        delta_loader: data loader for delta set
        meta_input_dict: dictionary of meta inputs
    """
    print('>> Train MQNet.')
    for epoch in tqdm(range(args.epochs_mqnet), leave=False, total=args.epochs_mqnet):
        mqnet_train_epoch(args, models, optimizers, criterion, delta_loader, meta_input_dict)
        schedulers['mqnet'].step()
    print('>> Finished.')


def meta_train(args, models, optimizers, schedulers, criterion, labeled_in_loader, unlabeled_loader, delta_loader):
    """
    Meta-training process for MQNet.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        criterion: loss function
        labeled_in_loader: data loader for labeled in-distribution data
        unlabeled_loader: data loader for unlabeled data
        delta_loader: data loader for delta set
        
    Returns:
        Updated models dictionary
    """
    features_in = get_labeled_features(args, models, labeled_in_loader)

    if args.mqnet_mode == 'CONF':
        informativeness, features_delta, in_ood_masks, indices = get_unlabeled_features(args, models, delta_loader)
    elif args.mqnet_mode == 'LL':
        informativeness, features_delta, in_ood_masks, indices = get_unlabeled_features_LL(args, models, delta_loader)

    purity = get_CSI_score(args, features_in, features_delta)
    assert informativeness.shape == purity.shape

    if args.mqnet_mode == 'CONF':
        meta_input = construct_meta_input(informativeness, purity)
    elif args.mqnet_mode == 'LL':
        meta_input = construct_meta_input_with_U(informativeness, purity, args, models, unlabeled_loader)

    # For enhancing training efficiency, generate meta-input & in-ood masks once, and save it into a dictionary
    meta_input_dict = {}
    for i, idx in enumerate(indices):
        meta_input_dict[idx.item()] = [meta_input[i].to(args.device), in_ood_masks[i]]

    # Mini-batch Training
    mqnet_train(args, models, optimizers, schedulers, criterion, delta_loader, meta_input_dict)

    return models
