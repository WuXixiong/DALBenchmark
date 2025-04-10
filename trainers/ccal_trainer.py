"""
Training functionality for CCAL method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from methods.methods_utils.simclr import semantic_train_epoch
from methods.methods_utils.simclr_CSI import csi_train_epoch
from methods.methods_utils.ccal_util import *


def semantic_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    """
    Training the semantic encoder.
    
    Args:
        args: arguments object with training parameters
        model: semantic model
        criterion: loss function
        optimizer: optimizer for semantic model
        scheduler: scheduler for semantic model
        loader: data loader
        simclr_aug: SimCLR augmentation function
        linear: linear classifier
        linear_optim: optimizer for linear classifier
    """
    print('>> Train a Semantic Model.')
    time_start = time.time()

    for epoch in tqdm(range(args.epochs_ccal)):
        semantic_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))


def distinctive_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    """
    Training the distinctive encoder.
    
    Args:
        args: arguments object with training parameters
        model: distinctive model
        criterion: loss function
        optimizer: optimizer for distinctive model
        scheduler: scheduler for distinctive model
        loader: data loader
        simclr_aug: SimCLR augmentation function
        linear: linear classifier
        linear_optim: optimizer for linear classifier
    """
    print('>> Train a Distinctive Model.')
    time_start = time.time()

    for epoch in tqdm(range(args.epochs_ccal)):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))


def csi_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    """
    Training CSI (Contrastive Semantic Inference).
    
    Args:
        args: arguments object with training parameters
        model: CSI model
        criterion: loss function
        optimizer: optimizer for CSI model
        scheduler: scheduler for CSI model
        loader: data loader
        simclr_aug: SimCLR augmentation function
        linear: linear classifier
        linear_optim: optimizer for linear classifier
    """
    print('>> Train CSI.')
    time_start = time.time()

    for epoch in tqdm(range(args.epochs_csi)):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))


def self_sup_train(args, trial, models, optimizers, schedulers, train_dst, I_index, O_index, U_index):
    """
    Self-supervised pretraining for CCAL and MQNet methods.
    
    Args:
        args: arguments object with training parameters
        trial: trial number for saving models
        models: dictionary of models
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        train_dst: training dataset
        I_index: indices of in-distribution samples
        O_index: indices of out-of-distribution samples
        U_index: indices of unlabeled samples
        
    Returns:
        Updated models dictionary
    """
    criterion = nn.CrossEntropyLoss()

    train_in_data = Subset(train_dst, I_index)
    train_ood_data = Subset(train_dst, O_index)
    train_unlabeled_data = Subset(train_dst, U_index)
    print("Self-sup training, # in: {}, # ood: {}, # unlabeled: {}".format(
        len(train_in_data), len(train_ood_data), len(train_unlabeled_data)))

    datalist = [train_in_data, train_ood_data, train_unlabeled_data]
    multi_datasets = torch.utils.data.ConcatDataset(datalist)

    if args.method == 'CCAL':
        # If pre-trained models exist, just load them
        semantic_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_semantic_' + str(trial) + '.pt'
        distinctive_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_distinctive_' + str(trial) + '.pt'
        
        if os.path.isfile(semantic_path) and os.path.isfile(distinctive_path):
            print('Load pre-trained semantic, distinctive models, named: {}, {}'.format(semantic_path, distinctive_path))
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)
            models['semantic'].load_state_dict(torch.load(semantic_path))
            models['distinctive'].load_state_dict(torch.load(distinctive_path))
        else:
            contrastive_loader = torch.utils.data.DataLoader(
                dataset=multi_datasets, 
                batch_size=args.ccal_batch_size, 
                shuffle=True
            )
            simclr_aug = get_simclr_augmentation(args, image_size=(32, 32, 3)).to(args.device)  # for CIFAR10, 100

            # Training the Semantic Coder
            if args.data_parallel == True:
                linear = models['semantic'].module.linear
            else:
                linear = models['semantic'].linear
            linear_optim = torch.optim.Adam(
                linear.parameters(), 
                lr=1e-3, 
                betas=(.9, .999), 
                weight_decay=args.weight_decay
            )
            args.shift_trans_type = 'none'
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)

            semantic_train(
                args, models['semantic'], criterion, optimizers['semantic'], 
                schedulers['semantic'], contrastive_loader, simclr_aug, linear, linear_optim
            )

            # Training the Distinctive Coder
            if args.data_parallel == True:
                linear = models['distinctive'].module.linear
            else:
                linear = models['distinctive'].linear
            linear_optim = torch.optim.Adam(
                linear.parameters(), 
                lr=1e-3, 
                betas=(.9, .999), 
                weight_decay=args.weight_decay
            )
            args.shift_trans_type = 'rotation'
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)

            distinctive_train(
                args, models['distinctive'], criterion, optimizers['distinctive'], 
                schedulers['distinctive'], contrastive_loader, simclr_aug, linear, linear_optim
            )

            # SSL save
            if args.ssl_save == True:
                # Create directory for semantic_path if it doesn't exist
                semantic_dir = os.path.dirname(semantic_path)
                if not os.path.exists(semantic_dir):
                    os.makedirs(semantic_dir)
                torch.save(models['semantic'].state_dict(), semantic_path)
                
                # Create directory for distinctive_path if it doesn't exist
                distinctive_dir = os.path.dirname(distinctive_path)
                if not os.path.exists(distinctive_dir):
                    os.makedirs(distinctive_dir)
                torch.save(models['distinctive'].state_dict(), distinctive_path)

    elif args.method == 'MQNet':
        if args.data_parallel == True:
            linear = models['csi'].module.linear
        else:
            linear = models['csi'].linear
        linear_optim = torch.optim.Adam(
            linear.parameters(), 
            lr=1e-3, 
            betas=(.9, .999), 
            weight_decay=args.weight_decay
        )
        args.shift_trans_type = 'rotation'
        args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
        args.shift_trans = args.shift_trans.to(args.device)

        # If a pre-trained CSI exists, just load it
        model_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_csi_'+str(trial) + '.pt'
        if os.path.isfile(model_path):
            print('Load pre-trained CSI model, named: {}'.format(model_path))
            models['csi'].load_state_dict(torch.load(model_path))
        else:
            contrastive_loader = torch.utils.data.DataLoader(
                dataset=multi_datasets, 
                batch_size=args.csi_batch_size, 
                shuffle=True
            )
            simclr_aug = get_simclr_augmentation(args, image_size=(32, 32, 3)).to(args.device)  # for CIFAR10, 100

            # Training CSI
            csi_train(
                args, models['csi'], criterion, optimizers['csi'], 
                schedulers['csi'], contrastive_loader, simclr_aug, linear, linear_optim
            )

            # SSL save
            if args.ssl_save == True:
                save_dir = os.path.dirname(model_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(models['csi'].state_dict(), model_path)

    return models
