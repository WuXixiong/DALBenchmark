"""
Training coordinator for managing different training methods.
Provides a central entry point to train different methods.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainers.base_trainer_tester import train, train_epoch, train_epoch_nlp, test, test_nlp, test_ood, test_ood_nlp
from trainers.lfosa_trainer import train_epoch_lfosa, train_epoch_lfosa_nlp
from trainers.eoal_trainer import train_epoch_eoal
from trainers.ll_trainer import train_epoch_ll
from trainers.tidal_trainer import train_epoch_tidal
from trainers.pal_trainer import train_pal_cls_epoch, train_pal_meta_epoch
from trainers.mqnet_trainer import meta_train
from trainers.ccal_trainer import self_sup_train
from utils.ema import ModelEMA
from utils.wnet import set_Wnet

def train_model(args, trial, models, criterion, optimizers, schedulers, dataloaders, criterion_xent=None, 
                criterion_cent=None, optimizer_centloss=None, O_index=None, cluster_centers=None, 
                cluster_labels=None, cluster_indices=None):
    """
    Coordinate the training process for different methods.
    This is the main entry point for training models with different methodologies.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        criterion: loss function
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        dataloaders: dictionary of data loaders
        criterion_xent: cross entropy loss (for OOD methods)
        criterion_cent: center loss (for LFOSA)
        optimizer_centloss: optimizer for center loss (for LFOSA)
        O_index: indices of out-of-distribution samples (for EOAL)
        cluster_centers: centers of clusters (for EOAL)
        cluster_labels: labels of each cluster (for EOAL)
        cluster_indices: indices mapping to cluster labels (for EOAL)
    """
    print('>> Train a Model.')
    log_dir = f'logs/tensorboard/{args.dataset}/{args.method}/{trial}_experiment'
    writer = SummaryWriter(log_dir=log_dir)

    # Standard methods
    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'CCAL', 'SIMILAR', 
                       'VAAL', 'WAAL', 'EPIG', 'EntropyCB', 'CoresetCB', 'AlphaMixSampling', 
                       'noise_stability', 'SAAL', 'VESSAL', 'corelog', 'coremse']:
        train(args, trial, models, criterion, optimizers, schedulers, dataloaders, writer)
    
    # TIDAL method
    elif args.method == 'TIDAL':
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            train_epoch_tidal(args, models, optimizers, dataloaders, epoch)
            schedulers['backbone'].step()
            schedulers['module'].step()
            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

        writer.close()
    
    # PAL method
    elif args.method == 'PAL':
        wnet, optimizer_wnet = set_Wnet(args, (args.num_IN_class+1)*2)
        wnet.train()
        
        if args.use_ema:
            ema_model = ModelEMA(args, models['backbone'], args.ema_decay)
        else:
            ema_model = None
            
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            coef = torch.exp(-5 * (min(1 - epoch/args.epochs, 1)) ** 2)
            train_pal_meta_epoch(args, models, optimizers, dataloaders, coef, wnet, optimizer_wnet)
            train_pal_cls_epoch(args, models, optimizers, dataloaders, ema_model)
            schedulers['backbone'].step()

            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

        writer.close()
    
    # LFOSA and EOAL methods
    elif args.method in ['LFOSA', 'EOAL']:
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            # Train the backbone model
            if args.textset:  # text dataset
                epoch_loss, epoch_accuracy = train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch)
            else:
                epoch_loss, epoch_accuracy = train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch)
            
            schedulers['backbone'].step()
            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

            # Train the OOD-specific part
            if args.method == 'LFOSA':
                if args.textset:  # text dataset
                    train_epoch_lfosa_nlp(args, models, criterion, optimizers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss)
                else:
                    train_epoch_lfosa(args, models, criterion, optimizers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss)
                schedulers['ood_detection'].step()
            elif args.method == 'EOAL':
                train_epoch_eoal(args, models, criterion, optimizers, dataloaders, criterion_xent, O_index, cluster_centers, cluster_labels, cluster_indices)
                schedulers['ood_detection'].step()
        
        writer.close()
    
    # Learning Loss (LL) method
    elif args.method == 'LL':
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            train_epoch_ll(args, models, epoch, criterion, optimizers, dataloaders)
            schedulers['backbone'].step()
            schedulers['module'].step()

            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

        writer.close()
    
    # MQNet method
    elif args.method == 'MQNet':
        if args.mqnet_mode == "CONF":
            for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
                if args.textset:  # text dataset
                    epoch_loss, epoch_accuracy = train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch)
                else:
                    epoch_loss, epoch_accuracy = train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch)
                schedulers['backbone'].step()
        elif args.mqnet_mode == "LL":
            for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
                train_epoch_ll(args, models, epoch, criterion, optimizers, dataloaders)
                schedulers['backbone'].step()
                schedulers['module'].step()

            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

        writer.close()

    print('>> Finished.')


def evaluate_model(args, models, dataloaders):
    """
    Evaluate the model(s) on the test set.
    
    Args:
        args: arguments object with evaluation parameters
        models: dictionary of models
        dataloaders: dictionary of data loaders
        
    Returns:
        Test accuracy
    """
    if args.textset:  # text dataset
        if 'ood_detection' in models and args.method in ['LFOSA', 'EOAL', 'PAL']:
            test_ood_nlp(args, models, dataloaders)
            return test_nlp(args, models, dataloaders)
        else:
            return test_nlp(args, models, dataloaders)
    else:
        if 'ood_detection' in models and args.method in ['LFOSA', 'EOAL', 'PAL']:
            test_ood(args, models, dataloaders)
            return test(args, models, dataloaders)
        else:
            return test(args, models, dataloaders)
