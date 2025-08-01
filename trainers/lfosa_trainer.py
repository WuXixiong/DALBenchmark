"""
Training functionality for LfOSA method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.general_utils import AverageMeter


def train_epoch_lfosa(args, models, criterion, optimizers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss):
    """
    Training epoch for LfOSA method.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        criterion: base loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        criterion_xent: cross entropy loss criterion
        criterion_cent: center loss criterion
        optimizer_centloss: optimizer for center loss
    """
    models['ood_detection'].train()
    xent_losses = AverageMeter('xent_losses')
    cent_losses = AverageMeter('cent_losses')
    losses = AverageMeter('losses')

    for data in dataloaders['query']:  # use unlabeled dataset
        # Adjust temperature and labels based on ood_classes
        inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        T = torch.tensor([args.known_T] * labels.shape[0], dtype=torch.float32).to(args.device)
        for i in range(len(labels)):
            if labels[i] not in args.target_list:  # if label belong to the ood
                T[i] = args.unknown_T

        outputs, features = models['ood_detection'](inputs)
        outputs = outputs / T.unsqueeze(1)
        
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        
        optimizers['ood_detection'].zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizers['ood_detection'].step()
        
        # By doing so, weight_cent would not impact on the learning of centers
        if args.weight_cent > 0.0:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_cent)
            optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))


def train_epoch_lfosa_nlp(args, models, criterion, optimizers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss):
    """
    Training epoch for LFOSA method with NLP models.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        criterion: base loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        criterion_xent: cross entropy loss criterion
        criterion_cent: center loss criterion
        optimizer_centloss: optimizer for center loss
    """
    models['ood_detection'].train()
    xent_losses = AverageMeter('xent_losses')
    cent_losses = AverageMeter('cent_losses')
    losses = AverageMeter('losses')

    for data in dataloaders['query']:  # use unlabeled dataset
        # Extract input_ids, attention_mask, and labels from the dictionary
        input_ids = data['input_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)
        labels = data['labels'].to(args.device)

        # Zero the gradients
        optimizers['ood_detection'].zero_grad()

        # Forward pass
        outputs = models['ood_detection'](input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        features = last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        outputs = outputs.logits  # logits

        T = torch.tensor([args.known_T] * labels.shape[0], dtype=torch.float32).to(args.device)
        for i in range(len(labels)):
            if labels[i] not in args.target_list:  # if label belongs to OOD
                T[i] = args.unknown_T

        outputs = outputs / T.unsqueeze(1)

        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        
        optimizers['ood_detection'].zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizers['ood_detection'].step()
        
        # By doing so, weight_cent would not impact on the learning of centers
        if args.weight_cent > 0.0:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_cent)
            optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))
