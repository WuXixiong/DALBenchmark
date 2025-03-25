"""
Base training functionality for all methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np
from utils.general_utils import accuracy, AverageMeter


def train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch):
    """
    Standard training epoch for regular supervised learning.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        criterion: loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        writer: tensorboard SummaryWriter object
        epoch: current epoch number
        
    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    models['backbone'].train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_batches = len(dataloaders['train'])

    for i, data in enumerate(dataloaders['train']):
        inputs, labels = data[0].to(args.device), data[1].to(args.device)

        optimizers['backbone'].zero_grad()

        scores, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()

        running_loss += loss.item()

        _, preds = torch.max(scores, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            writer.add_scalar('training_loss_batch', avg_loss, epoch * total_batches + i)
            running_loss = 0.0

    epoch_loss = running_loss / total_batches
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


def train(args, models, criterion, optimizers, schedulers, dataloaders, writer=None):
    """
    Standard training loop for models that don't require special handling.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        criterion: loss function
        optimizers: dictionary of optimizers
        schedulers: dictionary of schedulers
        dataloaders: dictionary of data loaders
        writer: tensorboard SummaryWriter object (optional)
        
    Returns:
        None
    """
    if writer is None:
        log_dir = f'logs/tensorboard/{args.method}_experiment'
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
        if args.dataset in ['AGNEWS', 'IMDB', 'SST5']:  # text dataset
            epoch_loss, epoch_accuracy = train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch)
        else:
            epoch_loss, epoch_accuracy = train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch)
        
        schedulers['backbone'].step()
        writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
        writer.add_scalar('training_loss', epoch_loss, epoch)
        writer.add_scalar('accuracy', epoch_accuracy, epoch)

    writer.close()

def test(args, models, dataloaders):
    """
    Test the model on the test set.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Accuracy on the test set
    """
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    models['backbone'].eval()
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # Compute output
            with torch.no_grad():
                if args.method == 'TIDAL':
                    scores, _, _ = models['backbone'](inputs, method='TIDAL')
                else:
                    scores, _ = models['backbone'](inputs)

            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), inputs.size(0))
        print('Test acc: * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test_nlp(args, models, dataloaders):
    """
    Test NLP models on the test set.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Accuracy on the test set
    """
    top1 = AverageMeter('Acc@1', ':6.2f')
    # Switch to evaluation mode
    models['backbone'].eval()
    
    # Create a list to collect all labels
    all_labels = []
    
    with torch.no_grad():  # Only need one torch.no_grad() block
        for i, data in enumerate(dataloaders['test']):
            # Extract and move data to the correct device
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            labels = data['labels'].to(args.device)
            
            # Collect all labels
            all_labels.extend(labels.cpu().numpy().tolist())
            
            scores = models['backbone'](
                input_ids=input_ids, 
                attention_mask=attention_mask
            ).logits  # Get logits from BertForSequenceClassification
            
            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), input_ids.size(0))
    
    # Get unique labels
    unique_labels = sorted(set(all_labels))
    
    print(f'Test acc: * Prec@1 {top1.avg:.3f}')
    print(f'Unique labels in test set: {unique_labels}')
    
    return top1.avg


def test_ood(args, models, dataloaders):
    """
    Test OOD detection models on the test set.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'ood_detection' is the OOD model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Accuracy on the test set
    """
    top1 = AverageMeter('Acc@1', ':6.2f')
    # Switch to evaluate mode
    models['ood_detection'].eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            
            # Compute output
            scores, _ = models['ood_detection'](inputs)
            
            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), inputs.size(0))
    
    print(f'OOD detection acc: * Prec@1 {top1.avg:.3f}')
    return top1.avg


def test_ood_nlp(args, models, dataloaders):
    """
    Test OOD detection NLP models on the test set.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'ood_detection' is the OOD model
        dataloaders: dictionary of data loaders containing a 'test' loader
        
    Returns:
        Accuracy on the test set
    """
    top1 = AverageMeter('Acc@1', ':6.2f')
    # Switch to evaluate mode
    models['ood_detection'].eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            # Extract input_ids, attention_mask, and labels from the dictionary
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            labels = data['labels'].to(args.device)
            
            # Compute output
            outputs = models['ood_detection'](
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            scores = outputs.logits
            
            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), input_ids.size(0))
    
    print(f'OOD detection acc: * Prec@1 {top1.avg:.3f}')
    return top1.avg


def train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch):
    """
    Training epoch for NLP models (BERT/RoBERTa variants).
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models where 'backbone' is the main model
        criterion: loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        writer: tensorboard SummaryWriter object
        epoch: current epoch number
        
    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    models['backbone'] = models['backbone'].to(args.device)
    models['backbone'].train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_batches = len(dataloaders['train'])

    for i, data in enumerate(dataloaders['train']):
        # Extract input_ids, attention_mask, and labels from the dictionary
        input_ids = data['input_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)
        labels = data['labels'].to(args.device)

        # Zero the gradients
        optimizers['backbone'].zero_grad()

        # Forward pass
        outputs = models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
        scores = outputs.logits  # For BertForSequenceClassification, logits contain class probabilities

        # Compute loss
        target_loss = criterion(scores, labels)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss = m_backbone_loss

        # Backward pass and optimization
        loss.backward()
        optimizers['backbone'].step()

        # Update running loss
        running_loss += loss.item()

        # Calculate predictions and accuracy
        _, preds = torch.max(scores, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        # Log training loss every 100 batches
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            writer.add_scalar('training_loss_batch', avg_loss, epoch * total_batches + i)
            running_loss = 0.0

    # Calculate epoch metrics
    epoch_loss = running_loss / total_batches
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy