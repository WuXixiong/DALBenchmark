"""
Training functionality for LL (Learning Loss) method.
"""
import torch
from utils.loss_functions import LossPredLoss


def train_epoch_ll(args, models, epoch, criterion, optimizers, dataloaders):
    """
    Training epoch for Learning Loss method.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        epoch: current epoch number
        criterion: loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
    """
    models['backbone'].train()
    models['module'].train()

    for data in dataloaders['train']:
        inputs, labels = data[0].to(args.device), data[1].to(args.device)

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        # Classification loss for in-distribution
        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

        # loss module for predLoss
        if epoch > args.epoch_loss:
            # After epoch_loss epochs, stop the gradient from the loss prediction module
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=1)

        loss = m_backbone_loss + m_module_loss
        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
