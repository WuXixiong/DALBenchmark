"""
Training functionality for TIDAL (Training Image Distribution Aware Learning) method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_epoch_tidal(args, models, optimizers, dataloaders, epoch):
    """
    Training epoch for TIDAL method.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        epoch: current epoch number
    """
    criterion = {}
    criterion['CE'] = nn.CrossEntropyLoss(reduction='none')
    criterion['KL_Div'] = nn.KLDivLoss(reduction='batchmean')
    models['backbone'].train()
    models['module'].train()

    for data in dataloaders['train']:
        with torch.cuda.device(0):
            inputs = data[0].to(args.device)
            labels = data[1].to(args.device)
            index = data[2].detach().numpy().tolist()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, emb, features = models['backbone'](inputs, method='TIDAL')
        target_loss = criterion['CE'](scores, labels)
        probs = torch.softmax(scores, dim=1)

        moving_prob = data[3].to(args.device)
        moving_prob = (moving_prob * epoch + probs * 1) / (epoch + 1)
        dataloaders['train'].dataset.moving_prob[index, :] = moving_prob.cpu().detach().numpy()

        models['module'].to(args.device)
        cumulative_logit = models['module'](features)
        m_module_loss = criterion['KL_Div'](F.log_softmax(cumulative_logit, 1), moving_prob.detach())
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss = m_backbone_loss + 1 * m_module_loss  # 1.0 # lambda WEIGHT

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
