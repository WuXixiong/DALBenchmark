"""
Implementation of WNet for weighting samples in meta-learning.
Used in PAL method.
"""
import torch
import torch.nn as nn


class WNet(nn.Module):
    """
    A simple 2-layer MLP with ReLU activation and sigmoid output.
    Used to predict weights for meta-learning.
    """
    def __init__(self, input, hidden, output):
        """
        Initialize WNet.
        
        Args:
            input: input dimension
            hidden: hidden layer dimension
            output: output dimension
        """
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        """
        Forward pass of WNet.
        
        Args:
            x: input tensor
            
        Returns:
            Output tensor with sigmoid activation
        """
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


def set_Wnet(args, classes):
    """
    Initialize WNet with specified parameters and its optimizer.
    
    Args:
        args: arguments object with training parameters
        classes: number of classes for input dimension
        
    Returns:
        Tuple of (wnet, optimizer_wnet)
    """
    wnet = WNet(classes, 512, 1).to(args.device)
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in wnet.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in wnet.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_wnet = torch.optim.Adam(grouped_parameters, lr=args.lr_wnet)
    return wnet, optimizer_wnet
