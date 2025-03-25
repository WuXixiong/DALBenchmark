"""
General utility functions for all modules.
"""
from argparse import ArgumentTypeError
import torch
import os
import time
from torch.utils.data.dataset import Subset
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import numpy as np
import math


def str_to_bool(v):
    """
    Handle boolean type in arguments.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_more_args(args):
    """
    Expand args with additional settings based on dataset selection.
    """
    # Setup CUDA device
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:' + str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'

    # Set dataset-specific parameters
    if args.dataset in ['CIFAR10', 'SVHN']:
        args.channel = 3
        args.im_size = (32, 32)
    elif args.dataset == 'MNIST':
        args.channel = 1
        args.im_size = (28, 28)
    elif args.dataset == 'CIFAR100':
        args.channel = 3
        args.im_size = (32, 32)
    elif args.dataset == 'ImageNet50':
        args.channel = 3
        args.im_size = (224, 224)
    elif args.dataset == 'TINYIMAGENET':
        args.channel = 3
        args.im_size = (64, 64)

    return args


class DataLoaderX(torch.utils.data.DataLoader):
    """
    DataLoader that uses BackgroundGenerator for faster data loading.
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_optimizer(args, params, lr=None, use_lars=False):
    """
    Create an optimizer according to args.optimizer, 
    with optional override of the lr parameter, and optional LARS wrapping.
    """
    if lr is None:
        lr = args.lr  # default to args.lr if not specified

    # Base optimizer
    if args.optimizer == "SGD":
        base_optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optimizer == "Adam":
        base_optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        base_optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=args.weight_decay
        )
    else:
        # Fallback to any other optimizer by name
        if 'momentum' in torch.optim.__dict__[args.optimizer].__init__.__code__.co_varnames:
            base_optimizer = torch.optim.__dict__[args.optimizer](
                params,
                lr=lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        else:
            base_optimizer = torch.optim.__dict__[args.optimizer](
                params,
                lr=lr,
                weight_decay=args.weight_decay
            )

    # Wrap with LARS if requested
    if use_lars:
        from torchlars import LARS
        optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
        return optimizer
    else:
        return base_optimizer


def create_scheduler(args, optimizer, total_epochs=None):
    """
    Create a learning-rate scheduler based on args.scheduler.
    total_epochs can be overridden for special cases (e.g., CCAL or CSI).
    """
    if total_epochs is None:
        total_epochs = args.epochs  # default to the typical total epochs

    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs, 
            eta_min=args.min_lr
        )
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.gamma
        )
    elif args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=args.milestone
        )
    else:
        # Fallback to any other scheduler by name
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)

    return scheduler


def lab_conv(knownclass, label):
    """
    Convert labels to indices within the known class list.
    Unknown classes get the label equal to len(knownclass).
    """
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label), dtype=int)
    for j in range(len(label)):
        for i in range(len(knownclass)):
            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = int(len(knownclass))     
    return label_convert


def calculate_cluster_centers(features, cluster_labels):
    """
    Calculate the center of each cluster based on feature vectors.
    """
    unique_clusters = torch.unique(cluster_labels)
    cluster_centers = torch.zeros((len(unique_clusters), features.shape[1])).cuda()
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = torch.where(cluster_labels == cluster_id)[0]
        cluster_features = features[cluster_indices]
        # Calculate the center of the cluster using the mean of features
        cluster_center = torch.mean(cluster_features, dim=0)
        cluster_centers[i] = cluster_center
    return cluster_centers
