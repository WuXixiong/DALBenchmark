"""
Training functionality for EOAL (End-to-end Open-set Active Learning) method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.general_utils import AverageMeter, lab_conv
from utils.loss_functions import entropic_bc_loss, reg_loss
from finch import FINCH


def train_epoch_eoal(args, models, criterion, optimizers, dataloaders, criterion_xent, O_index, cluster_centers, cluster_labels, cluster_indices):
    """
    Training epoch for EOAL method.
    
    Args:
        args: arguments object with training parameters
        models: dictionary of models
        criterion: base loss function
        optimizers: dictionary of optimizers
        dataloaders: dictionary of data loaders
        criterion_xent: cross entropy loss criterion
        O_index: indices of out-of-distribution samples
        cluster_centers: centers of clusters
        cluster_labels: labels of each cluster
        cluster_indices: indices mapping to cluster labels
    """
    models['ood_detection'].train()
    models['model_bc'].train()
    xent_losses = AverageMeter('xent_losses')
    open_losses = AverageMeter('open_losses')
    k_losses = AverageMeter('k_losses')
    losses = AverageMeter('losses')
    invalidList = O_index

    for data in dataloaders['query']:  # use unlabeled dataset
        # Adjust temperature and labels based on ood_classes
        inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        T = torch.tensor([args.known_T] * labels.shape[0], dtype=torch.float32).to(args.device)
        labels = lab_conv(args.target_list, labels)  # Convert labels to indices
        outputs, features = models['ood_detection'](inputs)
        out_open = models['model_bc'](features)  # Binary classifier
        out_open = out_open.view(features.size(0), 2, -1)
        
        labels_unk = []
        for i in range(len(labels)):
            # Annotate "unknown"
            if labels[i] not in args.target_list:
                T[i] = args.unknown_T
                tmp_idx = indexes[i]
                cluster_indices = list(cluster_indices)
                tmp_idx = int(tmp_idx)
                tmp_idx = torch.tensor(tmp_idx).to(args.device)
                tmp_idx = tmp_idx.long()
                cluster_indices = cluster_indices.long()
                cluster_indices = torch.tensor(cluster_indices).to(args.device)
                loc = torch.where(cluster_indices == tmp_idx)[0]
                loc = loc.cpu()
                labels_unk += list(np.array(cluster_labels[loc].cpu().data))
        
        labels_unk = torch.tensor(labels_unk).to(args.device)
        open_loss_pos, open_loss_neg, open_loss_pos_ood, open_loss_neg_ood = entropic_bc_loss(
            out_open, labels, args.pareta_alpha, args.num_IN_class, len(invalidList), args.w_ent
        )

        if len(invalidList) > 0:
            regu_loss_val, _, _ = reg_loss(
                features, labels, cluster_centers, labels_unk, args.num_IN_class
            )
            loss_open = 0.5 * (open_loss_pos + open_loss_neg) + 0.5 * (open_loss_pos_ood + open_loss_neg_ood)
        else:
            loss_open = 0.5 * (open_loss_pos + open_loss_neg)

        outputs = outputs / T.unsqueeze(1)
        outputs = outputs.to(args.device)
        labels = labels.to(args.device)
        loss_xent = criterion_xent(outputs, labels)
        
        if len(invalidList) > 0:
            loss = loss_xent + loss_open + args.reg_w * regu_loss_val
        else:
            loss = loss_xent + loss_open

        optimizers['ood_detection'].zero_grad()
        optimizers['model_bc'].zero_grad()
        loss.backward()
        optimizers['ood_detection'].step()
        optimizers['model_bc'].step()
        
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        open_losses.update(loss_open.item(), labels.size(0))
        if len(invalidList) > 0:
            k_losses.update(regu_loss_val.item(), labels.size(0))


def unknown_clustering(args, model, model_bc, trainloader_C, knownclass):
    """
    Perform clustering on unknown (OOD) samples.
    
    Args:
        args: arguments object with training parameters
        model: main model
        model_bc: binary classifier model
        trainloader_C: data loader for clustering
        knownclass: list of known classes
        
    Returns:
        Tuple of (cluster_centers, embeddings, cluster_labels, queryIndex)
    """
    model.eval()
    model_bc.eval()
    feat_all = torch.zeros([1, 512], device='cuda')  # originally 128
    labelArr, labelArr_true, queryIndex, y_pred = [], [], [], []

    for i, data in enumerate(trainloader_C):
        labels = data[1].to(args.device)
        index = data[2].to(args.device)
        data = data[0].to(args.device)

        labels_true = labels
        labelArr_true += list(labels_true.cpu().data.numpy())
        labels = lab_conv(knownclass, labels)
        
        outputs, features = model(data)
        softprobs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(softprobs, 1)
        y_pred += list(predicted.cpu().data.numpy())
        feat_all = torch.cat([feat_all, features.data], 0)
        queryIndex += index
        labelArr += list(labels.cpu().data.numpy())
    
    queryIndex = [tensor.cpu() for tensor in queryIndex]
    queryIndex = np.array(queryIndex)
    y_pred = np.array(y_pred)

    embeddings = feat_all[1:].cpu().numpy()
    _, _, req_c = FINCH(embeddings, req_clust=args.w_unk_cls * len(knownclass), verbose=False)
    cluster_labels = req_c
    
    # Convert back to tensors after clustering
    embeddings = torch.tensor(embeddings, device='cuda')
    labelArr_true = torch.tensor(labelArr_true)
    queryIndex = torch.tensor(queryIndex)
    cluster_labels = torch.tensor(cluster_labels)
    
    # Calculate cluster centers
    cluster_centers = calculate_cluster_centers(embeddings, cluster_labels)
    return cluster_centers, embeddings, cluster_labels, queryIndex


def calculate_cluster_centers(features, cluster_labels):
    """
    Calculate centers for each cluster.
    
    Args:
        features: feature embeddings
        cluster_labels: labels of clusters
        
    Returns:
        Tensor of cluster centers
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
