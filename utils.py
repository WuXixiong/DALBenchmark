from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
from torchlars import LARS
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Subset

from methods.methods_utils.mqnet_util import *
from methods.methods_utils.ccal_util import *
from methods.methods_utils.simclr import semantic_train_epoch
from methods.methods_utils.simclr_CSI import csi_train_epoch

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from transformers import RobertaForSequenceClassification
from transformers import DistilBertForSequenceClassification


# LFOSA
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=5, feat_dim=5, use_gpu=True): # for 4 known classes
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

# PAL
class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

class WNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

def set_Wnet(args, classes):
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

class DataLoaderX(torch.utils.data.DataLoader):
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

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

def semantic_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train a Semantic Model.')
    time_start = time.time()

    for epoch in range(args.epochs_ccal):
        semantic_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def distinctive_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train a Distinctive Model.')
    time_start = time.time()

    for epoch in range(args.epochs_ccal):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def csi_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train CSI.')
    time_start = time.time()

    for epoch in range(args.epochs_csi):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def self_sup_train(args, trial, models, optimizers, schedulers, train_dst, I_index, O_index, U_index):
    criterion = nn.CrossEntropyLoss()

    train_in_data = Subset(train_dst, I_index)
    train_ood_data = Subset(train_dst, O_index)
    train_unlabeled_data = Subset(train_dst, U_index)
    print("Self-sup training, # in: {}, # ood: {}, # unlabeled: {}".format(len(train_in_data), len(train_ood_data), len(train_unlabeled_data)))

    datalist = [train_in_data, train_ood_data, train_unlabeled_data]
    multi_datasets = torch.utils.data.ConcatDataset(datalist)

    if args.method == 'CCAL':
        # if a pre-trained CSI exist, just load it
        semantic_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_semantic_' + str(trial) + '.pt'
        distinctive_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_distinctive_' + str(trial) + '.pt'
        if os.path.isfile(semantic_path) and os.path.isfile(distinctive_path):
            print('Load pre-trained semantic, distinctive models, named: {}, {}'.format(semantic_path, distinctive_path))
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)
            models['semantic'].load_state_dict(torch.load(semantic_path))
            models['distinctive'].load_state_dict(torch.load(distinctive_path))
        else:
            contrastive_loader = torch.utils.data.DataLoader(dataset=multi_datasets, batch_size=args.ccal_batch_size, shuffle=True)
            simclr_aug = get_simclr_augmentation(args, image_size=(32, 32, 3)).to(args.device)  # for CIFAR10, 100

            # Training the Semantic Coder
            if args.data_parallel == True:
                linear = models['semantic'].module.linear
            else:
                linear = models['semantic'].linear
            linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)
            args.shift_trans_type = 'none'
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)

            semantic_train(args, models['semantic'], criterion, optimizers['semantic'], schedulers['semantic'],
                           contrastive_loader, simclr_aug, linear, linear_optim)

            # Training the Distinctive Coder
            if args.data_parallel == True:
                linear = models['distinctive'].module.linear
            else:
                linear = models['distinctive'].linear
            linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)
            args.shift_trans_type = 'rotation'
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)

            distinctive_train(args, models['distinctive'], criterion, optimizers['distinctive'], schedulers['distinctive'],
                              contrastive_loader, simclr_aug, linear, linear_optim)

            # SSL save
            if args.ssl_save == True:
                torch.save(models['semantic'].state_dict(), semantic_path)
                torch.save(models['distinctive'].state_dict(), distinctive_path)

    elif args.method == 'MQNet':
        if args.data_parallel == True:
            linear = models['csi'].module.linear
        else:
            linear = models['csi'].linear
        linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)
        args.shift_trans_type = 'rotation'
        args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
        args.shift_trans = args.shift_trans.to(args.device)

        # if a pre-trained CSI exist, just load it
        model_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_csi_'+str(trial) + '.pt'
        if os.path.isfile(model_path):
            print('Load pre-trained CSI model, named: {}'.format(model_path))
            models['csi'].load_state_dict(torch.load(model_path))
        else:
            contrastive_loader = torch.utils.data.DataLoader(dataset=multi_datasets, batch_size=args.csi_batch_size, shuffle=True)
            simclr_aug = get_simclr_augmentation(args, image_size=(32, 32, 3)).to(args.device)  # for CIFAR10, 100

            # Training CSI
            csi_train(args, models['csi'], criterion, optimizers['csi'], schedulers['csi'],
                      contrastive_loader, simclr_aug, linear, linear_optim)

            # SSL save
            if args.ssl_save == True:
                torch.save(models['csi'].state_dict(), model_path)

    return models

def mqnet_train_epoch(args, models, optimizers, criterion, delta_loader, meta_input_dict):
    models['mqnet'].train()
    models['backbone'].eval()

    batch_idx = 0
    while (batch_idx < args.steps_per_epoch):
        for data in delta_loader:
            optimizers['mqnet'].zero_grad()
            inputs, labels, indexs = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

            # get pred_scores through MQNet
            meta_inputs = torch.tensor([]).to(args.device)
            in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(args.device)
            for idx in indexs:
                meta_inputs = torch.cat((meta_inputs, meta_input_dict[idx.item()][0].reshape((-1, 2))), 0)
                in_ood_masks = torch.cat((in_ood_masks, meta_input_dict[idx.item()][1]), 0)

            pred_scores = models['mqnet'](meta_inputs)

            # get target loss
            mask_labels = labels*in_ood_masks # make the label of OOD points to 0 (to calculate loss)

            out, features = models['backbone'](inputs)
            true_loss = criterion(out, mask_labels)  # ground truth loss
            mask_true_loss = true_loss*in_ood_masks # make the true_loss of OOD points to 0

            loss = LossPredLoss(pred_scores, mask_true_loss.reshape((-1, 1)), margin=1)

            loss.backward()
            optimizers['mqnet'].step()

            batch_idx += 1

def mqnet_train(args, models, optimizers, schedulers, criterion, delta_loader, meta_input_dict):
    print('>> Train MQNet.')
    for epoch in tqdm(range(args.epochs_mqnet), leave=False, total=args.epochs_mqnet):
        mqnet_train_epoch(args, models, optimizers, criterion, delta_loader, meta_input_dict)
        schedulers['mqnet'].step()
    print('>> Finished.')

def meta_train(args, models, optimizers, schedulers, criterion, labeled_in_loader, unlabeled_loader, delta_loader):
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

def train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders):
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
            # After 120 epochs, stop the gradient from the loss prediction module
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

# def train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders):
def train_epoch_tidal(args, models, optimizers, dataloaders, epoch):
    criterion = {}
    criterion['CE'] = nn.CrossEntropyLoss(reduction='none')
    criterion['KL_Div'] = nn.KLDivLoss(reduction='batchmean')
    models['backbone'].train()
    models['module'].train()

    for data in dataloaders['train']: # , leave=False, total=len(dataloaders['train'])
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
        loss = m_backbone_loss + 1 * m_module_loss # 1.0 # lambda WEIGHT

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
    # return loss

def train_epoch_lfosa(args, models, criterion, optimizers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss):
    models['ood_detection'].train()
    xent_losses = AverageMeter('xent_losses')
    cent_losses = AverageMeter('cent_losses')
    losses = AverageMeter('losses')

    for data in dataloaders['query']: # use unlabeled dateset
        # Adjust temperature and labels based on ood_classes
        inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        T = torch.tensor([args.known_T] * labels.shape[0], dtype=torch.float32).to(args.device)
        for i in range(len(labels)):
            if labels[i] not in args.target_list: # if label belong to the ood
                T[i] = args.unknown_T

        outputs, features = models['ood_detection'](inputs)
        outputs = outputs / T.unsqueeze(1)
        # print(models['ood_detection'].linear)
        # print(f"labels min: {labels.min()}, labels max: {labels.max()}")
        # print(f"outputs shape: {outputs.shape}")
        # print(f"outputs shape: {outputs.shape}, labels shape: {labels.shape}")
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizers['ood_detection'].zero_grad() # line 261 optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizers['ood_detection'].step()
        # by doing so, weight_cent would not impact on the learning of centers
        if args.weight_cent > 0.0:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_cent)
            optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

def train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch):
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

def train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch):
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

def train_epoch_eoal(args, models, criterion, optimizers, dataloaders, criterion_xent, O_index, cluster_centers, cluster_labels, cluster_indices):
    models['ood_detection'].train()
    models['model_bc'].train()
    xent_losses = AverageMeter('xent_losses')
    open_losses = AverageMeter('open_losses')
    k_losses = AverageMeter('k_losses')
    losses = AverageMeter('losses')
    invalidList = O_index

    for data in dataloaders['query']: # use unlabeled dateset
        # Adjust temperature and labels based on ood_classes
        inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        T = torch.tensor([args.known_T] * labels.shape[0], dtype=torch.float32).to(args.device)
        labels = lab_conv(args.target_list, labels) # labels = lab_conv(knownclass, labels)
        outputs, features = models['ood_detection'](inputs) # outputs, features
        out_open = models['model_bc'](features) # orginally features not outputs, maybe because of the different network structure
        out_open = out_open.view(features.size(0), 2, -1) # outputs.size(0)
        # ent = open_entropy(out_open)
        labels_unk = []
        print(len(labels))
        for i in range(len(labels)):
            # Annotate "unknown"
            if labels[i] not in args.target_list:
                T[i] = args.unknown_T
                tmp_idx = indexes[i]
                cluster_indices = list(cluster_indices)
                tmp_idx = int(tmp_idx)
                print(cluster_indices.index(tmp_idx))
                tmp_idx = torch.tensor(tmp_idx).to(args.device)
                tmp_idx = tmp_idx.long()
                cluster_indices = cluster_indices.long()
                cluster_indices = torch.tensor(cluster_indices).to(args.device)
                loc = torch.where(cluster_indices == tmp_idx)[0]
                loc = loc.cpu()
                print(f"tmp_idx: {tmp_idx}")
                print(f"max cluster_indices: {max(cluster_indices)}")
                print(f"min cluster_indices: {min(cluster_indices)}")
                labels_unk += list(np.array(cluster_labels[loc].cpu().data)) 
        print('labels_unk???')
        print(labels_unk)
        labels_unk = torch.tensor(labels_unk).to(args.device)
        open_loss_pos, open_loss_neg, open_loss_pos_ood, open_loss_neg_ood = entropic_bc_loss(out_open, labels, args.pareta_alpha, args.num_IN_class, len(invalidList), args.w_ent)

        if len(invalidList) > 0:
            regu_loss, _, _ = reg_loss(features, labels, cluster_centers, labels_unk, args.num_IN_class)  # originally features
            loss_open = 0.5 * (open_loss_pos + open_loss_neg) + 0.5 * (open_loss_pos_ood + open_loss_neg_ood)
        else:
            loss_open = 0.5 * (open_loss_pos + open_loss_neg)

        outputs = outputs / T.unsqueeze(1)
        outputs = outputs.to(args.device)
        labels = labels.to(args.device)
        loss_xent = criterion_xent(outputs, labels)
        if len(invalidList) > 0:
            loss = loss_xent + loss_open + args.reg_w * regu_loss
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
            k_losses.update(regu_loss.item(), labels.size(0))


def train_pal_cls_epoch(args, models, optimizers, dataloaders, ema_model):
    # if not args.no_progress:
    #     p_bar = tqdm(range(args.eval_step),
    #                 disable=args.local_rank not in [-1, 0])
    # output_args["CLS"] = 'True'
    # output_args["Meta"] = 'False'
    # output_args["Hat"] = '0'
    # output_args["Wet"] = '0'
    # c_s_time = time.time()
    #for _ in range(args.eval_step):
    losses = AverageMeter("losses")
    losses_x = AverageMeter("losses_x")
    models['backbone'].train()
    for train_data in dataloaders['train']:
        # ## Data loading
        # try:
        #     feature_id, targets_x, index_x = labeled_iter.__next__()
        # except:
        #     if args.world_size > 1:
        #         labeled_epoch += 1
        #         labeled_trainloader.sampler.set_epoch(labeled_epoch)
        #     labeled_iter = iter(labeled_trainloader)
        #     feature_id, targets_x, index_x = labeled_iter.__next__()
        feature_id, targets_x, index_x = train_data

        b_size = feature_id.shape[0]
        input_l = feature_id.to(args.device)
        targets_x = targets_x.to(args.device)

        ## Feed data
        # logits, logits_open = model(input_l)
        logits, logits_open = models['backbone'](input_l)
        ## Loss for labeled samples
        Lx = F.cross_entropy(logits[:b_size],
                                targets_x, reduction='mean')
        Lo = ova_loss(logits_open[:b_size], targets_x)
        loss = Lx + Lo
        loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())

        # output_args["batch"] = batch_idx
        # output_args["loss_x"] = losses_x.avg
        # output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

        # optimizer.step()
        optimizers['backbone'].step()
        # if args.opt != 'adam':
        #     scheduler.step()
        if args.use_ema:
            ema_model.update(models['backbone'])
        models['backbone'].zero_grad()
    #     if not args.no_progress:
    #         p_bar.set_description(default_out.format(**output_args))
    #         p_bar.update()
    
    # c_e_time = time.time()

    # c_time = (c_e_time - c_s_time)/ 60
    # if not args.no_progress:
    #     p_bar.close()
    # return models['backbone'], ema_model

def train_pal_meta_epoch(args, models, optimizers, dataloaders, coef, wnet, optimizer_wnet):
    losses_hat = AverageMeter("losses_hat")
    losses_wet = AverageMeter("losses_wet")
    losses_ova = AverageMeter("losses_ova")
    meta_model = models['ood_detection']
    meta_model.train()

    # 将迭代器初始化移到循环外，避免每次循环都重新初始化
    labeled_iter = iter(dataloaders['train'])
    unlabeled_iter = iter(dataloaders['unlabeled'])

    for _ in range(args.meta_step):
        try:
            # 获取下一个 labeled 数据
            feature_id, targets_x, _ = next(labeled_iter)
        except StopIteration:
            # 重新初始化迭代器并继续
            labeled_iter = iter(dataloaders['train'])
            feature_id, targets_x, _ = next(labeled_iter)
        try:
            # 获取下一个 unlabeled 数据
            feature_al, _, _ = next(unlabeled_iter)
        except StopIteration:
            # 重新初始化迭代器并继续
            unlabeled_iter = iter(dataloaders['unlabeled'])
            feature_al, _, _ = next(unlabeled_iter)
        b_size = feature_id.shape[0]

        # 先将数据移动到设备，再进行拼接，避免在CPU上进行不必要的操作
        feature_id = feature_id.to(args.device)
        feature_al = feature_al.to(args.device)
        inputs = torch.cat([feature_id, feature_al], 0)
        input_l = feature_id  # 已经在设备上，无需再次调用 to()
        targets_x = targets_x.to(args.device)

        logits, logits_open = meta_model(inputs, method='PAL')
        logits_open_w = logits_open[b_size:]
        weight = wnet(logits_open_w)

        # 加上一个很小的 epsilon，避免除以零，并消除不必要的 if 语句
        norm = torch.sum(weight) + 1e-8
        Lx = F.cross_entropy(logits[:b_size], targets_x, reduction='mean')
        Lo = ova_loss(logits_open[:b_size], targets_x)
        losses_ova.update(Lo.item())
        L_o_u1, cost_w = ova_ent(logits_open_w)
        cost_w = cost_w.view(-1, 1)  # 使用 view 代替 reshape，提高效率

        loss_hat = Lx + coef * (torch.sum(weight * cost_w) / norm + Lo)

        meta_model.zero_grad()
        loss_hat.backward()
        optimizers['ood_detection'].step()

        losses_hat.update(loss_hat.item())
        y_l_hat, _ = meta_model(input_l, method='PAL')
        L_cls = F.cross_entropy(y_l_hat, targets_x, reduction='mean')

        # 计算上层目标
        optimizer_wnet.zero_grad()
        L_cls.backward()
        optimizer_wnet.step()
        losses_wet.update(L_cls.item())

def train(args, models, criterion, optimizers, schedulers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss, O_index, cluster_centers, cluster_labels, cluster_indices, wnet, optimizer_wnet):
    print('>> Train a Model.')
    log_dir = f'logs/tensorboard/{args.method}_experiment'
    writer = SummaryWriter(log_dir=log_dir)

    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'CCAL', 'SIMILAR', 'VAAL', 'WAAL', 'EPIG', 'EntropyCB', 'CoresetCB', 'AlphaMixSampling', 'noise_stability', 'SAAL', 'VESSAL', 'Corelog']:  # add new methods like VAAL
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            if args.dataset in ['AGNEWS']:
                epoch_loss, epoch_accuracy = train_epoch_nlp(args, models, criterion, optimizers, dataloaders, writer, epoch)
            else:
                epoch_loss, epoch_accuracy = train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch)
            schedulers['backbone'].step()
            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

        writer.close()
    
    elif args.method =='TIDAL':
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            train_epoch_tidal(args, models, optimizers, dataloaders, epoch)
            schedulers['backbone'].step()
            schedulers['module'].step()
    
    elif args.method == 'PAL':
        # ood_num = (args.num_IN_class+1)*2
        # wnet, optimizer_wnet = set_Wnet(args, ood_num)
        wnet.train()
        if args.use_ema:
            ema_model = ModelEMA(args, models['backbone'], args.ema_decay)
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            coef = math.exp(-5 * (min(1 - epoch/args.epochs, 1)) ** 2)
            train_pal_meta_epoch(args, models, optimizers, dataloaders, coef, wnet, optimizer_wnet)
            train_pal_cls_epoch(args, models, optimizers, dataloaders, ema_model)
            schedulers['backbone'].step()

    
    elif args.method in ['LFOSA', 'EOAL']: #LFOSA, EOAL
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            epoch_loss, epoch_accuracy = train_epoch(args, models, criterion, optimizers, dataloaders, writer, epoch)
            schedulers['backbone'].step()
            writer.add_scalar('learning_rate', schedulers['backbone'].get_last_lr()[0], epoch)
            writer.add_scalar('training_loss', epoch_loss, epoch)
            writer.add_scalar('accuracy', epoch_accuracy, epoch)

            if args.method == 'LFOSA':
                train_epoch_lfosa(args, models, criterion, optimizers, dataloaders, criterion_xent, criterion_cent, optimizer_centloss)
                schedulers['ood_detection'].step()
            elif args.method == 'EOAL':
                train_epoch_eoal(args, models, criterion, optimizers, dataloaders, criterion_xent, O_index, cluster_centers, cluster_labels, cluster_indices)
            schedulers['backbone'].step()
        writer.close()

    elif args.method in ['LL']: #MQNet
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders)
            schedulers['backbone'].step()
            schedulers['module'].step()

    elif args.method in ['MQNet']: #MQNet
        if args.mqnet_mode == "CONF":
            for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
                train_epoch(args, models, criterion, optimizers, dataloaders)
                schedulers['backbone'].step()
        elif args.mqnet_mode == "LL":
            for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
                train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders)
                schedulers['backbone'].step()
                schedulers['module'].step()

    print('>> Finished.')

def test(args, models, dataloaders):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    models['backbone'].eval()
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # Compute output
            with torch.no_grad():
                if args.method == 'TIDAL':
                    scores, _ , _ = models['backbone'](inputs, method = 'TIDAL')
                else:
                    scores, _ = models['backbone'](inputs)

            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), inputs.size(0))
        print('Test acc: * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_nlp(args, models, dataloaders):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluation mode
    models['backbone'].eval()

    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            # Extract and move data to the correct device
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            labels = data['labels'].to(args.device)

            # Compute output
            with torch.no_grad():
                if args.method == 'TIDAL':
                    scores, _, _ = models['backbone'](
                        input_ids=input_ids, attention_mask=attention_mask, method='TIDAL'
                    )
                else:
                    scores = models['backbone'](
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits  # Get logits from BertForSequenceClassification

            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), input_ids.size(0))

        print('Test acc: * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test_ood(args, models, dataloaders):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    models['ood_detection'].eval()
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # Compute output
            with torch.no_grad():
                scores, _ = models['ood_detection'](inputs)

            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), inputs.size(0))
        print('Test acc: * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_more_args(args):
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:' + str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'

    if args.dataset in ['CIFAR10', 'SVHN']:
        args.channel = 3
        args.im_size = (32, 32)
        #args.num_IN_class = 4

    elif args.dataset == 'MNIST':
        args.channel = 1
        args.im_size = (28, 28)
        #args.num_IN_class = 40

    elif args.dataset == 'CIFAR100':
        args.channel = 3
        args.im_size = (32, 32)
        #args.num_IN_class = 40

    elif args.dataset == 'ImageNet50':
        args.channel = 3
        args.im_size = (224, 224)
        #args.num_IN_class = 50

    elif args.dataset == 'TinyImageNet':
        args.channel = 3
        args.im_size = (64, 64)

    return args

def get_models(args, nets, model, models):
    # text dataset
    if args.model in ['DistilBert', 'Roberta']:
        if args.model == 'DistilBert':
            backbone = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=args.n_class)
        elif args.model == 'Roberta':
            backbone = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=args.n_class)
        models = {'backbone': backbone}
        return models

    # Normal
    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'VAAL', 'WAAL', 'EPIG', 'EntropyCB', 'CoresetCB', 'AlphaMixSampling', 'noise_stability', 'SAAL', 'VESSAL', 'Corelog']: # add new methods
        backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
        models = {'backbone': backbone}
    
    # TIDAL
    if args.method == 'TIDAL':
        backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)
        module = nets.tdnet.TDNet()
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            module = nets.nets_utils.MyDataParallel(module, device_ids=args.gpu)
        models = {'backbone': backbone, 'module': module}

    # SIMILAR
    elif args.method =='SIMILAR':
        backbone = nets.__dict__[model](args.channel, args.num_IN_class+1, args.im_size).to(args.device)
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
        models = {'backbone': backbone}

    # LL
    elif args.method == 'LL':
        model_ = model + '_LL'
        backbone = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        loss_module = nets.__dict__['LossNet'](args.im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            loss_module = nets.nets_utils.MyDataParallel(loss_module, device_ids=args.gpu)

        models = {'backbone': backbone, 'module': loss_module}

    # CCAL
    elif args.method == 'CCAL':
        backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)

        model_ = model+'_CSI'
        model_sem = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        model_dis = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            model_sem = nets.nets_utils.MyDataParallel(model_sem, device_ids=args.gpu)
            model_dis = nets.nets_utils.MyDataParallel(model_dis, device_ids=args.gpu)

        if models == None: #initial round
            models = {'backbone': backbone, 'semantic': model_sem, 'distinctive': model_dis}
        else:
            models['backbone'] = backbone

    # MQNet
    elif args.method == 'MQNet':
        model_ = model + '_LL'
        backbone = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        loss_module = nets.__dict__['LossNet'](args.im_size).to(args.device)

        model_ = model + '_CSI'
        model_csi = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            loss_module = nets.nets_utils.MyDataParallel(loss_module, device_ids=args.gpu)
            model_csi = nets.nets_utils.MyDataParallel(model_csi, device_ids=args.gpu)

        if models == None: #initial round
            models = {'backbone': backbone, 'module': loss_module, 'csi': model_csi} #, 'mqnet': mqnet
        else:
            models['backbone'] = backbone
            models['module'] = loss_module

    #LfOSA, EOAL
    elif args.method in ['LFOSA', 'EOAL', 'PAL']:
        backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)
        ood_detection = nets.__dict__[model](args.channel, args.num_IN_class+1, args.im_size).to(args.device) # the 1 more class for predict unknown

        if args.method == 'EOAL':
            #bc = ResClassifier_MME(num_classes=2 * (args.n_class),norm=False, input_size=128).cuda()
            model_bc = nets.eoalnet.ResClassifier_MME(num_classes=2 * (args.num_IN_class),norm=False, input_size=512).to(args.device) # original input size was 128

        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            ood_detection = nets.nets_utils.MyDataParallel(ood_detection, device_ids=args.gpu)

        models = {'backbone': backbone, 'ood_detection': ood_detection}
        if args.method == 'EOAL':
            models['model_bc'] = model_bc

    return models

def init_mqnet(args, nets, models, optimizers, schedulers):
    models['mqnet'] = nets.__dict__['QueryNet'](input_size=2, inter_dim=64).to(args.device)

    optim_mqnet = torch.optim.SGD(models['mqnet'].parameters(), lr=args.lr_mqnet)
    sched_mqnet = torch.optim.lr_scheduler.MultiStepLR(optim_mqnet, milestones=[int(args.epochs_mqnet / 2)])

    optimizers['mqnet'] = optim_mqnet
    schedulers['mqnet'] = sched_mqnet
    return models, optimizers, schedulers

def get_optim_configurations(args, models):
    print("lr: {}, momentum: {}, decay: {}".format(args.lr, args.momentum, args.weight_decay))
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(models['backbone'].parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        if args.method =='TIDAL':
            optimizer = torch.optim.SGD(models['backbone'].parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
            optim_module = torch.optim.SGD(models['module'].parameters(), lr=args.lr,
                                         momentum=args.momentum, weight_decay=args.weight_decay)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            optimizer_ood = torch.optim.SGD(models['ood_detection'].parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
            if args.method == 'EOAL':
                params_bc = list(models['model_bc'].parameters())
                optim_C = torch.optim.SGD(params_bc, lr=args.lr_model, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(models['backbone'].parameters(), args.lr, weight_decay=args.weight_decay)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            optimizer_ood = torch.optim.SGD(models['ood_detection'].parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
            if args.method == 'EOAL':
                params_bc = list(models['model_bc'].parameters())
                optim_C = torch.optim.SGD(params_bc, lr=args.lr_model, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(models['backbone'].parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](models['backbone'].parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            optimizer_ood = torch.optim.SGD(models['ood_detection'].parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
            if args.method == 'EOAL':
                params_bc = list(models['model_bc'].parameters())
                optim_C = torch.optim.SGD(params_bc, lr=args.lr_model, momentum=0.9, weight_decay=0.0005, nesterov=True)

    # LR scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            scheduler_ood = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ood, args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            scheduler_ood = torch.optim.lr_scheduler.StepLR(optimizer_ood, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            scheduler_ood = torch.optim.lr_scheduler.MultiStepLR(optimizer_ood, milestones=args.milestone)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            scheduler_ood = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer_ood)

    # Normal
    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'VAAL', 'WAAL', 'EPIG', 'EntropyCB', 'CoresetCB', 'AlphaMixSampling', 'noise_stability', 'SAAL', 'VESSAL', 'Corelog']: # also add new methods
        optimizers = {'backbone': optimizer}
        schedulers = {'backbone': scheduler}

    # LL (+ loss_pred module)
    elif args.method in ['LL', 'TIDAL']:
        optim_module = torch.optim.SGD(models['module'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sched_module = torch.optim.lr_scheduler.MultiStepLR(optim_module, milestones=args.milestone)

        optimizers = {'backbone': optimizer, 'module': optim_module}
        schedulers = {'backbone': scheduler, 'module': sched_module}

    # CCAL (+ 2 contrastive coders)
    elif args.method == 'CCAL':
        optim_sem = torch.optim.SGD(models['semantic'].parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched_sem = torch.optim.lr_scheduler.CosineAnnealingLR(optim_sem, args.epochs_ccal, eta_min=args.min_lr)
        scheduler_warmup_sem = GradualWarmupScheduler(optim_sem, multiplier=10.0, total_epoch=args.warmup, after_scheduler=sched_sem)

        optim_dis = torch.optim.SGD(models['distinctive'].parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched_dis = torch.optim.lr_scheduler.CosineAnnealingLR(optim_dis, args.epochs_ccal, eta_min=args.min_lr)
        scheduler_warmup_dis = GradualWarmupScheduler(optim_dis, multiplier=10.0, total_epoch=args.warmup, after_scheduler=sched_dis)

        optimizers = {'backbone': optimizer, 'semantic': optim_sem, 'distinctive': optim_dis}
        schedulers = {'backbone': scheduler, 'semantic': scheduler_warmup_sem, 'distinctive': scheduler_warmup_dis}

    # MQ-Net
    elif args.method == 'MQNet':
        optim_module = torch.optim.SGD(models['module'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sched_module = torch.optim.lr_scheduler.MultiStepLR(optim_module, milestones=args.milestone)

        optimizer_csi = torch.optim.SGD(models['csi'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
        optim_csi = LARS(optimizer_csi, eps=1e-8, trust_coef=0.001)

        sched_csi = torch.optim.lr_scheduler.CosineAnnealingLR(optim_csi, args.epochs_csi)
        scheduler_warmup_csi = GradualWarmupScheduler(optim_csi, multiplier=10.0, total_epoch=args.warmup, after_scheduler=sched_csi)

        optimizers = {'backbone': optimizer, 'module': optim_module, 'csi': optim_csi}
        schedulers = {'backbone': scheduler, 'module': sched_module, 'csi': scheduler_warmup_csi}
    
    # lfosa
    elif args.method in ['LFOSA', 'EOAL', 'PAL']:
        optimizers = {'backbone': optimizer, 'ood_detection': optimizer_ood}
        schedulers = {'backbone': scheduler, 'ood_detection': scheduler_ood}
        if args.method == 'EOAL':
            optimizers['model_bc'] = optim_C

    return criterion, optimizers, schedulers

# EOAL
from finch import FINCH
def unknown_clustering(args, model, model_bc, trainloader_C, knownclass):
    model.eval()
    model_bc.eval()
    feat_all = torch.zeros([1, 512], device='cuda') # originally 128
    labelArr, labelArr_true, queryIndex, y_pred = [], [], [], []

    for i, data in enumerate(trainloader_C):
        labels = data[1].to(args.device)
        index = data[2].to(args.device)
        data = data[0].to(args.device)

        labels_true = labels
        labelArr_true += list(labels_true.cpu().data.numpy())
        labels = lab_conv(knownclass, labels)
        # if use_gpu:
        #     data, labels = data.cuda(), labels.cuda()
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
    _, _, req_c = FINCH(embeddings, req_clust= args.w_unk_cls * len(knownclass), verbose=False)
    cluster_labels = req_c
    # Convert back to tensors after clustering
    embeddings = torch.tensor(embeddings, device='cuda')
    labelArr_true = torch.tensor(labelArr_true)
    queryIndex = torch.tensor(queryIndex)
    cluster_labels = torch.tensor(cluster_labels)
    cluster_centers = calculate_cluster_centers(embeddings, cluster_labels)
    return cluster_centers, embeddings, cluster_labels, queryIndex

def lab_conv(knownclass, label):
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
    unique_clusters = torch.unique(cluster_labels)
    cluster_centers = torch.zeros((len(unique_clusters), features.shape[1])).cuda()
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = torch.where(cluster_labels == cluster_id)[0]
        cluster_features = features[cluster_indices]
        # Calculate the center of the cluster using the mean of features
        cluster_center = torch.mean(cluster_features, dim=0)
        cluster_centers[i] = cluster_center
    return cluster_centers

def entropic_bc_loss(out_open, label, pareto_alpha, num_classes, query, weight):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                        out_open.size(2)+1)).cuda()  
    label_range = torch.arange(0, out_open.size(0))  
    label_p[label_range, label] = 1  
    label_n = 1 - label_p
    if query > 0:
        label_p[label==num_classes,:] = pareto_alpha/num_classes
        label_n[label==num_classes,:] = pareto_alpha/num_classes
    label_p = label_p[:,:-1]
    label_n = label_n[:,:-1]
    if (query > 0) and (weight!=0):
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[label<num_classes, 1, :]
                                                        + 1e-8) * (1 - pareto_alpha) * label_p[label<num_classes], 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[label<num_classes, 0, :] +
                                                    1e-8) * (1 - pareto_alpha) * label_n[label<num_classes], 1)[0]) ##### take max negative alone
        open_loss_pos_ood = torch.mean(torch.sum(-torch.log(out_open[label==num_classes, 1, :] +
                                                    1e-8) * label_p[label==num_classes], 1))
        open_loss_neg_ood = torch.mean(torch.sum(-torch.log(out_open[label==num_classes, 0, :] +
                                                    1e-8) * label_n[label==num_classes], 1))
        
        return open_loss_pos, open_loss_neg, open_loss_neg_ood, open_loss_pos_ood
    else:
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * (1 - 0) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                    1e-8) * (1 - 0) * label_n, 1)[0]) ##### take max negative alone
        return open_loss_pos, open_loss_neg, 0, 0
    
def reg_loss(features, labels, cluster_centers, cluster_labels, num_classes):
    features_k, _ = features[labels<num_classes], labels[labels<num_classes]
    features_u, _ = features[labels==num_classes], labels[labels==num_classes]
    k_dists = torch.cdist(features_k, cluster_centers)
    uk_dists = torch.cdist(features_u, cluster_centers)
    pk = torch.softmax(-k_dists, dim=1)
    pu = torch.softmax(-uk_dists, dim=1)

    k_ent = -torch.sum(pk*torch.log(pk+1e-20), 1)
    u_ent = -torch.sum(pu*torch.log(pu+1e-20), 1)
    true = torch.gather(uk_dists, 1, cluster_labels.long().view(-1, 1)).view(-1)
    print(f"cluster_labels size: {len(cluster_labels)}, uk_dists size: {len(uk_dists)}")

    non_gt = torch.tensor([[i for i in range(len(cluster_centers)) if cluster_labels[x] != i] for x in range(len(uk_dists))]).long().cuda()
    # non_gt = torch.tensor([[i for i in range(len(cluster_centers)) if x < len(cluster_labels) and cluster_labels[x] != i] for x in range(len(uk_dists))]).long().cuda()
    others = torch.gather(uk_dists, 1, non_gt)
    intra_loss = torch.mean(true)
    inter_loss = torch.exp(-others+true.unsqueeze(1))
    inter_loss = torch.mean(torch.log(1+torch.sum(inter_loss, dim = 1)))
    loss = 0.1*intra_loss + 1*inter_loss
    return loss, k_ent.sum(), u_ent.sum()

# PAL
def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    open_l = torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1)
    open_l_neg = torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0]
    # print(open_l.shape)
    # print(open_l_neg.shape)
    Lo = open_loss_neg + open_loss
    return Lo

def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    L_c = torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1)
    return Le, L_c