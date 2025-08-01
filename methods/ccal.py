"""
This implementation of the CCAL active learning algorithm is adapted from:
- The MQNet repository (https://github.com/kaist-dmlab/MQNet), which provides an implementation of CCAL and other methods.

MQNet's implementation is based on:
- Official CCAL code: https://github.com/RUC-DWBI-ML/CCAL

@InProceedings{Du_2021_ICCV,
    author    = {Du, Pan and Zhao, Suyun and Chen, Hui and Chai, Shuwen and Chen, Hong and Li, Cuiping},
    title     = {Contrastive Coding for Active Learning Under Class Distribution Mismatch},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8927-8936}
}
"""
from .almethod import ALMethod
import torch
import numpy as np
from .methods_utils.ccal_eval import *
from .methods_utils.ccal_util import *

class CCAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, O_index, dataloaders, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.O_index = O_index
        self.dataloaders = dataloaders

    def get_label_i_loader(self):
        # Measure uncertainty of each utils points in the subset
        label_i_index = [[] for i in range(self.args.num_IN_class)]
        if self.args.textset:
            for i in self.I_index:
                for k in range(self.args.num_IN_class):
                    if self.unlabeled_dst[i]["labels"].item() == k: # label
                        label_i_index[k].append(i)
        else:
            for i in self.I_index:
                for k in range(self.args.num_IN_class):
                    if self.unlabeled_dst[i][1] == k:
                        label_i_index[k].append(i)

        label_i_loader = []
        for i in range(len(label_i_index)):
            sampler_label_i = torch.utils.data.sampler.SubsetRandomSampler(label_i_index[i])  # make indices initial to the samples
            label_loader_i =  torch.utils.data.DataLoader(self.unlabeled_dst, sampler=sampler_label_i, batch_size=self.args.test_batch_size,
                                        pin_memory=True)
            label_i_loader.append(label_loader_i)
        return label_i_loader

    def select(self, **kwargs):
        # subset selection (if needed, to avoid out of memory)
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

        label_i_loader = self.get_label_i_loader()

        unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_subset, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        simclr_aug = get_simclr_augmentation(self.args, image_size=(32, 32, 3)).to(self.args.device)  # for CIFAR10, 100
        query_idx, subset_idx = eval_unlabeled_detection(self.args, self.models, unlabeled_loader, self.dataloaders['train'],
                                                        label_i_loader, simclr_aug=simclr_aug)

        Q_index = list(np.array(self.U_index_sub)[subset_idx])
        scores = list(np.ones(len(subset_idx)))  # equally assign 1 (meaningless)

        return Q_index, scores