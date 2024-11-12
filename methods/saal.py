from .almethod import ALMethod
import torch
import numpy as np
import random
import copy
import pdb
from sklearn.metrics import pairwise_distances
from scipy import stats
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

class SAAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def run(self):
        self.models['backbone'].eval()
        # 1-1) Sampling
        print('...Acquisition Only')
        # subpool_indices = random.sample(self.U_index, self.args.pool_subset)
        subpool_indices = random.sample(self.U_index, len(self.U_index))
        pool_data_dropout = []
        pool_target_dropout = []

        for idx in subpool_indices:
            data, target, _ = self.unlabeled_dst[idx]
            pool_data_dropout.append(data)
            pool_target_dropout.append(target)

        pool_data_dropout = torch.stack(pool_data_dropout)
        pool_target_dropout = torch.tensor(pool_target_dropout)

        points_of_interest = max_sharpness_acquisition_pseudo(
            pool_data_dropout, pool_target_dropout, self.args, self.models['backbone']
        )
        points_of_interest = points_of_interest.detach().cpu().numpy()

        if 'Diversity' in self.args.acqMode:
            pool_index = init_centers(points_of_interest, int(self.args.n_query))
        else:
            pool_index = points_of_interest.argsort()[::-1][:int(self.args.n_query)]

        pool_index = torch.from_numpy(pool_index)
        return pool_index.cpu().tolist(), None  # index, score

def max_sharpness_acquisition_pseudo(pool_data_dropout, pool_target_dropout, args, model):
    pool_pseudo_target_dropout = torch.zeros(pool_data_dropout.size(0), dtype=torch.long)
    original_loss = []
    max_perturbed_loss = []

    data_size = pool_data_dropout.shape[0]
    num_batch = int(np.ceil(data_size / args.pool_batch_size))

    for idx in range(num_batch):
        start_idx = idx * args.pool_batch_size
        end_idx = min((idx + 1) * args.pool_batch_size, data_size)
        batch = pool_data_dropout[start_idx:end_idx]

        output, _ = model(batch.cuda())
        softmaxed = F.softmax(output, dim=1).cpu()
        pseudo_target = softmaxed.argmax(dim=1)
        pool_pseudo_target_dropout[start_idx:end_idx] = pseudo_target
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, pseudo_target.cuda())
        original_loss.append(loss.cpu().detach())

    original_loss = torch.cat(original_loss, dim=0)

    for idx in range(num_batch):
        start_idx = idx * args.pool_batch_size
        end_idx = min((idx + 1) * args.pool_batch_size, data_size)
        batch = pool_data_dropout[start_idx:end_idx]
        pseudo_target = pool_pseudo_target_dropout[start_idx:end_idx].long()

        model_copy = copy.deepcopy(model)
        model_copy.zero_grad()

        output, _ = model_copy(batch.cuda())
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss1 = criterion(output, pseudo_target.cuda())
        loss1.mean().backward()

        norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in model_copy.parameters() if p.grad is not None]),
            p=2
        )
        scale = args.rho / (norm + 1e-12)
        with torch.no_grad():
            for p in model_copy.parameters():
                if p.grad is not None:
                    e_w = (p ** 2) * p.grad * scale
                    p.add_(e_w)

        output_updated, _ = model_copy(batch.cuda())
        loss2 = criterion(output_updated, pseudo_target.cuda())
        max_perturbed_loss.append(loss2.cpu().detach())

    max_perturbed_loss = torch.cat(max_perturbed_loss, dim=0)

    if args.acqMode == 'Max' or args.acqMode == 'Max_Diversity':
        return max_perturbed_loss
    elif args.acqMode == 'Diff' or args.acqMode == 'Diff_Diversity':
        return max_perturbed_loss - original_loss
    else:
        raise ValueError(f"Unknown acquisition mode: {args.acqMode}")

def init_centers(X, K):
    X_array = np.expand_dims(X, 1)  # Shape: (N, 1)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X_array])
    mu = [X_array[ind]]  # Initial center
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    D2 = None  # Initialize D2

    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X_array, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X_array, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X_array[ind])
        indsAll.append(ind)
        cent += 1

    # Compute gram matrix without squeezing
    gram = np.matmul(X_array[indsAll], X_array[indsAll].T)  # Shape: (K, K)

    # Now gram is a 2D array, and we can compute its eigenvalues
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]  # Not used elsewhere, consider removing if unnecessary

    return np.array(indsAll)

