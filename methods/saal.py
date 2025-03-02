from .almethod import ALMethod   
import torch
import numpy as np
import random
import copy
import pdb  # Consider removing or replacing pdb.set_trace() in production
from sklearn.metrics import pairwise_distances
from scipy import stats
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

class SAAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        # Randomly select only a portion of the unlabeled data
        subset_idx = np.random.choice(len(self.U_index),
                                      size=(min(self.args.subset, len(self.U_index)),),
                                      replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def collate_batch(self, batch_list):
        """
        Batch the data in batch_list according to the dataset type:
          - For text datasets: Each element in the list is a dictionary; stack each field to form a batched tensor.
          - For non-text datasets: Directly stack into a tensor.
        """
        if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch_list]).to(self.args.device),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch_list]).to(self.args.device)
            }
        else:
            return torch.stack(batch_list).to(self.args.device)

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def run(self):
        # Set the backbone model to evaluation mode
        self.models['backbone'].eval()
        print('...Acquisition Only')

        # Optionally sample only a subset of data; the original code samples from the entire U_index
        subpool_indices = random.sample(self.U_index, self.args.pool_subset)
        
        pool_data_dropout = []
        for idx in subpool_indices:
            data = self.unlabeled_dst[idx]
            if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                # Retain the complete dictionary data (including input_ids, attention_mask, etc.)
                pool_data_dropout.append(data)
            else:
                # For non-text datasets, assume data is a tuple and the first element is the input tensor
                pool_data_dropout.append(data[0].to(self.args.device))
        
        # For non-text datasets, stack the list into a tensor; for text datasets, keep as a list for batch processing
        if self.args.dataset not in ['AGNEWS', 'IMDB', 'SST5']:
            pool_data_dropout = torch.stack(pool_data_dropout)

        # Compute the acquisition scores using the maximum sharpness function
        points_of_interest = self.max_sharpness_acquisition_pseudo(
            pool_data_dropout,
            self.args,
            self.models['backbone']  # Consider using the passed-in model consistently
        )
        points_of_interest = points_of_interest.detach().cpu().numpy()

        # Post-process based on acqMode, adding diversity if specified
        if 'Diversity' in self.args.acqMode:
            pool_index = self.init_centers(points_of_interest, int(self.args.n_query))
        else:
            # Sort by scores and select the top n_query indices
            pool_index = points_of_interest.argsort()[::-1][:int(self.args.n_query)]

        pool_index = torch.from_numpy(pool_index)
        return pool_index.cpu().tolist(), None  # Return index and score (score is None here)

    def max_sharpness_acquisition_pseudo(self, pool_data_dropout, args, model):
        """
        Compute (i) the original loss and (ii) the loss after a small parameter perturbation.
        Depending on acqMode, return different scores such as 'Max' or 'Diff'.
        """
        model = model.to(self.args.device)
        # Determine the data size based on the dataset type
        if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
            data_size = len(pool_data_dropout)
        else:
            data_size = pool_data_dropout.shape[0]

        # Tensor to store pseudo-labels
        pool_pseudo_target_dropout = torch.zeros(data_size, dtype=torch.long, device=self.args.device)
        original_loss_list = []
        max_perturbed_loss_list = []

        # Process data in batches to avoid excessive GPU memory usage
        num_batch = int(np.ceil(data_size / args.pool_batch_size))
        
        # Initialize criterion once outside the loop to improve efficiency
        criterion = nn.CrossEntropyLoss(reduction='none')

        # ---------- 1) First, compute the original loss and obtain pseudo-labels ----------
        model.eval()
        for idx in tqdm(range(num_batch), desc="Computing original loss"):
            start_idx = idx * args.pool_batch_size
            end_idx = min((idx + 1) * args.pool_batch_size, data_size)
            batch = self.collate_batch(pool_data_dropout[start_idx:end_idx])
            
            with torch.no_grad():
                if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                    outputs = self.models['backbone'](input_ids=batch['input_ids'],
                                                      attention_mask=batch['attention_mask'])
                    logits = outputs.logits
                else:
                    logits, _ = self.models['backbone'](batch)
                
                softmaxed = F.softmax(logits, dim=1)
                pseudo_target = softmaxed.argmax(dim=1)
                pool_pseudo_target_dropout[start_idx:end_idx] = pseudo_target

            loss = criterion(logits, pseudo_target)
            original_loss_list.append(loss.detach())

        original_loss = torch.cat(original_loss_list, dim=0)

        # ---------- 2) Apply a small perturbation to the parameters and compute the perturbed loss ----------
        model.eval()
        for idx in tqdm(range(num_batch), desc="Computing perturbed loss"):
            start_idx = idx * args.pool_batch_size
            end_idx = min((idx + 1) * args.pool_batch_size, data_size)
            batch = self.collate_batch(pool_data_dropout[start_idx:end_idx])
            pseudo_target = pool_pseudo_target_dropout[start_idx:end_idx]

            # (a) Save the current model parameters
            original_params = [p.data.clone() for p in model.parameters() if p.requires_grad]

            # (b) Compute gradients for the batch
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                    outputs = self.models['backbone'](input_ids=batch['input_ids'],
                                                      attention_mask=batch['attention_mask'])
                    logits = outputs.logits
                else:
                    logits, _ = self.models['backbone'](batch)
                
                loss1 = criterion(logits, pseudo_target)
                loss1.mean().backward()

            # (c) Perturb the parameters based on the gradients
            with torch.no_grad():
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(p=2).item() ** 2
                grad_norm = grad_norm ** 0.5

                scale = args.rho / (grad_norm + 1e-12)

                idx_param = 0
                for p in model.parameters():
                    if p.grad is not None:
                        e_w = (original_params[idx_param] ** 2) * p.grad * scale
                        p.add_(e_w)
                    idx_param += 1

            # (d) Compute the loss after perturbation
            with torch.no_grad():
                if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                    outputs = self.models['backbone'](input_ids=batch['input_ids'],
                                                      attention_mask=batch['attention_mask'])
                    logists_updated = outputs.logits
                else:
                    logists_updated, _ = self.models['backbone'](batch)
                
                loss2 = criterion(logists_updated, pseudo_target)
            max_perturbed_loss_list.append(loss2.detach())

            # (e) Restore the original model parameters
            with torch.no_grad():
                idx_param = 0
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.copy_(original_params[idx_param])
                        idx_param += 1

        max_perturbed_loss = torch.cat(max_perturbed_loss_list, dim=0)

        if args.acqMode == 'Max' or args.acqMode == 'Max_Diversity':
            return max_perturbed_loss
        elif args.acqMode == 'Diff' or args.acqMode == 'Diff_Diversity':
            return max_perturbed_loss - original_loss
        else:
            raise ValueError(f"Unknown acquisition mode: {args.acqMode}")

    def init_centers(self, X, K):
        """
        Simplified k-center initialization for selecting representative samples when 'Diversity' is specified.
        """
        # Expand dimensions of X to shape (N, 1)
        X_array = np.expand_dims(X, 1)
        # Find the index of the sample with the maximum L2 norm
        ind = np.argmax([np.linalg.norm(s, 2) for s in X_array])
        mu = [X_array[ind]]  # Initialize centers with the selected sample
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        D2 = None

        pbar = tqdm(total=K, desc="K-center init")
        pbar.update(1)  # One center has already been initialized

        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X_array, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X_array, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]

            # Instead of breaking into debugger, consider raising an error or logging
            if sum(D2) == 0.0:
                pdb.set_trace()

            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X_array[ind])
            indsAll.append(ind)
            cent += 1
            pbar.update(1)

        pbar.close()

        # The following computation of the Gram matrix and eigenvalues is not used.
        # gram = np.matmul(X_array[indsAll], X_array[indsAll].T)  # Shape: (K, K)
        # val, _ = np.linalg.eig(gram)
        # val = np.abs(val)
        
        return np.array(indsAll)
