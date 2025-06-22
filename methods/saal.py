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
        self.is_multilabel = getattr(args, 'is_multilabel', False)
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
        if self.args.textset:
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch_list]).to(self.args.device),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch_list]).to(self.args.device)
            }
        else:
            # Check if batch_list is already a tensor
            if isinstance(batch_list, torch.Tensor):
                return batch_list.to(self.args.device)
            # Check if it's a list of one tensor
            elif len(batch_list) == 1 and isinstance(batch_list[0], torch.Tensor):
                return batch_list[0].to(self.args.device)
            # Otherwise, try to stack as normal
            else:
                return torch.stack(batch_list).to(self.args.device)

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def run(self):
        # Set the backbone model to evaluation mode
        self.models['backbone'].eval()
        print(f'...SAAL Acquisition ({"multi-label" if self.is_multilabel else "single-label"})')

        # Optionally sample only a subset of data; the original code samples from the entire U_index
        subpool_indices = random.sample(self.U_index, self.args.pool_subset)
        
        pool_data_dropout = []
        for idx in subpool_indices:
            data = self.unlabeled_dst[idx]
            if self.args.textset:
                # Retain the complete dictionary data (including input_ids, attention_mask, etc.)
                pool_data_dropout.append(data)
            else:
                # For non-text datasets, assume data is a tuple and the first element is the input tensor
                pool_data_dropout.append(data[0].to(self.args.device))
        
        # For non-text datasets, stack the list into a tensor; for text datasets, keep as a list for batch processing
        if not self.args.textset:
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
        
        print(f"SAAL selection completed:")
        print(f"  Selected {len(pool_index)} samples")
        print(f"  Task type: {'Multi-label' if self.is_multilabel else 'Single-label'}")
        
        return pool_index.cpu().tolist(), None  # Return index and score (score is None here)

    def generate_pseudo_labels(self, logits):
        """
        Generate pseudo-labels based on task type.
        
        Args:
            logits: Model output logits
            
        Returns:
            pseudo_target: Pseudo-labels for the batch
        """
        if self.is_multilabel:
            # For multi-label: use sigmoid and threshold-based pseudo-labels
            probs = torch.sigmoid(logits)
            threshold = getattr(self.args, 'saal_multilabel_threshold', 0.5)
            pseudo_target = (probs > threshold).float()
        else:
            # For single-label: use argmax of softmax
            softmaxed = F.softmax(logits, dim=1)
            pseudo_target = softmaxed.argmax(dim=1)
        
        return pseudo_target

    def compute_loss(self, logits, target):
        """
        Compute loss based on task type.
        
        Args:
            logits: Model output logits
            target: Target labels (pseudo or true)
            
        Returns:
            loss: Computed loss tensor
        """
        if self.is_multilabel:
            # For multi-label: use BCE loss
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            target = target.float()
            loss = criterion(logits, target)
            # Sum across classes to get per-sample loss
            loss = loss.sum(dim=1)
        else:
            # For single-label: use CrossEntropy loss
            criterion = nn.CrossEntropyLoss(reduction='none')
            target = target.long()
            loss = criterion(logits, target)
        
        return loss

    def max_sharpness_acquisition_pseudo(self, pool_data_dropout, args, model):
        """
        Compute (i) the original loss and (ii) the loss after a small parameter perturbation.
        Depending on acqMode, return different scores such as 'Max' or 'Diff'.
        Modified to support both single-label and multi-label tasks.
        """
        model = model.to(self.args.device)
        # Determine the data size based on the dataset type
        if self.args.textset:
            data_size = len(pool_data_dropout)
        else:
            data_size = pool_data_dropout.shape[0]

        # Tensor to store pseudo-labels
        if self.is_multilabel:
            # For multi-label: pseudo-labels are multi-hot vectors
            num_classes = getattr(args, 'num_IN_class', args.n_class)
            pool_pseudo_target_dropout = torch.zeros(data_size, num_classes, device=self.args.device)
        else:
            # For single-label: pseudo-labels are class indices
            pool_pseudo_target_dropout = torch.zeros(data_size, dtype=torch.long, device=self.args.device)
        
        original_loss_list = []
        max_perturbed_loss_list = []

        # Process data in batches to avoid excessive GPU memory usage
        num_batch = int(np.ceil(data_size / args.pool_batch_size))

        # ---------- 1) First, compute the original loss and obtain pseudo-labels ----------
        model.eval()
        print(f"Computing original loss for SAAL ({'multi-label' if self.is_multilabel else 'single-label'})...")
        for idx in tqdm(range(num_batch), desc="Computing original loss"):
            start_idx = idx * args.pool_batch_size
            end_idx = min((idx + 1) * args.pool_batch_size, data_size)
            batch = self.collate_batch(pool_data_dropout[start_idx:end_idx])
            
            with torch.no_grad():
                if self.args.textset:
                    outputs = self.models['backbone'](input_ids=batch['input_ids'],
                                                      attention_mask=batch['attention_mask'])
                    logits = outputs.logits
                else:
                    logits, _ = self.models['backbone'](batch)
                
                # Generate pseudo-labels based on task type
                pseudo_target = self.generate_pseudo_labels(logits)
                pool_pseudo_target_dropout[start_idx:end_idx] = pseudo_target

            # Compute loss based on task type
            loss = self.compute_loss(logits, pseudo_target)
            original_loss_list.append(loss.detach())

        original_loss = torch.cat(original_loss_list, dim=0)

        # ---------- 2) Apply a small perturbation to the parameters and compute the perturbed loss ----------
        model.eval()
        print("Computing perturbed loss for SAAL...")
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
                if self.args.textset:
                    outputs = self.models['backbone'](input_ids=batch['input_ids'],
                                                      attention_mask=batch['attention_mask'])
                    logits = outputs.logits
                else:
                    logits, _ = self.models['backbone'](batch)
                
                # Compute loss for gradient calculation
                loss1 = self.compute_loss(logits, pseudo_target)
                loss1.mean().backward()

            # (c) Perturb the parameters based on the gradients
            with torch.no_grad():
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(p=2).item() ** 2
                grad_norm = grad_norm ** 0.5

                # Adjust rho for multi-label tasks if needed
                rho = args.rho
                if self.is_multilabel and hasattr(args, 'saal_multilabel_rho_scale'):
                    rho = rho * args.saal_multilabel_rho_scale

                scale = rho / (grad_norm + 1e-12)

                idx_param = 0
                for p in model.parameters():
                    if p.grad is not None:
                        e_w = (original_params[idx_param] ** 2) * p.grad * scale
                        p.add_(e_w)
                    idx_param += 1

            # (d) Compute the loss after perturbation
            with torch.no_grad():
                if self.args.textset:
                    outputs = self.models['backbone'](input_ids=batch['input_ids'],
                                                      attention_mask=batch['attention_mask'])
                    logits_updated = outputs.logits
                else:
                    logits_updated, _ = self.models['backbone'](batch)
                
                # Compute loss after perturbation
                loss2 = self.compute_loss(logits_updated, pseudo_target)
            max_perturbed_loss_list.append(loss2.detach())

            # (e) Restore the original model parameters
            with torch.no_grad():
                idx_param = 0
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.copy_(original_params[idx_param])
                        idx_param += 1

        max_perturbed_loss = torch.cat(max_perturbed_loss_list, dim=0)

        # Return scores based on acquisition mode
        if args.acqMode == 'Max' or args.acqMode == 'Max_Diversity':
            scores = max_perturbed_loss
        elif args.acqMode == 'Diff' or args.acqMode == 'Diff_Diversity':
            scores = max_perturbed_loss - original_loss
        else:
            raise ValueError(f"Unknown acquisition mode: {args.acqMode}")

        print(f"SAAL sharpness computation completed:")
        print(f"  Original loss range: [{original_loss.min().item():.4f}, {original_loss.max().item():.4f}]")
        print(f"  Perturbed loss range: [{max_perturbed_loss.min().item():.4f}, {max_perturbed_loss.max().item():.4f}]")
        print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

        return scores

    def init_centers(self, X, K):
        """
        Simplified k-center initialization for selecting representative samples when 'Diversity' is specified.
        This method works identically for both single-label and multi-label tasks.
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

        pbar = tqdm(total=K, desc="K-center init for SAAL")
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

            # Handle the case where all distances are zero
            if sum(D2) == 0.0:
                print("Warning: All distances are zero in k-center initialization")
                # Add small random noise to break ties
                D2 += np.random.rand(len(D2)) * 1e-8
                if sum(D2) == 0.0:  # Still zero, just select randomly
                    remaining_indices = [i for i in range(len(X)) if i not in indsAll]
                    if remaining_indices:
                        ind = np.random.choice(remaining_indices)
                        mu.append(X_array[ind])
                        indsAll.append(ind)
                        cent += 1
                        pbar.update(1)
                        continue
                    else:
                        break  # No more samples to select

            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            
            # Handle potential numerical issues
            if np.any(np.isnan(Ddist)) or np.any(np.isinf(Ddist)):
                print("Warning: NaN or Inf detected in distance distribution, using uniform distribution")
                Ddist = np.ones(len(D2)) / len(D2)
            
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X_array[ind])
            indsAll.append(ind)
            cent += 1
            pbar.update(1)

        pbar.close()
        
        print(f"K-center initialization completed: selected {len(indsAll)} centers")
        return np.array(indsAll)

    def analyze_pseudo_labels(self, pool_pseudo_target_dropout):
        """
        Optional method to analyze the quality of generated pseudo-labels.
        """
        if self.is_multilabel:
            # Analyze multi-label pseudo-labels
            avg_labels_per_sample = pool_pseudo_target_dropout.sum(dim=1).mean().item()
            label_frequencies = pool_pseudo_target_dropout.sum(dim=0)
            
            print(f"SAAL pseudo-label analysis (multi-label):")
            print(f"  Average labels per sample: {avg_labels_per_sample:.2f}")
            print(f"  Label frequencies: {label_frequencies.cpu().numpy()}")
            print(f"  Most frequent label: {label_frequencies.argmax().item()}")
            print(f"  Least frequent label: {label_frequencies.argmin().item()}")
        else:
            # Analyze single-label pseudo-labels
            class_counts = torch.bincount(pool_pseudo_target_dropout, minlength=self.args.n_class)
            
            print(f"SAAL pseudo-label analysis (single-label):")
            print(f"  Class distribution: {class_counts.cpu().numpy()}")
            print(f"  Most frequent class: {class_counts.argmax().item()}")
            print(f"  Least frequent class: {class_counts.argmin().item()}")