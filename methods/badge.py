import torch 
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import pairwise_distances
from .almethod import ALMethod

class BADGE(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.is_multilabel = getattr(args, 'is_multilabel', False)

    @torch.no_grad()
    def get_grad_features(self):
        """
        Compute gradient features for all unlabeled samples using vectorized operations.
        The shape of grad_embeddings[i] is [n_class * embDim] for single-label or 
        [n_class * embDim] for multi-label (computed differently).
        """
        self.models['backbone'].eval()
        device = self.args.device
        
        if self.args.textset:
            embDim = self.models['backbone'].config.hidden_size
        else:
            embDim = self.models['backbone'].get_embedding_dim()

        n_class = self.args.num_IN_class
        num_unlabeled = len(self.U_index)

        # Create an empty tensor to store gradient features of all unlabeled data
        grad_embeddings = torch.zeros(num_unlabeled, n_class * embDim, device=device)

        unlabeled_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers
        )

        offset = 0
        for i, data in tqdm(enumerate(unlabeled_loader), total=len(unlabeled_loader), desc="Computing Grad Features", unit="batch"):
            if self.args.textset:
            # Extract input_ids, attention_mask, and labels from the dictionary
                input_ids = data['input_ids'].to(self.args.device)
                attention_mask = data['attention_mask'].to(self.args.device)
                outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                features = last_hidden_state[:, 0, :]
                batch_size = len(input_ids)
            else:
                inputs = data[0].to(device)
                # Forward pass: obtain logits and features
                logits, features = self.models['backbone'](inputs)  # logits.shape = [B, n_class], features.shape = [B, embDim]
                batch_size = len(inputs)

            if self.is_multilabel:
                # Multi-label case: use sigmoid probabilities
                batch_probs = torch.sigmoid(logits)  # [B, n_class]
                
                # For multi-label, we compute gradient based on the most confident predictions
                # Use binary approach: for each class, compute gradient assuming it's the target
                predicted_labels = (batch_probs > 0.5).float()  # [B, n_class]
                
                # Compute gradient contribution for each class independently
                # For multi-label, the gradient is based on the derivative of sigmoid
                # alpha = predicted_binary - sigmoid_prob for each class
                alpha = predicted_labels - batch_probs  # [B, n_class]
                
                # Alternative approach: use the most confident positive and negative predictions
                # This maintains the spirit of BADGE while adapting to multi-label
                max_positive_prob, max_pos_inds = batch_probs.max(dim=1)  # Most confident positive
                min_prob, min_inds = batch_probs.min(dim=1)  # Most confident negative (lowest prob)
                
                # Create a sparse representation focusing on most confident predictions
                alpha_sparse = torch.zeros_like(batch_probs)
                batch_indices = torch.arange(batch_probs.size(0), device=device)
                
                # Set gradient for most confident positive prediction
                alpha_sparse[batch_indices, max_pos_inds] = predicted_labels[batch_indices, max_pos_inds] - batch_probs[batch_indices, max_pos_inds]
                
                # Set gradient for most confident negative prediction  
                alpha_sparse[batch_indices, min_inds] = predicted_labels[batch_indices, min_inds] - batch_probs[batch_indices, min_inds]
                
                # Use the sparse version to maintain computational efficiency similar to single-label
                alpha = alpha_sparse
                
            else:
                # Single-label case: original BADGE computation
                batch_probs = torch.softmax(logits, dim=1)          # [B, n_class]
                max_inds = torch.argmax(batch_probs, dim=1)         # [B]
                
                # Compute (one_hot(maxInds) - batch_probs), shape [B, n_class]
                one_hot_max = torch.nn.functional.one_hot(max_inds, num_classes=n_class).float()
                alpha = one_hot_max - batch_probs                    # [B, n_class]

            # Outer product with features => [B, n_class, embDim]
            alpha = alpha.unsqueeze(-1)                          # [B, n_class, 1]
            features = features.unsqueeze(1)                     # [B, 1, embDim]

            grad_emb_batch = alpha * features  # [B, n_class, embDim]
            grad_emb_batch = grad_emb_batch.view(batch_size, -1)  # [B, n_class * embDim]
            
            # Copy to grad_embeddings at the appropriate location
            grad_embeddings[offset : offset + batch_size] = grad_emb_batch
            offset += batch_size

        return grad_embeddings.cpu().numpy()

    def get_grad_features_full_multilabel(self):
        """
        Alternative implementation for multi-label that computes full gradient for all classes.
        This version is more computationally expensive but theoretically more complete.
        """
        if not self.is_multilabel:
            return self.get_grad_features()
            
        self.models['backbone'].eval()
        device = self.args.device
        
        if self.args.textset:
            embDim = self.models['backbone'].config.hidden_size
        else:
            embDim = self.models['backbone'].get_embedding_dim()

        n_class = self.args.num_IN_class
        num_unlabeled = len(self.U_index)

        # For full multi-label, we might want to use all classes, not just the most confident ones
        grad_embeddings = torch.zeros(num_unlabeled, n_class * embDim, device=device)

        unlabeled_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers
        )

        offset = 0
        for i, data in tqdm(enumerate(unlabeled_loader), total=len(unlabeled_loader), desc="Computing Full Multi-label Grad Features", unit="batch"):
            if self.args.textset:
                input_ids = data['input_ids'].to(self.args.device)
                attention_mask = data['attention_mask'].to(self.args.device)
                outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                features = last_hidden_state[:, 0, :]
                batch_size = len(input_ids)
            else:
                inputs = data[0].to(device)
                logits, features = self.models['backbone'](inputs)
                batch_size = len(inputs)

            # For multi-label, compute gradient using sigmoid
            batch_probs = torch.sigmoid(logits)  # [B, n_class]
            predicted_labels = (batch_probs > 0.5).float()  # [B, n_class]
            
            # Gradient of BCE loss: predicted - true, but we use predicted as pseudo-true
            # This gives us the gradient direction that would change current predictions
            alpha = predicted_labels - batch_probs  # [B, n_class]
            
            # Weight by confidence to focus on uncertain predictions
            confidence_weights = torch.abs(batch_probs - 0.5) * 2  # [0, 1], higher for more confident
            uncertainty_weights = 1 - confidence_weights  # Higher for less confident (more uncertain)
            alpha = alpha * uncertainty_weights  # Weight by uncertainty
            
            # Outer product with features
            alpha = alpha.unsqueeze(-1)  # [B, n_class, 1]
            features = features.unsqueeze(1)  # [B, 1, embDim]
            
            grad_emb_batch = alpha * features  # [B, n_class, embDim]
            grad_emb_batch = grad_emb_batch.view(batch_size, -1)  # [B, n_class * embDim]
            
            grad_embeddings[offset : offset + batch_size] = grad_emb_batch
            offset += batch_size

        return grad_embeddings.cpu().numpy()

    def k_means_plus_centers(self, X, K):
        """
        k-means++ algorithm for selecting initial cluster centers.
        X: numpy array, shape = [N, D]
        K: number of centers to select
        """
        # First center: Select the one farthest from the origin (or mean), or randomly
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        
        # D2[i] stores the distance of sample i to the nearest selected center
        D2 = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
        centInds = np.zeros(len(X), dtype=np.int64)

        with tqdm(total=K, desc="Selecting K-means++ Centers", unit="center") as pbar:
            while len(mu) < K:
                # Update distance to the new center; replace if smaller
                if len(mu) > 1:
                    newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                    to_update = newD < D2
                    centInds[to_update] = len(mu) - 1
                    D2[to_update] = newD[to_update]

                # Compute probability distribution (D2^2) / sum(D2^2)
                D2_sq = D2 * D2
                D_dist = D2_sq / D2_sq.sum()
                
                # Sample next center according to this probability distribution
                ind = np.random.choice(len(X), p=D_dist)
                while ind in indsAll:
                    ind = np.random.choice(len(X), p=D_dist)
                
                mu.append(X[ind])
                indsAll.append(ind)
                pbar.update(1)

        return indsAll

    def select(self, **kwargs):
        # Get the appropriate gradient features based on task type
        if self.is_multilabel and hasattr(self.args, 'badge_full_multilabel') and self.args.badge_full_multilabel:
            # Use full multi-label gradient computation if specified
            unlabeled_features = self.get_grad_features_full_multilabel()
            print("Using full multi-label gradient features for BADGE")
        else:
            # Use standard gradient computation (works for both single and multi-label)
            unlabeled_features = self.get_grad_features()
            if self.is_multilabel:
                print("Using sparse multi-label gradient features for BADGE")

        # 2) Use k-means++ to select n_query centers
        selected_indices = self.k_means_plus_centers(
            X=unlabeled_features,
            K=self.args.n_query
        )

        # 3) Construct return values
        scores = [1.0] * len(selected_indices)  # Scores are all 1, not actually used
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores