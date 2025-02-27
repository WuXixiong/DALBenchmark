import torch 
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import pairwise_distances
from .almethod import ALMethod

class BADGE(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    @torch.no_grad()
    def get_grad_features(self):
        """
        Compute gradient features for all unlabeled samples using vectorized operations.
        The shape of grad_embeddings[i] is [n_class * embDim].
        """
        self.models['backbone'].eval()
        device = self.args.device
        
        if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
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
            if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
            # Extract input_ids, attention_mask, and labels from the dictionary
                input_ids = data['input_ids'].to(self.args.device)
                attention_mask = data['attention_mask'].to(self.args.device)
                outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                features = last_hidden_state[:, 0, :]
            else:
                inputs = data[0].to(device)
                # Forward pass: obtain logits and features
                logits, features = self.models['backbone'](inputs)  # logits.shape = [B, n_class], features.shape = [B, embDim]
            batch_probs = torch.softmax(logits, dim=1)          # [B, n_class]
            max_inds = torch.argmax(batch_probs, dim=1)         # [B]

            # Compute (one_hot(maxInds) - batch_probs), shape [B, n_class]
            # Outer product with features => [B, n_class, embDim]
            one_hot_max = torch.nn.functional.one_hot(max_inds, num_classes=n_class).float()
            alpha = one_hot_max - batch_probs                    # [B, n_class]
            alpha = alpha.unsqueeze(-1)                          # [B, n_class, 1]
            features = features.unsqueeze(1)                     # [B, 1, embDim]

            grad_emb_batch = alpha * features  # [B, n_class, embDim]

            if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                grad_emb_batch = grad_emb_batch.view(len(input_ids), -1)
                batch_size = len(input_ids)
            else:
                grad_emb_batch = grad_emb_batch.view(len(inputs), -1)  # [B, n_class * embDim]
                # Copy to grad_embeddings at the appropriate location
                batch_size = len(inputs)
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
        # 1) Get gradient features
        unlabeled_features = self.get_grad_features()

        # 2) Use k-means++ to select n_query centers
        selected_indices = self.k_means_plus_centers(
            X=unlabeled_features,
            K=self.args.n_query
        )

        # 3) Construct return values
        scores = [1.0] * len(selected_indices)  # Scores are all 1, not actually used
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores
    