from .almethod import ALMethod
import torch
import numpy as np

class Coreset(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        self.is_multilabel = getattr(args, 'is_multilabel', False)

    def get_features(self):
        """
        Extract features from labeled and unlabeled sets.
        This method works for both single-label and multi-label tasks since 
        it only extracts features, not predictions.
        """
        self.models['backbone'].eval()
        labeled_features, unlabeled_features = None, None
        with torch.no_grad():
            labeled_in_loader = torch.utils.data.DataLoader(
                self.labeled_in_set, 
                batch_size=self.args.test_batch_size, 
                num_workers=self.args.workers
            )
            unlabeled_loader = torch.utils.data.DataLoader(
                self.unlabeled_set, 
                batch_size=self.args.test_batch_size, 
                num_workers=self.args.workers
            )

            # Generate entire labeled_in features set
            for data in labeled_in_loader:
                if self.args.textset:
                    # Extract input_ids, attention_mask, and labels from the dictionary
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.hidden_states
                    last_hidden_state = hidden_states[-1]
                    features = last_hidden_state[:, 0, :]  # Use [CLS] token representation
                else:
                    inputs = data[0].to(self.args.device)
                    _, features = self.models['backbone'](inputs)  # features.shape = [B, embDim]

                if labeled_features is None:
                    labeled_features = features
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)

            # Generate entire unlabeled features set
            for data in unlabeled_loader:
                if self.args.textset:
                    # Extract input_ids, attention_mask, and labels from the dictionary
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.hidden_states
                    last_hidden_state = hidden_states[-1]
                    features = last_hidden_state[:, 0, :]  # Use [CLS] token representation
                else:
                    inputs = data[0].to(self.args.device)
                    _, features = self.models['backbone'](inputs)  # features.shape = [B, embDim]

                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
                    
        print(f"Coreset feature extraction completed:")
        print(f"  Labeled features shape: {labeled_features.shape}")
        print(f"  Unlabeled features shape: {unlabeled_features.shape}")
        print(f"  Multi-label mode: {self.is_multilabel}")
        
        return labeled_features, unlabeled_features

    def k_center_greedy(self, labeled, unlabeled, n_query):
        """
        K-center greedy algorithm for core-set selection.
        This algorithm is task-agnostic and works the same for both 
        single-label and multi-label tasks.
        
        Args:
            labeled: Features of labeled samples [N_labeled, feature_dim]
            unlabeled: Features of unlabeled samples [N_unlabeled, feature_dim]
            n_query: Number of samples to select
            
        Returns:
            greedy_indices: Indices of selected samples in unlabeled set
        """
        device = labeled.device
        
        # Get the minimum distances between the labeled and unlabeled examples 
        # (iteratively, to avoid memory issues)
        min_dist = torch.min(torch.cdist(labeled[0:2, :], unlabeled), 0).values
        
        # Process labeled samples in chunks to avoid memory issues
        chunk_size = 100
        for j in range(2, labeled.shape[0], chunk_size):
            if j + chunk_size < labeled.shape[0]:
                dist_matrix = torch.cdist(labeled[j:j + chunk_size, :], unlabeled)
            else:
                dist_matrix = torch.cdist(labeled[j:, :], unlabeled)
            min_dist = torch.stack((min_dist, torch.min(dist_matrix, 0).values))
            min_dist = torch.min(min_dist, 0).values

        min_dist = min_dist.reshape((1, min_dist.size(0)))
        farthest = torch.argmax(min_dist)

        greedy_indices = torch.tensor([farthest], device=device)
        
        # Iteratively select the farthest sample
        for i in range(n_query - 1):
            # Compute distance from the last selected sample to all unlabeled samples
            dist_matrix = torch.cdist(
                unlabeled[greedy_indices[-1], :].reshape((1, -1)), 
                unlabeled
            )
            min_dist = torch.stack((min_dist, dist_matrix))
            min_dist = torch.min(min_dist, 0).values

            farthest = torch.tensor([torch.argmax(min_dist)], device=device)
            greedy_indices = torch.cat((greedy_indices, farthest), 0)

        return greedy_indices.cpu().numpy()

    def select(self, **kwargs):
        """
        Select samples using core-set approach.
        This method works identically for both single-label and multi-label tasks.
        """
        # Extract features from labeled and unlabeled sets
        labeled_features, unlabeled_features = self.get_features()
        
        # Apply k-center greedy algorithm to select diverse samples
        selected_indices = self.k_center_greedy(
            labeled_features, 
            unlabeled_features, 
            self.args.n_query
        )
        
        # Create dummy scores (not used in core-set, all samples equally important)
        scores = list(np.ones(len(selected_indices)))  # equally assign 1 (meaningless)

        # Convert local indices to global indices
        Q_index = [self.U_index[idx] for idx in selected_indices]

        print(f"Coreset selection completed:")
        print(f"  Selected {len(Q_index)} samples")
        print(f"  Task type: {'Multi-label' if self.is_multilabel else 'Single-label'}")
        
        return Q_index, scores

    def get_diversity_metrics(self, labeled_features, unlabeled_features, selected_indices=None):
        """
        Optional method to compute diversity metrics for analysis.
        Useful for understanding the quality of core-set selection.
        """
        if selected_indices is None:
            return {}
            
        selected_features = unlabeled_features[selected_indices]
        
        # Compute average distance within selected set
        if len(selected_indices) > 1:
            selected_distances = torch.cdist(selected_features, selected_features)
            # Exclude diagonal (distance to self)
            mask = ~torch.eye(len(selected_indices), dtype=bool)
            avg_intra_distance = selected_distances[mask].mean().item()
        else:
            avg_intra_distance = 0.0
        
        # Compute average distance from selected to labeled set
        selected_to_labeled_dist = torch.cdist(selected_features, labeled_features)
        avg_selected_to_labeled = selected_to_labeled_dist.min(dim=1)[0].mean().item()
        
        # Compute coverage: average minimum distance from unlabeled to selected
        unlabeled_to_selected_dist = torch.cdist(unlabeled_features, selected_features)
        coverage = unlabeled_to_selected_dist.min(dim=1)[0].mean().item()
        
        metrics = {
            'avg_intra_distance': avg_intra_distance,
            'avg_selected_to_labeled': avg_selected_to_labeled,
            'coverage': coverage,
            'selected_count': len(selected_indices)
        }
        
        return metrics