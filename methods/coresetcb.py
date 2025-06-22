from .almethod import ALMethod 
import torch
import numpy as np
import copy
from tqdm import tqdm

class CoresetCB(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.cur_cycle = cur_cycle
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        self.is_multilabel = getattr(args, 'is_multilabel', False)
    
    def get_features(self):
        """
        Extract features and probabilities from labeled and unlabeled sets.
        Modified to support both single-label and multi-label tasks.
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
    
            unlabeled_probs = []
            # Generate entire labeled_in features set
            for data in labeled_in_loader:
                if self.args.textset:
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.hidden_states
                    last_hidden_state = hidden_states[-1]
                    features = last_hidden_state[:, 0, :]
                else:
                    inputs = data[0].to(self.args.device)
                    _, features = self.models['backbone'](inputs)
    
                if labeled_features is None:
                    labeled_features = features
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)
    
            # Generate entire unlabeled features set
            for data in unlabeled_loader:
                if self.args.textset:
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.hidden_states
                    last_hidden_state = hidden_states[-1]
                    features = last_hidden_state[:, 0, :]
                    unlabel_out = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask).logits
                else:
                    inputs = data[0].to(self.args.device)
                    unlabel_out, features = self.models['backbone'](inputs)
                
                # Convert logits to probabilities based on task type
                if self.is_multilabel:
                    # For multi-label: use sigmoid probabilities
                    prob = torch.sigmoid(unlabel_out)
                    
                    # Optional: normalize probabilities for better class balance computation
                    if getattr(self.args, 'coresetcb_normalize_multilabel', True):
                        prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-10)
                else:
                    # For single-label: use softmax probabilities
                    prob = torch.nn.functional.softmax(unlabel_out, dim=1)
                
                prob = prob.cpu().numpy()
                unlabeled_probs.append(prob)
                
                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
                    
            unlabeled_probs = np.vstack(unlabeled_probs)  # Convert preds to a 2D numpy array
            
        print(f"CoresetCB feature extraction completed:")
        print(f"  Task type: {'Multi-label' if self.is_multilabel else 'Single-label'}")
        print(f"  Labeled features shape: {labeled_features.shape}")
        print(f"  Unlabeled features shape: {unlabeled_features.shape}")
        print(f"  Unlabeled probabilities shape: {unlabeled_probs.shape}")
        if self.is_multilabel:
            print(f"  Average probability sum per sample: {unlabeled_probs.sum(axis=1).mean():.4f}")
            
        return unlabeled_probs, labeled_features, unlabeled_features

    def get_labeled_class_counts(self):
        """
        Get class distribution in the current labeled set.
        Modified to support both single-label and multi-label tasks.
        """
        num_classes = self.args.num_IN_class
        labelled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        
        if self.is_multilabel:
            # For multi-label: count each label occurrence
            label_counts = np.zeros(num_classes)
            for i in range(len(labelled_subset)):
                if self.args.textset:
                    labels = labelled_subset[i]['labels']
                else:
                    labels = labelled_subset[i][1]
                
                # Handle different label formats
                if isinstance(labels, torch.Tensor):
                    if labels.dim() > 0 and labels.shape[0] > 1:
                        # Multi-hot encoded
                        active_labels = torch.nonzero(labels, as_tuple=True)[0].cpu().numpy()
                        label_counts[active_labels] += 1
                    else:
                        # Single label in tensor format
                        label_counts[labels.item()] += 1
                elif isinstance(labels, (list, np.ndarray)):
                    # List of active labels
                    label_counts[labels] += 1
                else:
                    # Single label
                    label_counts[labels] += 1
            
            counts = label_counts
        else:
            # For single-label: standard bincount
            if self.args.textset:
                labelled_classes = [labelled_subset[i]['labels'] for i in range(len(labelled_subset))]
            else:
                labelled_classes = [labelled_subset[i][1] for i in range(len(labelled_subset))]
            counts = np.bincount(labelled_classes, minlength=num_classes)
            
        return counts

    def k_center_greedy(self, labeled, unlabeled, n_query, probs):
        """
        K-center greedy algorithm with class balance.
        Modified to support both single-label and multi-label tasks.
        """
        num_classes = self.args.num_IN_class
        
        # Set lambda parameter based on dataset
        if self.args.dataset == 'CIFAR10':
            lamda = 5
        elif self.args.dataset == 'CIFAR100':
            lamda = 50
        elif self.args.dataset == 'RCV1' and self.is_multilabel:
            # Special handling for RCV1 multi-label dataset
            lamda = 10  # Adjusted for multi-label characteristics
        else:
            lamda = 20

        # Get current class distribution
        counts = self.get_labeled_class_counts()
        
        # Calculate class balance targets
        class_threshold = int((2 * self.args.n_query + (self.cur_cycle + 1) * self.args.n_query) / num_classes)
        class_share = class_threshold - counts
        
        if self.is_multilabel:
            # For multi-label: allow for multiple active labels per sample
            # Adjust the sharing strategy to account for label co-occurrence
            samples_share = np.array([max(0, c) for c in class_share]).reshape(num_classes, 1)
            
            # Scale down the sharing for multi-label to account for multiple labels per sample
            multilabel_scale = getattr(self.args, 'coresetcb_multilabel_scale', 0.5)
            samples_share = samples_share * multilabel_scale
        else:
            # For single-label: standard sharing
            samples_share = np.array([0 if c < 0 else c for c in class_share]).reshape(num_classes, 1)

        N = len(probs)
        z = np.zeros(N, dtype=bool)
        probs = np.array(probs)

        # Initialize min_dist (distance-based coreset component)
        if labeled is None or labeled.shape[0] == 0:
            min_dist = torch.full((unlabeled.shape[0],), float('inf'), device=unlabeled.device)
        else:
            batch_size = 100
            min_dist = torch.full((unlabeled.shape[0],), float('inf'), device=unlabeled.device)
            for j in range(0, labeled.shape[0], batch_size):
                batch_labeled = labeled[j:j+batch_size, :]
                dist_matrix = torch.cdist(batch_labeled, unlabeled)
                min_dist = torch.min(min_dist, torch.min(dist_matrix, dim=0).values)

        greedy_indices = []

        print(f"CoresetCB selection starting:")
        print(f"  Class distribution: {counts}")
        print(f"  Target distribution: {class_threshold}")
        print(f"  Class sharing needs: {class_share}")
        print(f"  Lambda parameter: {lamda}")

        for i in tqdm(range(n_query), desc="CoresetCB Selection"):
            # Get indices of remaining samples
            remain_indices = np.arange(N)[~z]
            N_remain = len(remain_indices)

            # Remaining probabilities
            Q_remain = probs[~z]  # Shape (N_remain, num_classes)

            # P_Z: cumulative probability of selected samples
            if self.is_multilabel:
                # For multi-label: sum probabilities of selected samples
                P_Z = probs.T @ z.astype(float)  # Shape (num_classes,)
            else:
                # For single-label: standard computation
                P_Z = probs.T @ z.astype(float)  # Shape (num_classes,)

            # Compute class balance term
            # X = samples_share - P_Z.reshape(-1,1) - Q_remain.T
            # samples_share has shape (num_classes, 1)
            # P_Z has shape (num_classes,)
            # Q_remain.T has shape (num_classes, N_remain)
            X = samples_share - P_Z.reshape(-1, 1) - Q_remain.T

            # Compute selection criterion: diversity + class balance
            mat_min_values = min_dist[~z].cpu().detach().numpy()
            
            if self.is_multilabel:
                # For multi-label: use different norm for class balance
                class_balance_term = (lamda / num_classes) * np.linalg.norm(X, axis=0, ord=2)  # L2 norm
            else:
                # For single-label: use L1 norm
                class_balance_term = (lamda / num_classes) * np.linalg.norm(X, axis=0, ord=1)  # L1 norm
            
            criterion = -mat_min_values + class_balance_term

            # Select sample with minimum criterion
            q_idx = np.argmin(criterion)
            z_idx = remain_indices[q_idx]

            # Update z
            z[z_idx] = True

            # Append to greedy_indices
            greedy_indices.append(z_idx)

            # Update min_dist for coreset component
            # Compute distances between selected sample and all unlabeled samples
            selected_feature = unlabeled[z_idx].unsqueeze(0)
            dist_new = torch.cdist(selected_feature, unlabeled).squeeze()

            # Update min_dist
            min_dist = torch.min(min_dist, dist_new)

        print(f"CoresetCB selection completed: selected {len(greedy_indices)} samples")
        return np.array(greedy_indices)

    def select(self, **kwargs):
        """
        Main selection method that combines coreset and class balance.
        """
        unlabeled_probs, labeled_features, unlabeled_features = self.get_features()
        selected_indices = self.k_center_greedy(
            labeled_features, unlabeled_features, self.args.n_query, unlabeled_probs
        )
        scores = list(np.ones(len(selected_indices)))  # Equally assign 1 (meaningless)

        Q_index = [self.U_index[idx] for idx in selected_indices]
        
        # Optional: analyze selection quality
        if getattr(self.args, 'coresetcb_verbose', False):
            self.analyze_selection(unlabeled_probs, selected_indices)
        
        return Q_index, scores
    
    def analyze_selection(self, unlabeled_probs, selected_indices):
        """
        Optional method to analyze the quality of selection.
        """
        selected_probs = unlabeled_probs[selected_indices]
        
        if self.is_multilabel:
            # Analyze label distribution in selected samples
            label_coverage = (selected_probs > 0.5).sum(axis=0)
            print(f"CoresetCB selection analysis:")
            print(f"  Label coverage: {label_coverage}")
            print(f"  Average labels per selected sample: {(selected_probs > 0.5).sum(axis=1).mean():.2f}")
        else:
            # Analyze class distribution in selected samples
            predicted_classes = selected_probs.argmax(axis=1)
            class_counts = np.bincount(predicted_classes, minlength=self.args.num_IN_class)
            print(f"CoresetCB selection analysis:")
            print(f"  Predicted class distribution: {class_counts}")
            print(f"  Class balance variance: {class_counts.var():.2f}")
    
    def get_diversity_metrics(self, unlabeled_features, selected_indices):
        """
        Optional method to compute diversity metrics.
        """
        if len(selected_indices) < 2:
            return {}
            
        selected_features = unlabeled_features[selected_indices]
        
        # Compute pairwise distances
        distances = torch.cdist(selected_features, selected_features)
        mask = ~torch.eye(len(selected_indices), dtype=bool)
        
        avg_distance = distances[mask].mean().item()
        min_distance = distances[mask].min().item()
        
        return {
            'avg_pairwise_distance': avg_distance,
            'min_pairwise_distance': min_distance,
            'selected_count': len(selected_indices)
        }