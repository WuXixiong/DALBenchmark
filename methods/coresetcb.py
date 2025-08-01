from .almethod import ALMethod 
import torch
import numpy as np
from tqdm import tqdm

class CoresetCB(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.cur_cycle = cur_cycle
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
    
    def get_features(self):
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
                prob = torch.nn.functional.softmax(unlabel_out, dim=1).cpu().numpy()
                unlabeled_probs.append(prob)
                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)
            unlabeled_probs = np.vstack(unlabeled_probs)  # Convert preds to a 2D numpy array
        return unlabeled_probs, labeled_features, unlabeled_features

    def k_center_greedy(self, labeled, unlabeled, n_query, probs):
        num_classes = self.args.num_IN_class
        if self.args.dataset == 'CIFAR10':
            lamda = 5
        elif self.args.dataset == 'CIFAR100':
            lamda = 50
        else:
            lamda = 20

        labelled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        if self.args.textset:
            labelled_classes = [labelled_subset[i]['labels'] for i in range(len(labelled_subset))]
        else:
            labelled_classes = [labelled_subset[i][1] for i in range(len(labelled_subset))]
        counts = np.bincount(labelled_classes, minlength=num_classes)
        class_threshold = int((2 * self.args.n_query + (self.cur_cycle + 1) * self.args.n_query) / num_classes)
        class_share = class_threshold - counts
        samples_share = np.array([0 if c < 0 else c for c in class_share]).reshape(num_classes, 1)

        N = len(probs)
        z = np.zeros(N, dtype=bool)
        probs = np.array(probs)

        # Initialize min_dist
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

        for i in tqdm(range(n_query)):
            # Get indices of remaining samples
            remain_indices = np.arange(N)[~z]
            N_remain = len(remain_indices)

            # Remaining probabilities
            Q_remain = probs[~z]  # Shape (N_remain, num_classes)

            # P_Z
            P_Z = probs.T @ z.astype(float)  # Shape (num_classes,)

            # X = samples_share - P_Z.reshape(-1,1) - Q_remain.T
            # samples_share has shape (num_classes, 1)
            # P_Z has shape (num_classes,)
            # Q_remain.T has shape (num_classes, N_remain)
            X = samples_share - P_Z.reshape(-1, 1) - Q_remain.T

            # Compute criterion
            mat_min_values = min_dist[~z].cpu().detach().numpy()
            criterion = -mat_min_values + (lamda / num_classes) * np.linalg.norm(X, axis=0, ord=1)

            # Select sample with minimum criterion
            q_idx = np.argmin(criterion)
            z_idx = remain_indices[q_idx]

            # Update z
            z[z_idx] = True

            # Append to greedy_indices
            greedy_indices.append(z_idx)

            # Update min_dist
            # Compute distances between selected sample and all unlabeled samples
            selected_feature = unlabeled[z_idx].unsqueeze(0)
            dist_new = torch.cdist(selected_feature, unlabeled).squeeze()

            # Update min_dist
            min_dist = torch.min(min_dist, dist_new)

        return np.array(greedy_indices)

    def select(self, **kwargs):
        unlabeled_probs, labeled_features, unlabeled_features = self.get_features()
        selected_indices = self.k_center_greedy(
            labeled_features, unlabeled_features, self.args.n_query, unlabeled_probs
        )
        scores = list(np.ones(len(selected_indices)))  # Equally assign 1 (meaningless)

        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores
    