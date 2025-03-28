import numpy as np 
import torch
import torch.nn.functional as F
import copy

from .almethod import ALMethod
from tqdm import tqdm


class noise_stability(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    @torch.no_grad()
    def add_noise_to_weights(self, model):
        """
        Add noise to the original model weights, then restore after inference outside the function.
        """
        for name, param in model.named_parameters():
            # Only add noise to weights (or specific layers)
            # For example, you can check if 'weight' in name, etc.
            if param.requires_grad and param.dim() > 1:  
                noise = torch.randn_like(param)
                scale_factor = (self.args.ns_subset * param.norm() / noise.norm()).item()
                param.add_(noise * scale_factor)

    @torch.no_grad()
    def backup_and_add_noise(self, model, backup_params):
        """
        Backup model parameters first, then add noise to the model.
        """
        for backup_p, model_p in zip(backup_params, model.parameters()):
            backup_p.copy_(model_p)  # Backup current parameters
        self.add_noise_to_weights(model)

    @torch.no_grad()
    def restore_weights(self, model, backup_params):
        """
        Restore model parameters to the backup version.
        """
        for backup_p, model_p in zip(backup_params, model.parameters()):
            model_p.copy_(backup_p)

    def run(self, **kwargs):
        # If noise is extremely small, just return random results
        if self.args.noise_scale < 1e-8:
            uncertainty = torch.randn(self.args.n_query)
            return uncertainty

        # DataLoader for inference on the unlabeled dataset
        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers
        )

        # Used to store final uncertainty scores, just for interface requirement
        uncertainty = torch.zeros(self.args.n_query).to(self.args.device)

        # First get all outputs of the original model on the unlabeled set
        use_feature = (self.args.dataset in ['house'])
        backbone = self.models['backbone']
        backbone.eval()
        outputs = self.get_all_outputs(backbone, selection_loader, use_feature)
        # Compute row norms of original outputs, used to normalize diff_k later
        # shape: [num_unlabeled, 1]
        row_norms = torch.norm(outputs, dim=1, keepdim=True)

        # Pre-backup model parameters to avoid deepcopy of entire model
        backup_params = [p.data.clone() for p in backbone.parameters()]

        # Collect list of diff_k, concatenate later
        diffs_list = []

        # In n_query iterations, add noise to the same model for inference, then restore parameters
        for _ in tqdm(range(self.args.n_query)):
            # Add noise to model parameters
            self.backup_and_add_noise(backbone, backup_params)
            # Noisy forward
            outputs_noisy = self.get_all_outputs(backbone, selection_loader, use_feature)
            # Restore
            self.restore_weights(backbone, backup_params)

            # diff_k = outputs_noisy - outputs
            diff_k = outputs_noisy - outputs
            # Normalize each row by the norm of original output
            diff_k = diff_k / row_norms
            diffs_list.append(diff_k)

        # Concatenate diff_k, final shape [num_unlabeled, n_query * out_dim]
        diffs = torch.cat(diffs_list, dim=1)

        # Use kcenter_greedy method to select K indices
        indsAll = self.kcenter_greedy(diffs, self.args.n_query)
        select_idx = [tensor.item() for tensor in indsAll]
        return select_idx, uncertainty.cpu()

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores


    def kcenter_greedy(self, X, K):
        """
        Perform k-center greedy on matrix X (shape = [N, dim]).
        Returns a list of selected sample indices of length K.
        """
        # Initialization
        N = X.shape[0]
        mu = torch.zeros(1, X.shape[1], device=X.device)
        indsAll = []
        with torch.no_grad():
            # Distance cache
            D2 = torch.cdist(X, mu).squeeze(1)  # (N,)
            while len(indsAll) < K:
                # Recalculate distance to new center and update minimum distance
                newD = torch.cdist(X, mu[-1:].detach()).squeeze(1)  # shape: [N]
                D2 = torch.min(D2, newD)
                # Find the point with the largest current distance
                ind = torch.argmax(D2)  
                # Add this point to the center set
                mu = torch.cat((mu, X[ind].unsqueeze(0)), dim=0)
                D2[ind] = 0
                indsAll.append(ind)
        return indsAll


    @torch.no_grad()
    def get_all_outputs(self, model, unlabeled_loader, use_feature=False):
        """
        Run forward inference once for all data in unlabeled_loader using the input model.
        Output either feature vectors or classification probabilities depending on use_feature.
        """
        model.eval()
        outputs = []
        for data in unlabeled_loader:
            if self.args.textset:
                # Extract input_ids and attention_mask, move to specified device
                input_ids = data['input_ids'].to(self.args.device)
                attention_mask = data['attention_mask'].to(self.args.device)
                # Get model output
                model_output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = model_output.logits
                if use_feature:
                    # If features are needed, extract the output of the first token (usually [CLS]) from the last hidden state
                    features = model_output.hidden_states[-1][:, 0, :]
                    batch_out = features
                else:
                    # Otherwise, compute classification probabilities
                    batch_out = F.softmax(logits, dim=1)
            else:
                # For other datasets, processing might differ, adjust as needed
                inputs = data[0].to(self.args.device)
                logits, features = model(inputs)
                batch_out = features if use_feature else F.softmax(logits, dim=1)
            
            outputs.append(batch_out)
        return torch.cat(outputs, dim=0)
