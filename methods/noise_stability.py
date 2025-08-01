# from .almethod import ALMethod
# import torch
# import numpy as np
# import copy
# import torch.nn.functional as F
# from tqdm import tqdm

# class noise_stability(ALMethod):
#     def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
#         super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
#         self.noise_scale = kwargs.get('noise_scale', 0.01)
#         self.n_sampling = args.noise_sampling
#         self.addendum = args.n_query

#     def run(self):
#         scores = self.rank_uncertainty()
#         # Convert non-zero elements to indices
#         selection_result = np.where(scores > 0)[0]
#         return selection_result, scores

#     def add_noise_to_weights(self, m):
#         with torch.no_grad():
#             if hasattr(m, 'weight'):
#                 noise = torch.randn(m.weight.size())
#                 noise = noise.to(self.args.device)
#                 noise *= (self.noise_scale * m.weight.norm() / noise.norm())
#                 m.weight.add_(noise)

#     def get_all_outputs(self, model, loader, use_feature=False):
#         model.eval()
#         outputs = []  # Use list to collect outputs
#         with torch.no_grad():
#             for data in loader:
#                 if self.args.textset:
#                     input_ids = data['input_ids'].to(self.args.device)
#                     attention_mask = data['attention_mask'].to(self.args.device)
                    
#                     output = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
#                     logits = output.logits
#                     hidden_states = output.hidden_states
#                     last_hidden_state = hidden_states[-1]
#                     fea = last_hidden_state[:, 0, :]
                    
#                     if use_feature:
#                         out = fea
#                     else:
#                         out = F.softmax(logits, dim=1)
                        
#                 else:
#                     inputs = data[0].to(self.args.device)
#                     out, fea = model(inputs)
                    
#                     if use_feature:
#                         out = fea
#                     else:
#                         out = F.softmax(out, dim=1)
                
#                 outputs.append(out)  # Actually collect the outputs
        
#         # Concatenate all outputs into a single tensor
#         if outputs:
#             return torch.cat(outputs, dim=0)
#         else:
#             # Return empty tensor with proper shape if no data
#             return torch.tensor([]).to(self.args.device)

#     def kcenter_greedy(self, X, K):
#         # Check if X is empty or has wrong dimensions
#         if X.numel() == 0:
#             print("Warning: Empty tensor passed to kcenter_greedy")
#             return []
        
#         if len(X.shape) != 2:
#             print(f"Warning: Expected 2D tensor, got shape {X.shape}")
#             return []
        
#         if X.shape[0] < K:
#             print(f"Warning: Requested {K} samples but only {X.shape[0]} available")
#             K = X.shape[0]
        
#         avg_norm = np.mean([torch.norm(X[i]).item() for i in range(X.shape[0])])
#         mu = torch.zeros(1, X.shape[1]).to(self.args.device)
#         indsAll = []
        
#         with tqdm(total=K) as pbar:
#             while len(indsAll) < K:
#                 if len(indsAll) == 0:
#                     D2 = torch.cdist(X, mu).squeeze(1)
#                 else:
#                     newD = torch.cdist(X, mu[-1:])
#                     newD = torch.min(newD, dim=1)[0]
#                     for i in range(X.shape[0]):
#                         if D2[i] > newD[i]:
#                             D2[i] = newD[i]
                
#                 for i, ind in enumerate(D2.topk(1)[1]):
#                     D2[ind] = 0
#                     mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
#                     indsAll.append(ind.item())  # Convert to int
                
#                 # update tqdm bar
#                 pbar.update(1)
        
#         selected_norm = np.mean([torch.norm(X[i]).item() for i in indsAll])
#         return indsAll

#     def rank_uncertainty(self):
#         print("| Calculating noise stability sampling uncertainty")
#         selection_loader = torch.utils.data.DataLoader(
#             self.unlabeled_set, 
#             batch_size=self.args.test_batch_size, 
#             num_workers=self.args.workers
#         )
        
#         if self.noise_scale < 1e-8:
#             uncertainty = torch.randn(len(self.unlabeled_set))
#             return uncertainty.cpu().numpy()
        
#         uncertainty = torch.zeros(len(self.unlabeled_set)).to(self.args.device)
#         use_feature = self.args.dataset in ['house']
        
#         # Get original outputs
#         outputs = self.get_all_outputs(self.models['backbone'], selection_loader, use_feature)
        
#         # Check if outputs is empty
#         if outputs.numel() == 0:
#             print("Warning: No outputs obtained from model")
#             return torch.zeros(len(self.unlabeled_set)).cpu().numpy()
        
#         diffs = []  # Use list to collect differences
        
#         print("| Running noise stability sampling with", self.n_sampling, "iterations")
#         for i in tqdm(range(self.n_sampling)):
#             noisy_model = copy.deepcopy(self.models['backbone'])
#             noisy_model.eval()
#             noisy_model.apply(self.add_noise_to_weights)
#             outputs_noisy = self.get_all_outputs(noisy_model, selection_loader, use_feature)
            
#             if outputs_noisy.numel() == 0:
#                 print(f"Warning: No outputs from noisy model in iteration {i}")
#                 continue
                
#             diff_k = outputs_noisy - outputs
            
#             # Normalize differences
#             for j in range(diff_k.shape[0]):
#                 norm_val = outputs[j].norm()
#                 if norm_val > 0:  # Avoid division by zero
#                     diff_k[j,:] /= norm_val
            
#             diffs.append(diff_k)
        
#         if not diffs:
#             print("Warning: No valid differences computed")
#             return torch.zeros(len(self.unlabeled_set)).cpu().numpy()
        
#         # Concatenate all differences
#         diffs_tensor = torch.cat(diffs, dim=1)
        
#         print("| Applying k-center greedy algorithm to select diverse samples")
#         indsAll = self.kcenter_greedy(diffs_tensor, self.addendum)
        
#         for ind in indsAll:
#             uncertainty[ind] = 1
            
#         return uncertainty.cpu().numpy()

#     def select(self, **kwargs):
#         selected_indices, scores = self.run()
#         Q_index = [self.U_index[idx] for idx in selected_indices]
#         return Q_index, scores

from .almethod import ALMethod
import torch
import numpy as np
import copy
import torch.nn.functional as F
from tqdm import tqdm

class noise_stability(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.noise_scale = kwargs.get('noise_scale', 0.01)
        self.n_sampling = args.noise_sampling
        self.addendum = args.n_query

    def run(self):
        scores = self.rank_uncertainty()
        # Convert non-zero elements to indices
        selection_result = np.where(scores > 0)[0]
        return selection_result, scores

    def add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                noise = torch.randn(m.weight.size())
                noise = noise.to(self.args.device)
                noise *= (self.noise_scale * m.weight.norm() / noise.norm())
                m.weight.add_(noise)
                # print('scale', 1.0 * noise.norm() / m.weight.norm(), 'weight', m.weight.view(-1)[:10])

    def get_all_outputs(self, model, loader, use_feature=False):
        model.eval()
        outputs = torch.tensor([]).to(self.args.device)
        with torch.no_grad():
            for data in loader:
                inputs = data[0].to(self.args.device)
                out, fea = model(inputs) 
                if use_feature:
                    out = fea
                else:
                    out = F.softmax(out, dim=1)
                outputs = torch.cat((outputs, out), dim=0)
        return outputs

    def kcenter_greedy(self, X, K):
        avg_norm = np.mean([torch.norm(X[i]).item() for i in range(X.shape[0])])
        mu = torch.zeros(1, X.shape[1]).to(self.args.device)
        indsAll = []
        with tqdm(total=K) as pbar:
            while len(indsAll) < K:
                if len(indsAll) == 0:
                    D2 = torch.cdist(X, mu).squeeze(1)
                else:
                    newD = torch.cdist(X, mu[-1:])
                    newD = torch.min(newD, dim=1)[0]
                    for i in range(X.shape[0]):
                        if D2[i] > newD[i]:
                            D2[i] = newD[i]
                for i, ind in enumerate(D2.topk(1)[1]):
                    # if i == 0:
                    #     print(len(indsAll), ind.item(), D2[ind].item(), X[ind,:5])
                    D2[ind] = 0
                    mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
                    indsAll.append(ind)
                
                # update tqdm bar
                pbar.update(1)
        
        selected_norm = np.mean([torch.norm(X[i]).item() for i in indsAll])
        return indsAll

    def rank_uncertainty(self):
        print("| Calculating noise stability sampling uncertainty")
        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size, 
            num_workers=self.args.workers
        )
        
        if self.noise_scale < 1e-8:
            uncertainty = torch.randn(len(self.unlabeled_set))
            return uncertainty.cpu().numpy()
        
        uncertainty = torch.zeros(len(self.unlabeled_set)).to(self.args.device)
        diffs = torch.tensor([]).to(self.args.device)
        use_feature = self.args.dataset in ['house']
        outputs = self.get_all_outputs(self.models['backbone'], selection_loader, use_feature)
        
        print("| Running noise stability sampling with", self.n_sampling, "iterations")
        for i in tqdm(range(self.n_sampling)):
            # print(f"| Sampling iteration [{i+1}/{self.n_sampling}]")
            noisy_model = copy.deepcopy(self.models['backbone'])
            noisy_model.eval()
            noisy_model.apply(self.add_noise_to_weights)
            outputs_noisy = self.get_all_outputs(noisy_model, selection_loader, use_feature)
            diff_k = outputs_noisy - outputs
            for j in range(diff_k.shape[0]):
                diff_k[j,:] /= outputs[j].norm() 
            diffs = torch.cat((diffs, diff_k), dim=1)
        
        print("| Applying k-center greedy algorithm to select diverse samples")
        indsAll = self.kcenter_greedy(diffs, self.addendum)
        for ind in indsAll:
            uncertainty[ind] = 1
            
        return uncertainty.cpu().numpy()

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores