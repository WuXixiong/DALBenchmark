import numpy as np
import torch
# Custom
import torch.nn.functional as F 
import copy

from .almethod import ALMethod
from tqdm import tqdm

class noise_stability(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'): # and isinstance(m, nn.Conv2d):
                noise = torch.randn(m.weight.size())
                noise = noise.cuda()
                noise *= (self.args.ns_subset * m.weight.norm() / noise.norm())
                m.weight.add_(noise)

    def run(self, **kwargs):
        if self.args.noise_scale < 1e-8:
            uncertainty = torch.randn(self.args.n_query)
            return uncertainty

        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        uncertainty = torch.zeros(self.args.n_query).cuda()

        diffs = torch.tensor([]).cuda()
        use_feature = self.args.dataset in ['house']
        outputs = get_all_outputs(self.models['backbone'], selection_loader, use_feature)
        for i in tqdm(range(self.args.n_query)):
            noisy_model = copy.deepcopy(self.models['backbone'])
            noisy_model.eval()

            noisy_model.apply(self.add_noise_to_weights)
            outputs_noisy = get_all_outputs(noisy_model, selection_loader, use_feature)

            diff_k = outputs_noisy - outputs
            for j in range(diff_k.shape[0]):
                diff_k[j,:] /= outputs[j].norm() 
            diffs = torch.cat((diffs, diff_k), dim = 1)
        
        indsAll = kcenter_greedy(diffs, self.args.n_query)
        select_idx = [tensor.item() for tensor in indsAll]
        return select_idx, uncertainty.cpu()

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

def kcenter_greedy(X, K):
    avg_norm = np.mean([torch.norm(X[i]).item() for i in range(X.shape[0])])
    mu = torch.zeros(1, X.shape[1]).cuda()
    indsAll = []
    while len(indsAll) < K:
        if len(indsAll) == 0:
            D2 = torch.cdist(X, mu).squeeze(1)
        else:
            newD = torch.cdist(X, mu[-1:])
            newD = torch.min(newD, dim = 1)[0]
            for i in range(X.shape[0]):
                if D2[i] >  newD[i]:
                    D2[i] = newD[i]

        for i, ind in enumerate(D2.topk(1)[1]):
            D2[ind] = 0
            mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
            indsAll.append(ind)
    
    return indsAll

def get_all_outputs(model, unlabeled_loader, use_feature=False):
    model.eval()
    outputs = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cuda()
            out, fea = model(inputs) 
            if use_feature:
                out = fea
            else:
                out = F.softmax(out, dim = 1)
            outputs = torch.cat((outputs, out), dim=0)

    return outputs
