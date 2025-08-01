"""
This implementation is primarily based on the official open-source code of:
GitHub: https://github.com/lixingjian/AL-noise-stability
Paper: https://ojs.aaai.org/index.php/AAAI/article/view/29270

@inproceedings{li2024deep,
  title={Deep active learning with noise stability},
  author={Li, Xingjian and Yang, Pengkun and Gu, Yangcheng and Zhan, Xueying and Wang, Tianyang and Xu, Min and Xu, Chengzhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={12},
  pages={13655--13663},
  year={2024}
}
"""
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