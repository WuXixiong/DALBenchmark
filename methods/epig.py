""" 
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""
from .almethod import ALMethod
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import random

class EPIG(ALMethod):
    def __init__(self, args, models, target_loader, unlabeled_dst, U_index, **kwargs):
        self.target_loader = target_loader # deal with addititonal arguments
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def select(self, **kwargs):
        scores = self.rank_uncertainty()
        scores = scores.cpu().numpy()
        selected_indices = np.argsort(scores)[:self.args.n_query]
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        selected_uindex = random.sample(range(int(len(self.unlabeled_set)/10)), int(self.args.n_query)) 
        sampler_unlabeled = SubsetRandomSampler(selected_uindex)  # make indices initial to the samples
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, sampler=sampler_unlabeled, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        scores = np.array([])
        batch_num = len(selection_loader)
        print("| Calculating probabilities of the unlabeled set")
        # probs_pool = self.predict_prob_dropout_split(len(self.unlabeled_set), selection_loader, n_drop=self.args.n_drop)
        probs_pool = self.predict_prob_dropout_split(int(len(self.unlabeled_set)/10), selection_loader, n_drop=self.args.n_drop)
        print("| Calculating probabilities of the target set")
        probs_target = self.predict_prob_dropout_split((int(self.args.n_class) * self.args.target_per_class), self.target_loader, n_drop=self.args.n_drop)
        scores = self.conditional_epig_from_probs(probs_pool, probs_target)  # [N_p, N_t]
        scores = torch.mean(scores, dim=-1)  # [N_p,]

        return scores
    
    def conditional_epig_from_probs(self, probs_pool, probs_targ):
        """
        EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

        Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

        Returns:
        Tensor[float], [N_p, N_t]
        """
        # Estimate the joint predictive distribution.
        probs_pool = probs_pool.unsqueeze(2)  # [2, N_p, 1, Cl]
        probs_targ = probs_targ.unsqueeze(1)  # [2, 1, N_t, Cl]
        probs_joint = probs_pool * probs_targ  # [2, N_p, N_t, Cl]
        probs_joint = probs_joint.sum(dim=0) # [N_p, N_t, Cl]
        # probs_joint = torch.mean(probs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the marginal predictive distributions.
        probs_pool = probs_pool.mean(0)
        probs_targ = probs_targ.mean(0)

    # Estimate the product of the marginal predictive distributions.
        probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
        nonzero_joint = probs_joint > 0  # [N_p, N_t, Cl, Cl]
        log_term = torch.clone(probs_joint)  # [N_p, N_t, Cl, Cl]
        log_term[nonzero_joint] = torch.log(probs_joint[nonzero_joint])  # [N_p, N_t, Cl, Cl]
        # probs_pool_targ_indep = probs_pool_targ_indep.permute(1, 2, 0, 3)  # 将维度排列成 [4890, 1000, 10, 10]
        log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])  # [N_p, N_t, Cl, Cl]
        scores = torch.sum(probs_joint * log_term, dim=-1)  # [N_p, N_t]

        del probs_targ, probs_pool, probs_joint, probs_pool_targ_indep, log_term, nonzero_joint
        torch.cuda.empty_cache()

        return scores  # [N_p, N_t]

    def predict_prob_dropout_split(self, dataset_length, to_predict_dataloader, n_drop):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, dataset_length, len(self.args.target_list)]).to(self.args.device)

        # Create a dataloader object to load the dataset
        # to_predict_dataloader = torch.utils.data.DataLoader(to_predict_dataset, batch_size=self.args['batch_size'], shuffle=False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in tqdm(range(n_drop)):

                evaluated_instances = 0
                # for i, data in enumerate(selection_loader):
                # inputs = data[0].to(self.args.device)
                for _, elements_to_predict in enumerate(to_predict_dataloader):
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict[0].to(self.args.device)
                    out = self.models['backbone'](elements_to_predict)[0]
                    # print(out)
                    pred = torch.nn.functional.softmax(out, dim=1)

                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice
        
        del elements_to_predict, out, pred, start_slice, end_slice, evaluated_instances
        torch.cuda.empty_cache()

        return probs