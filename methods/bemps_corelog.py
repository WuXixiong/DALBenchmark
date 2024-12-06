from .almethod import ALMethod
import torch
import numpy as np
import random
from tqdm import tqdm

'''
@article{tan2021diversity,
  title={Diversity Enhanced Active Learning with Strictly Proper Scoring Rules},
  author={Tan, Wei and Du, Lan and Buntine, Wray},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
'''

class Corelog(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def run(self):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        with torch.no_grad():
            selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            X = 0.0939 # X should be changed in the future...

            # get prob
            probs_B_K_C = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)

            ## Pr(y|theta,x)
            pr_YThetaX_X_E_Y = probs_B_K_C
            pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

            ## Generate random number of x'
            xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
            pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

            ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
            pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
            pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

            sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
            pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

            ## Calculate pr(y_hat)
            pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
            pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)


            ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
            pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
            pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

            pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
            pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
            pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

            core_log = torch.mul(pr_YhThetaXp_X_Y_Xp_E_Yh, torch.log(torch.div(pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh)))
            core_log_X_Y = torch.sum(torch.sum(core_log.sum(dim=-1), dim=-1),dim=-1)

            ## Calculate RR
            pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
            rr = torch.sum(torch.mul(pr_YLX_X_Y, core_log_X_Y), dim=-1) / pr_YhThetaXp_Xp_E_Yh.shape[0]
        return rr

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

    def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        probs = torch.zeros([n_drop, len(to_predict_dataset), len(self.args.target_list)]).to(self.args.device)

        with torch.no_grad():
            print('Processing model dropout...')
            for i in tqdm(range(n_drop)):
                evaluated_instances = 0
                for _, data in enumerate(to_predict_dataloader):
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    preds = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                    preds = preds.logits  # Extract logits
                    batch_size = input_ids.size(0)
                    start_slice = evaluated_instances
                    end_slice = start_slice + batch_size
                    probs[i][start_slice:end_slice] = preds
                    evaluated_instances = end_slice

        return probs


## Random generator for X prime
def random_generator_for_x_prime(x_dim, size):
    sample_indices = random.sample(range(0, x_dim), round(x_dim * size))
    return sorted(sample_indices)