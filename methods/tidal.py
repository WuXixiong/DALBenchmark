from .almethod import ALMethod
import torch
import numpy as np
from tqdm import tqdm

class TIDAL(ALMethod):
    '''
    https://github.com/hyperconnect/TiDAL
    '''
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        # subset selection (for diversity)
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def select(self, **kwargs):
        scores = self.rank_uncertainty()
        selected_indices = np.argsort(-scores)[:self.args.n_query]
        # Q_index = [self.U_index[idx] for idx in selected_indices]
        Q_index = self.U_index_sub[selected_indices]

        return Q_index, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

        scores = np.array([])
        print("| Calculating uncertainty of Unlabeled set")

        uncertainties = self.get_cumulative_entropy(self.models, selection_loader, self.args)
        uncertainties_np = uncertainties.cpu().numpy()  # 取负
        scores= np.append(scores, uncertainties_np)

        return scores

    def get_cumulative_entropy(self, models, unlabeled_loader, args):
        models['backbone'].eval()
        models['module'].eval()
        models['module'].cuda()
        #test_models = nets.tdnet.TDNet()
        #test_models = test_models.cuda()
        # test_models.eval()
        unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)

        with torch.cuda.device(0):
            sub_logit_all = torch.tensor([])
            pred_label_all = torch.tensor([])

        with torch.no_grad():
            # first_batch = next(iter(unlabeled_loader))
            # print(first_batch)
            unlabeled_loader = torch.utils.data.DataLoader(unlabeled_subset, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            for inputs, _, _, _  in unlabeled_loader:
                with torch.cuda.device(0):
                    inputs = inputs.cuda()
                main_logit, _, features = models['backbone'](inputs, method='TIDAL')
                _, pred_label = torch.max(main_logit, dim=1)
                pred_label = pred_label.detach().cpu()
                # pred_label = pred_label.detach()
                pred_label_all = torch.cat((pred_label_all, pred_label), 0)
                sub_logit = models['module'](features)
                # sub_logit = test_models(features)
                sub_logit = sub_logit.detach().cpu()
                sub_logit_all = torch.cat((sub_logit_all, sub_logit), 0)
                # pred_label = pred_label.detach().cpu()
                # sub_logit = sub_logit.detach().cpu()

        sub_prob = torch.softmax(sub_logit_all, dim=1)

        if args.tidal_query != "None":
            if args.tidal_query == 'AUM':
                n_classes = sub_prob.size(1)
                sub_assigned_prob_onehot = torch.eye(n_classes)[pred_label_all.type(torch.int64)] * sub_prob
                sub_assigned_prob = torch.sum(sub_assigned_prob_onehot, dim=1)
                sub_second_prob = torch.max(sub_prob - sub_assigned_prob_onehot, dim=1)[0]
                AUM = sub_assigned_prob - sub_second_prob
                uncertainty = -AUM
            elif args.tidal_query == 'Entropy':
                sub_entropy = -(sub_prob * torch.log(sub_prob)).sum(dim=1)
                uncertainty = sub_entropy

        return uncertainty.cpu()
