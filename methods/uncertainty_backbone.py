from .almethod import ALMethod
import torch
import numpy as np

class Uncertainty(ALMethod):
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

            scores = np.array([])
            batch_num = len(selection_loader)
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data[0].to(self.args.device)
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                    '''
                    How to calculate (measure) uncertainty ??
                    '''
        return scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores