"""
This implementation of the these classic uncertainty-based active learning algorithms is adapted from:
- The DeepAL+ repository (https://github.com/SineZHAN/deepALplus), which provides an implementation of these classic uncertainty-based active learning algorithms and other methods.

@article{zhan2022comparative,
  title={A comparative survey of deep active learning},
  author={Zhan, Xueying and Wang, Qingzhong and Huang, Kuan-hao and Xiong, Haoyi and Dou, Dejing and Chan, Antoni B},
  journal={arXiv preprint arXiv:2203.13450},
  year={2022}
}
"""
from .almethod import ALMethod 
import torch
import numpy as np
from tqdm import tqdm

class Uncertainty(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, selection_method="CONF", **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        
        selection_choices = [
            "CONF", "Entropy", "Margin", 
            "MeanSTD", "BALD", "VarRatio", 
            "MarginDropout", "CONFDropout", "EntropyDropout"
        ]
        if selection_method not in selection_choices:
            raise NotImplementedError(f"Selection algorithm '{selection_method}' unavailable.")
        
        self.selection_method = selection_method

    def run(self):
        """
        Main function: Compute scores for all unlabeled data, 
        then sort and select samples based on the scores.
        """
        # Compute scores
        scores = self.rank_uncertainty()
        # Select the top n_query indices with the lowest scores (corresponding to the highest uncertainty)
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        """
        Compute the "uncertainty score" of the unlabeled set under the model
        based on different self.selection_method options.
        Note: The smaller the score, the higher the uncertainty 
        (because we use np.argsort(scores)[:n] in the end).
        """
        model = self.models['backbone'].to(self.args.device)
        model.eval()  # For deterministic inference (without dropout), use eval()

        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size, 
            num_workers=self.args.workers
        )

        # If using MC Dropout-based methods, self.predict_prob_dropout_split(...) will be called separately,
        # so nothing is done in this for-loop. Otherwise, scores are computed within the loop.
        scores = np.array([])
        batch_num = len(selection_loader)

        # For non-MC Dropout methods, a single forward pass is sufficient
        print("| Calculating uncertainty of Unlabeled set...")
        if self.selection_method in ["CONF", "Entropy", "Margin", "VarRatio"]:
            # In single forward-pass mode, keep the model in eval()
            with torch.no_grad():
                for data in tqdm(selection_loader, total=batch_num):
                    # Extract input based on whether the dataset is text or image
                    if self.args.textset:
                        input_ids = data['input_ids'].to(self.args.device)
                        attention_mask = data['attention_mask'].to(self.args.device)
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    else:
                        inputs = data[0].to(self.args.device)
                        logits, _ = model(inputs)

                    if self.selection_method == "CONF":
                        # Lower confidence means higher uncertainty → Take the maximum confidence value, 
                        # then we use its "inverse."
                        # Maximum confidence = max probability
                        confs = torch.max(torch.softmax(logits, dim=1), dim=1).values
                        # Since we use argsort(scores) in ascending order, 
                        # we store confidence as scores so that lower confidence gets selected first.
                        scores = np.append(scores, confs.cpu().numpy())

                    elif self.selection_method == "Entropy":
                        # Compute entropy: standard formula is -\sum p*log p
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        # Normal entropy = -sum(p * log(p)). The expression below lacks a negative sign → it's negative entropy.
                        # To ensure "higher entropy → higher uncertainty," we manually add a negative sign.
                        ent = -(probs * np.log(probs + 1e-6)).sum(axis=1)
                        # We usually want "high entropy → high uncertainty."
                        # To make the highest entropy appear first using np.argsort(scores), we store -ent.
                        # Alternatively, store ent and use np.argsort(-ent) later. Keeping it consistent with the original implementation:
                        scores = np.append(scores, -ent)  
                        # This way, the highest entropy results in the smallest -ent, so it is ranked first.

                    elif self.selection_method == "Margin":
                        # margin = p(y1) - p(y2), where y1 is the highest probability class, y2 is the second highest probability class
                        probs = torch.softmax(logits, dim=1)
                        top1_vals, top1_idxs = probs.max(dim=1)
                        # Temporarily set top1 positions to -1 so that the next max operation gets the second highest
                        tmp_probs = probs.clone()
                        tmp_probs[range(len(top1_idxs)), top1_idxs] = -1.0
                        top2_vals, _ = tmp_probs.max(dim=1)
                        margins = (top1_vals - top2_vals).cpu().numpy()
                        # Smaller margin means higher uncertainty → We store margin so that argsort selects the smallest margins first.
                        scores = np.append(scores, margins)

                    elif self.selection_method == "VarRatio":
                        # VarRatio is typically 1 - max_prob, where a larger value indicates higher uncertainty.
                        probs = torch.softmax(logits, dim=1)
                        max_probs = torch.max(probs, dim=1).values
                        uncertainties = 1.0 - max_probs
                        # Since higher values mean higher uncertainty, store negative values to ensure they appear first in ascending order sorting.
                        scores = np.append(scores, -uncertainties.cpu().numpy())

        # ---------------------
        # Handling MC Dropout-based methods
        # ---------------------
        if self.selection_method in [
            "MeanSTD", "BALD", "MarginDropout", 
            "CONFDropout", "EntropyDropout"
        ]:
            # Note: predict_prob_dropout_split internally sets model.train() 
            # to activate dropout and perform multiple forward passes
            probs_mc = self.predict_prob_dropout_split(
                self.unlabeled_set, 
                selection_loader, 
                n_drop=self.args.n_drop
            )

            # Compute uncertainty from MC dropout multiple sampling results
            if self.selection_method == "MeanSTD":
                # probs_mc.shape = [n_drop, N, num_classes]
                # Compute std over n_drop samples, then take the mean over classes
                sigma_c = torch.std(probs_mc, dim=0)  # shape=[N, num_classes]
                uncertainties = sigma_c.mean(dim=1)   # shape=[N]
                # Higher uncertainty → larger uncertainties
                # Store negative values to ensure correct ranking
                scores = -uncertainties.cpu().numpy()

            elif self.selection_method == "BALD":
                # pb = E[p(y|x, w)] (mean probability)
                pb = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                # H(mean) = -\sum pb * log pb
                entropy1 = -(pb * torch.log(pb + 1e-6)).sum(dim=1)
                # E[H(p(y|x,w))] = E[ -\sum p*log p ]
                entropy2 = -(probs_mc * torch.log(probs_mc + 1e-6)).sum(dim=2).mean(dim=0)
                # Mutual information = entropy2 - entropy1
                uncertainties = entropy2 - entropy1
                scores = -uncertainties.cpu().numpy()

            elif self.selection_method == "MarginDropout":
                # First, average over n_drop samples -> mean probability -> top1 - top2
                mean_probs = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                sorted_probs, _ = torch.sort(mean_probs, descending=True, dim=1)
                # margin = p1 - p2, the smaller the margin, the higher the uncertainty
                margin_vals = (sorted_probs[:, 0] - sorted_probs[:, 1])
                # Since a smaller margin means higher uncertainty, we store margin directly, 
                # so that the smallest margin samples get selected.
                # No need to take the negative.
                scores = np.append(scores, margin_vals.cpu().numpy())

            elif self.selection_method == "CONFDropout":
                # Take the maximum confidence from n_drop samples, then apply some aggregation (e.g., mean)
                # Alternatively, compute mean_probs first and then take the maximum value
                mean_probs = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                max_conf = torch.max(mean_probs, dim=1).values
                # Lower confidence means higher uncertainty, so we store confidence directly.
                scores = np.append(scores, max_conf.cpu().numpy())

            elif self.selection_method == "EntropyDropout":
                mean_probs = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                ent = -(mean_probs * torch.log(mean_probs + 1e-6)).sum(dim=1)
                # Higher entropy means higher uncertainty, so take the negative to ensure
                # that higher entropy is ranked first.
                scores = np.append(scores, -ent.cpu().numpy())

        return scores

    def select(self, **kwargs):
        """
        Exposed method: Returns selected unlabeled sample indices and their scores.
        """
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):
        """
        Set model to train() to activate dropout, perform n_drop forward passes,
        and return a probability tensor of shape (n_drop, dataset_size, num_classes).
        """
        model = self.models['backbone'].to(self.args.device)
        model.train()  # VERY IMPORTANT: This activates dropout

        n_classes = len(self.args.target_list)
        probs = torch.zeros([n_drop, len(to_predict_dataset), n_classes], device=self.args.device)

        print('Processing Monte Carlo dropout...')
        # Re-sample n_drop times
        for i in tqdm(range(n_drop)):
            evaluated_instances = 0
            for batch_data in to_predict_dataloader:
                with torch.no_grad():
                    if self.args.textset:
                        input_ids = batch_data['input_ids'].to(self.args.device)
                        attention_mask = batch_data['attention_mask'].to(self.args.device)
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        pred_probs = torch.softmax(logits, dim=-1)
                        batch_size = input_ids.size(0)
                    else:
                        inputs = batch_data[0].to(self.args.device)
                        logits, _ = model(inputs)
                        pred_probs = torch.softmax(logits, dim=1)
                        batch_size = inputs.size(0)

                    # Accumulate and store probabilities
                    start_slice = evaluated_instances
                    end_slice = start_slice + batch_size
                    probs[i, start_slice:end_slice, :] = pred_probs
                    evaluated_instances = end_slice

        return probs
    