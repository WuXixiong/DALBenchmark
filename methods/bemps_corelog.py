from .almethod import ALMethod
import torch
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch.nn.functional import normalize, softmax
import torch.nn.functional as F

class corelog(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
		# subset settings
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def run(self):
        selection_indices = self.rank_uncertainty()  # Already the final seleced indices
        return selection_indices, None 
    
    def rank_uncertainty(self):
        self.models['backbone'].eval()
        with torch.no_grad():
            unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)
            selection_loader = torch.utils.data.DataLoader(
                unlabeled_subset, batch_size=self.args.n_query, num_workers=self.args.workers
            )
            # X and T are the ratio parameters
            X = 0.1
            T = 0.5

            # 1. Get the probability distributions for the entire dataset
            probs_B_K_C = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)

            # 2. Split probs_B_K_C along the sample dimension into chunks
            n_chunks = 100  # number of chunks to split the dataset into
            chunked_probs = torch.chunk(probs_B_K_C, n_chunks, dim=1)

            # 2.1 Compute the global random indices (over the entire dataset)
            N = probs_B_K_C.shape[1]  # total number of samples
            global_xp_indices = random_generator_for_x_prime(N, X)
            # Compute the approximate chunk size (may not be perfectly equal due to remainder)
            chunk_size = int(np.ceil(N / n_chunks))

            rr_chunks = []  # to store intermediate rr results from each chunk

            # 3. Process each chunk
            for i, chunk in enumerate(chunked_probs):
                # Determine the global range for the current chunk
                chunk_start = i * chunk_size
                # Use the actual size of the chunk along dim=1 (samples)
                current_chunk_size = chunk.shape[1]
                chunk_end = chunk_start + current_chunk_size

                # Determine the local indices from the global selection that fall in this chunk
                local_indices = [idx - chunk_start for idx in global_xp_indices if chunk_start <= idx < chunk_end]
                # If no indices fall in the current chunk, default to selecting the first sample
                if len(local_indices) == 0:
                    local_indices = [0]

                # Transform current chunk: [n_drop, chunk_size, num_class] -> [chunk_size, n_drop, num_class]
                pr_YThetaX_chunk = chunk.permute(1, 0, 2)
                pr_ThetaL = 1 / pr_YThetaX_chunk.shape[1]

                # Adjust the probability distribution
                pr_YThetaX_chunk = pr_ThetaL * pr_YThetaX_chunk
                pr_YThetaX_chunk_Y = torch.transpose(pr_YThetaX_chunk, 1, 2)
                sum_pr = torch.sum(pr_YThetaX_chunk_Y, dim=-1, keepdim=True)
                pr_ThetaLXY_chunk = pr_YThetaX_chunk_Y / (sum_pr + 1e-10)
                pr_ThetaLXY_chunk = pr_ThetaLXY_chunk.unsqueeze(dim=1)

                # Instead of randomly selecting within the chunk, use the indices derived from global selection
                pr_YhThetaXp_chunk = pr_YThetaX_chunk[local_indices, :, :]

                # Compute pr(y_hat) for the chunk
                pr_Yhat_chunk = torch.matmul(pr_ThetaLXY_chunk, pr_YhThetaXp_chunk)

                epsilon = 1e-10
                # Expand dimensions to prepare for subsequent operations (similar to original code)
                pr_YhThetaXp_chunk_unsq = pr_YhThetaXp_chunk.unsqueeze(dim=0).unsqueeze(dim=0)
                pr_YhThetaXp_chunk_rep = pr_YhThetaXp_chunk_unsq.repeat(pr_Yhat_chunk.shape[0],
                                                                        pr_Yhat_chunk.shape[2],
                                                                        1, 1, 1)
                pr_Yhat_chunk_unsq = pr_Yhat_chunk.unsqueeze(dim=0)
                pr_Yhat_chunk_rep = pr_Yhat_chunk_unsq.repeat(pr_YhThetaXp_chunk.shape[1],
                                                            1, 1, 1, 1)
                pr_Yhat_chunk_trans = pr_Yhat_chunk_rep.transpose(0, 3).transpose(0, 1)

                ratio_chunk = (pr_YhThetaXp_chunk_rep + epsilon) / (pr_Yhat_chunk_trans + epsilon)
                core_log_chunk = pr_YhThetaXp_chunk_rep * torch.log(ratio_chunk)
                core_log_sum_chunk = torch.sum(core_log_chunk.sum(dim=-1), dim=-1)
                core_log_trans_chunk = core_log_sum_chunk.transpose(1, 2)
                core_log_final_chunk = core_log_trans_chunk.transpose(0, 1)

                pr_YLX_chunk = torch.sum(pr_YThetaX_chunk_Y, dim=-1)
                rr_Xp_X_Y_chunk = pr_YLX_chunk.unsqueeze(0) * core_log_final_chunk
                rr_Xp_X_chunk = torch.sum(rr_Xp_X_Y_chunk, dim=-1)

                rr_chunks.append(rr_Xp_X_chunk)

            # 4. Merge the intermediate results from all chunks
            concat_dim = 1
            target_shape = list(rr_chunks[0].shape)
            for t in rr_chunks:
                for i, size in enumerate(t.shape):
                    if i != concat_dim:
                        target_shape[i] = max(target_shape[i], size)
            rr_chunks_fixed = [self.pad_tensor_to_shape(t, target_shape, concat_dim) for t in rr_chunks]
            rr_Xp_X_merged = torch.cat(rr_chunks_fixed, dim=concat_dim)
            rr_X_Xp = rr_Xp_X_merged.transpose(0, 1)

            # 5. Cluster the results for final selection
            rr = clustering(rr_X_Xp, probs_B_K_C, T, self.args.n_query)
            return rr

    def pad_tensor_to_shape(self, tensor, target_shape, concat_dim=1):
        """
        Pad the tensor to make its dimensions, except for concat_dim, match the target_shape.
        If a dimension of the tensor is larger than the target_shape, keep the original size or trim as needed.
        Note that the order of the pad parameter in F.pad is [left pad of last dimension, right pad of last dimension, left pad of second last dimension, right pad of second last dimension, ...]
        """
        curr_shape = list(tensor.shape)
        pad_dims = []
        # Calculate each dimension in reverse order (starting from the last dimension)
        for i in range(len(curr_shape)-1, -1, -1):
            if i == concat_dim:
                # Do not modify the concatenation dimension
                pad_dims.extend([0, 0])
            else:
                diff = target_shape[i] - curr_shape[i]
                # If the current size is larger than the target, do not trim (or trim as needed; here we keep the original size)
                diff = diff if diff > 0 else 0
                # Pad diff units at the end of this dimension
                pad_dims.extend([0, diff])
        return F.pad(tensor, pad_dims)

    def select(self, **kwargs):
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
                    if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
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

def random_generator_for_x_prime(x_dim, size):
    num_to_sample = max(round(x_dim * size), 1)
    sample_indices = random.sample(range(0, x_dim), num_to_sample)
    return sorted(sample_indices)

def clustering(rr_X_Xp, probs_B_K_C, T, batch_size):

    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, max(round(probs_B_K_C.shape[0] * T), batch_size))
    # rr_topk_X = torch.topk(rr_X, max(round(probs_B_K_C.shape[1] * T), batch_size))
    rr_topk_X_indices = rr_topk_X.indices.cpu().detach().numpy()

    rr_X_Xp = rr_X_Xp[rr_topk_X_indices]
    rr_X_Xp = normalize(rr_X_Xp, dim=-1)

    rr = kmeans(rr_X_Xp, batch_size)
    rr = [rr_topk_X_indices[x] for x in rr]
    return rr

def kmeans(rr, k):
    rr = rr.cpu().numpy()
    if len(rr) < k:
        k = len(rr)
    kmeans = KMeans(n_clusters=k).fit(rr)
    centers = kmeans.cluster_centers_
    centroids = cdist(centers, rr).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(rr)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis=None)
    return centroids
