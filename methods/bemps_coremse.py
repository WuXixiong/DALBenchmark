from .almethod import ALMethod
import torch
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch.nn.functional import normalize, softmax
import torch.nn.functional as F

class coremse(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
		# subset settings
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def run(self):
        selection_indices = self.rank_uncertainty()  # 已经是最终选出的索引
        return selection_indices, None  # 或者 scores 传个空，或者传别的

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
                pr_YThetaX_chunk = chunk.permute(1, 0, 2) # 108: pr_YThetaX_X_E_Y = probs_B_K_C
                pr_ThetaL = 1 / pr_YThetaX_chunk.shape[1] # 109: pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

                # Adjust the probability distribution
                pr_YThetaX_chunk = pr_ThetaL * pr_YThetaX_chunk # 116: pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
                pr_YThetaX_chunk_Y = torch.transpose(pr_YThetaX_chunk, 1, 2) # 117: pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y
                sum_pr = torch.sum(pr_YThetaX_chunk_Y, dim=-1, keepdim=True) # 119: sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
                pr_ThetaLXY_chunk = pr_YThetaX_chunk_Y / (sum_pr + 1e-10) # 120: pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1
                pr_ThetaLXY_chunk = pr_ThetaLXY_chunk.unsqueeze(dim=1) # 123: pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)

                # Instead of randomly selecting within the chunk, use the indices derived from global selection
                pr_YhThetaXp_chunk = pr_YThetaX_chunk[local_indices, :, :] # 113: pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

                # Compute pr(y_hat) for the chunk
                pr_Yhat_chunk = torch.matmul(pr_ThetaLXY_chunk, pr_YhThetaXp_chunk) # 124: pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)

                epsilon = 1e-10
                # Expand dimensions to prepare for subsequent operations (similar to original code)
                pr_YhThetaXp_chunk_unsq = pr_YhThetaXp_chunk.unsqueeze(dim=0).unsqueeze(dim=0) # 127: pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
                pr_YhThetaXp_chunk_rep = pr_YhThetaXp_chunk_unsq.repeat(pr_Yhat_chunk.shape[0],
                                                                        pr_Yhat_chunk.shape[2],
                                                                        1, 1, 1) # 128: pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)
                pr_Yhat_chunk_unsq = pr_Yhat_chunk.unsqueeze(dim=0) # 130: pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
                pr_Yhat_chunk_rep = pr_Yhat_chunk_unsq.repeat(pr_YhThetaXp_chunk.shape[1],
                                                            1, 1, 1, 1) # 131: pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
                pr_Yhat_chunk_trans = pr_Yhat_chunk_rep.transpose(0, 3).transpose(0, 1) # 132: pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

                core_mse_chunk = (pr_YhThetaXp_chunk_rep - pr_Yhat_chunk_trans).pow(2) # 134: core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
                core_mse_X_Y_XP_chunk = torch.sum(core_mse_chunk.sum(dim=-1), dim=-1) # 135: core_mse_X_Y_Xp = torch.sum(core_mse.sum(dim=-1), dim=-1)
                core_mse_X_Xp_Y_chunk = torch.transpose(core_mse_X_Y_XP_chunk, 1, 2) # 136: core_mse_X_Xp_Y = core_mse_X_Y_Xp.transpose(1,2)
                core_mse_Xp_X_Y_chunk = torch.transpose(core_mse_X_Xp_Y_chunk, 0, 1) # 137: core_mse_Xp_X_Y = core_mse_X_Xp_Y.transpose(0,1)

                # calculate rr in each chunk
                pr_YLX_chunk = torch.sum(pr_YThetaX_chunk_Y, dim=-1) # 140: pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
                rr_Xp_X_Y_chunk = pr_YLX_chunk.unsqueeze(0) * core_mse_Xp_X_Y_chunk # 142: rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_mse_Xp_X_Y
                rr_Xp_X_chunk = torch.sum(rr_Xp_X_Y_chunk, dim=-1) # 144: rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)

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

            rr_X_Xp = rr_Xp_X_merged.transpose(0, 1) # 145: rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

            # 5. Cluster the results for final selection
            rr = clustering(rr_X_Xp, probs_B_K_C, T, self.args.n_query)
            return rr



    def pad_tensor_to_shape(self, tensor, target_shape, concat_dim=1):
        """
        对 tensor 进行 padding 使其在除 concat_dim 外的各维度达到 target_shape 的尺寸。
        若某个维度 tensor 大于 target_shape则保持原尺寸或视情况裁剪。
        注意 F.pad 的 pad 参数顺序为 [最后一维左补, 最后一维右补, 倒数第二维左补, 倒数第二维右补, ...]
        """
        curr_shape = list(tensor.shape)
        pad_dims = []
        # 依次计算每个维度（从最后一维开始）
        for i in range(len(curr_shape)-1, -1, -1):
            if i == concat_dim:
                # 拼接维度不做修改
                pad_dims.extend([0, 0])
            else:
                diff = target_shape[i] - curr_shape[i]
                # 如果当前尺寸大于目标，则不裁剪（或根据需求进行裁剪，这里保持原尺寸）
                diff = diff if diff > 0 else 0
                # 在该维度末尾填充 diff 个单位
                pad_dims.extend([0, diff])
        return F.pad(tensor, pad_dims)

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    # drop out for images
    # def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):

    #     # self.device = "cuda" if torch.cuda.is_available() else "cpu"

    #     # Ensure model is on right device and is in TRAIN mode.
    #     # Train mode is needed to activate randomness in dropout modules.
    #     self.models['backbone'].train()
    #     self.models['backbone'] = self.models['backbone'].to(self.args.device)

    #     # Create a tensor to hold probabilities
    #     probs = torch.zeros([n_drop, len(to_predict_dataset), len(self.args.target_list)]).to(self.args.device)

    #     # Create a dataloader object to load the dataset
    #     # to_predict_dataloader = torch.utils.data.DataLoader(to_predict_dataset, batch_size=self.args['batch_size'], shuffle=False)

    #     with torch.no_grad():
    #         # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
    #         print('Processing model dropout...')
    #         for i in tqdm(range(n_drop)):

    #             evaluated_instances = 0
    #             # for i, data in enumerate(selection_loader):
    #             # inputs = data[0].to(self.args.device)
    #             for _, elements_to_predict in enumerate(to_predict_dataloader):
    #                 # Calculate softmax (probabilities) of predictions
    #                 elements_to_predict = elements_to_predict[0].to(self.args.device)
    #                 out = self.models['backbone'](elements_to_predict)[0]
    #                 # print(out)
    #                 pred = torch.nn.functional.softmax(out, dim=1)

    #                 # Accumulate the calculated batch of probabilities into the tensor to return
    #                 start_slice = evaluated_instances
    #                 end_slice = start_slice + elements_to_predict.shape[0]
    #                 probs[i][start_slice:end_slice] = pred
    #                 evaluated_instances = end_slice

    #     return probs
    
    def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        probs = torch.zeros([n_drop, len(to_predict_dataset), len(self.args.target_list)], device=self.args.device)

        with torch.no_grad():
            print('Processing model dropout...')
            for i in tqdm(range(n_drop)):
                evaluated_instances = 0
                for _, data in enumerate(to_predict_dataloader):
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    preds = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask).logits
                    # 将logits转换为概率分布
                    pred_probs = softmax(preds, dim=-1)
                    batch_size = input_ids.size(0)
                    start_slice = evaluated_instances
                    end_slice = start_slice + batch_size
                    probs[i][start_slice:end_slice] = pred_probs
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
    rr_X_Xp = normalize(rr_X_Xp, dim=-1)  # 根据需要选择正确的归一化维度

    rr = kmeans(rr_X_Xp, batch_size)
    rr = [rr_topk_X_indices[x] for x in rr]
    return rr

def kmeans(rr, k):
    rr = rr.cpu().numpy()
    if len(rr) < k:
        # 如果样本数不足以形成k个聚类，可根据逻辑进行特殊处理
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
