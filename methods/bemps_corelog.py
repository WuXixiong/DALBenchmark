from .almethod import ALMethod
import torch
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch.nn.functional import normalize, softmax

class Corelog(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def run(self):
        scores = self.rank_uncertainty()
        # print('the shape of scores is')
        # print(scores.shape)
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        with torch.no_grad():
            selection_loader = torch.utils.data.DataLoader(
                self.unlabeled_set,
                batch_size=self.args.test_batch_size,
                num_workers=self.args.workers
            )
            # X和T是比例参数，可根据需要调整
            X = 0.001  
            T = 0.5

            # 获取概率分布
            probs_B_K_C = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            print("Shape of probs_B_K_C:", probs_B_K_C.shape)

            pr_YThetaX_X_E_Y = probs_B_K_C.permute(1,0,2)
            print("Shape of pr_YThetaX_X_E_Y:", pr_YThetaX_X_E_Y.shape)

            pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]
            print("Value of pr_ThetaL:", pr_ThetaL)

            # 修正xp_indices的生成方式，从B个样本中挑选
            xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
            print("Shape of xp_indices:", len(xp_indices))

            pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]
            print("Shape of pr_YhThetaXp_Xp_E_Yh:", pr_YhThetaXp_Xp_E_Yh.shape)

            # pr(theta|L,x,y)
            pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
            pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)
            sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1, keepdim=True)
            pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / (sum_pr_YThetaX_X_Y_1 + 1e-10)

            # 计算pr(y_hat)
            pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
            # 确保矩阵相乘维度对齐，这里逻辑比较复杂，不给出具体重排代码，请根据实际情况进行transpose或einsum
            # 需要根据具体模型概率分布含义，对 pr_YhThetaXp_Xp_E_Yh 进行必要的转置以匹配矩阵乘法维度。
            # 示例（如果需要）：pr_YhThetaXp_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.permute(1,2,0) # 根据需求变换
            
            pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)
            print("Shape of pr_Yhat_X_Xp_Y_Yh:", pr_Yhat_X_Xp_Y_Yh.shape)

            # 后续计算中加入epsilon避免log(0)
            epsilon = 1e-10
            pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim=0).unsqueeze(dim=0)
            pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)
            pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim=0)
            pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1], 1, 1, 1, 1)
            pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0, 3).transpose(0, 1)

            ratio = (pr_YhThetaXp_X_Y_Xp_E_Yh + epsilon) / (pr_Yhat_X_Y_Xp_E_Yh + epsilon)
            core_log = pr_YhThetaXp_X_Y_Xp_E_Yh * torch.log(ratio)
            core_log_X_Y_Xp = torch.sum(core_log.sum(dim=-1), dim=-1)
            core_log_X_Xp_Y = core_log_X_Y_Xp.transpose(1, 2)
            core_log_Xp_X_Y = core_log_X_Xp_Y.transpose(0, 1)

            pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
            rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_log_Xp_X_Y
            rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
            rr_X_Xp = rr_Xp_X.transpose(0, 1)

            rr = clustering(rr_X_Xp, probs_B_K_C, T, self.args.n_query)
            print("Final clustering result shape:", len(rr))

            return rr

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

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
    print("Input rr_X_Xp shape:", rr_X_Xp.shape)
    print("Input probs_B_K_C shape:", probs_B_K_C.shape)
    print("T value:", T)
    print("Batch size:", batch_size)

    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, max(round(probs_B_K_C.shape[0] * T), batch_size))
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
