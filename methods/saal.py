from .almethod import ALMethod 
import torch
import numpy as np
import random
import copy
import pdb
from sklearn.metrics import pairwise_distances
from scipy import stats
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # 用于进度条

class SAAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        # 仅随机挑选一部分未标注数据用于构建子集
        subset_idx = np.random.choice(len(self.U_index),
                                      size=(min(self.args.subset, len(self.U_index)),),
                                      replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def run(self):
        self.models['backbone'].eval()
        print('...Acquisition Only')

        # 若有需要，可只采样一部分数据; 原代码中是 len(self.U_index) 整体
        # subpool_indices = random.sample(self.U_index, self.args.pool_subset)
        subpool_indices = random.sample(self.U_index, len(self.U_index))

        # 将数据一次性加载到CPU内存中（如数据过大，可考虑分批次或 DataLoader）
        pool_data_dropout = []
        for idx in subpool_indices:
            data = self.unlabeled_dst[idx]
            pool_data_dropout.append(data[0])
        pool_data_dropout = torch.stack(pool_data_dropout)

        # 计算获取函数（分数）
        points_of_interest = self.max_sharpness_acquisition_pseudo(
            pool_data_dropout,
            self.args,
            self.models['backbone']
        )
        points_of_interest = points_of_interest.detach().cpu().numpy()

        # 根据 acqMode 进行后处理，是否加入样本间多样性
        if 'Diversity' in self.args.acqMode:
            pool_index = self.init_centers(points_of_interest, int(self.args.n_query))
        else:
            # 按分数排序，取前 n_query 个
            pool_index = points_of_interest.argsort()[::-1][:int(self.args.n_query)]

        pool_index = torch.from_numpy(pool_index)
        return pool_index.cpu().tolist(), None  # index, score

    def max_sharpness_acquisition_pseudo(self, pool_data_dropout, args, model):
        """
        计算 (i) 原始loss 和 (ii) 参数扰动后的loss，
        根据acqMode返回不同的分数，如 'Max' 或 'Diff'。
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        data_size = pool_data_dropout.shape[0]
        # 伪标签
        pool_pseudo_target_dropout = torch.zeros(data_size, dtype=torch.long, device=device)

        # 存放原始loss和扰动后loss
        original_loss_list = []
        max_perturbed_loss_list = []

        # 分批处理，避免一次性显存占用过大
        num_batch = int(np.ceil(data_size / args.pool_batch_size))

        # ---------- 1) 先计算原始loss并得到伪标签 ----------
        model.eval()  # 只需要前向传播，不需要参数更新
        for idx in tqdm(range(num_batch), desc="Computing original loss"):
            start_idx = idx * args.pool_batch_size
            end_idx = min((idx + 1) * args.pool_batch_size, data_size)

            batch = pool_data_dropout[start_idx:end_idx].to(device)
            with torch.no_grad():
                output, _ = model(batch)
                softmaxed = F.softmax(output, dim=1)
                pseudo_target = softmaxed.argmax(dim=1)
                pool_pseudo_target_dropout[start_idx:end_idx] = pseudo_target

            # 计算原始loss（不需要梯度）
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(output, pseudo_target)
            original_loss_list.append(loss.detach())

        original_loss = torch.cat(original_loss_list, dim=0)

        # ---------- 2) 对参数做一次性的小扰动，计算扰动后loss ----------
        # 注意，这里对每个batch单独计算梯度、更新参数、再还原
        model.eval()
        for idx in tqdm(range(num_batch), desc="Computing perturbed loss"):
            start_idx = idx * args.pool_batch_size
            end_idx = min((idx + 1) * args.pool_batch_size, data_size)

            batch = pool_data_dropout[start_idx:end_idx].to(device)
            pseudo_target = pool_pseudo_target_dropout[start_idx:end_idx]

            # ---------- (a) 获取并保存当前模型参数 ----------
            original_params = [p.data.clone() for p in model.parameters() if p.requires_grad]

            # ---------- (b) 计算该 batch 的梯度 ----------
            # 首先要清空梯度
            model.zero_grad(set_to_none=True)
            # 启用梯度计算
            with torch.enable_grad():
                output, _ = model(batch)
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss1 = criterion(output, pseudo_target)
                # 对 mean 后的loss进行反传
                loss1.mean().backward()

            # ---------- (c) 根据梯度做参数扰动 ----------
            # norm of gradients
            with torch.no_grad():
                # 计算整合后的梯度范数
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(p=2).item() ** 2
                grad_norm = grad_norm ** 0.5

                # 计算缩放系数
                scale = args.rho / (grad_norm + 1e-12)

                # 对参数做 e_w = (p^2)*grad*scale 类型的更新
                idx_param = 0
                for p in model.parameters():
                    if p.grad is not None:
                        e_w = (original_params[idx_param] ** 2) * p.grad * scale
                        p.add_(e_w)
                    idx_param += 1

            # ---------- (d) 计算扰动后loss ----------
            with torch.no_grad():
                output_updated, _ = model(batch)
                loss2 = criterion(output_updated, pseudo_target)
            max_perturbed_loss_list.append(loss2.detach())

            # ---------- (e) 恢复原始参数 ----------
            with torch.no_grad():
                idx_param = 0
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.copy_(original_params[idx_param])
                        idx_param += 1

        max_perturbed_loss = torch.cat(max_perturbed_loss_list, dim=0)

        if args.acqMode == 'Max' or args.acqMode == 'Max_Diversity':
            return max_perturbed_loss
        elif args.acqMode == 'Diff' or args.acqMode == 'Diff_Diversity':
            return max_perturbed_loss - original_loss
        else:
            raise ValueError(f"Unknown acquisition mode: {args.acqMode}")

    def init_centers(self, X, K):
        """
        简化版 k-center 初始化，用于 'Diversity' 时选取代表性样本。
        """
        X_array = np.expand_dims(X, 1)  # Shape: (N, 1)
        ind = np.argmax([np.linalg.norm(s, 2) for s in X_array])
        mu = [X_array[ind]]  # 初始中心
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        D2 = None

        # tqdm 用于查看 k-center 选择过程
        pbar = tqdm(total=K, desc="K-center init")
        pbar.update(1)  # 已经初始化了一个中心

        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X_array, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X_array, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]

            if sum(D2) == 0.0:
                pdb.set_trace()

            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X_array[ind])
            indsAll.append(ind)
            cent += 1
            pbar.update(1)

        pbar.close()

        # 计算 Gram 矩阵（若后续还需用到可保留此处）
        gram = np.matmul(X_array[indsAll], X_array[indsAll].T)  # Shape: (K, K)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        # vgt = val[val > 1e-2]  # 若不再使用，可去掉

        return np.array(indsAll)
