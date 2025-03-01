import numpy as np
import torch
import torch.nn.functional as F
import copy

from .almethod import ALMethod
from tqdm import tqdm


class noise_stability(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    @torch.no_grad()
    def add_noise_to_weights(self, model):
        """
        在原模型权重上叠加噪声，然后在函数外进行推理后再还原。
        """
        for name, param in model.named_parameters():
            # 只对权重（或某些指定层）加噪音
            # 例如可以检查 if 'weight' in name 等
            if param.requires_grad and param.dim() > 1:  
                noise = torch.randn_like(param)
                scale_factor = (self.args.ns_subset * param.norm() / noise.norm()).item()
                param.add_(noise * scale_factor)

    @torch.no_grad()
    def backup_and_add_noise(self, model, backup_params):
        """
        先备份模型参数，再给模型加噪。
        """
        for backup_p, model_p in zip(backup_params, model.parameters()):
            backup_p.copy_(model_p)  # 备份当前参数
        self.add_noise_to_weights(model)

    @torch.no_grad()
    def restore_weights(self, model, backup_params):
        """
        将模型参数还原到备份版本。
        """
        for backup_p, model_p in zip(backup_params, model.parameters()):
            model_p.copy_(backup_p)

    def run(self, **kwargs):
        # 如果噪声非常小，可以直接随机返回结果
        if self.args.noise_scale < 1e-8:
            uncertainty = torch.randn(self.args.n_query)
            return uncertainty

        # DataLoader，用于对未标记数据集做推理
        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers
        )

        # 用于储存最终不确定性分数，这里只是接口需求
        uncertainty = torch.zeros(self.args.n_query).to(self.args.device)

        # 先把原模型对未标记集的输出全部取出
        use_feature = (self.args.dataset in ['house'])
        backbone = self.models['backbone']
        backbone.eval()
        outputs = self.get_all_outputs(backbone, selection_loader, use_feature)
        # 计算原输出各样本的范数（行范数），后续 diff_k 要除以这个值
        # shape: [num_unlabeled, 1]
        row_norms = torch.norm(outputs, dim=1, keepdim=True)

        # 事先备份一份模型参数，这样我们不用copy.deepcopy整个模型
        backup_params = [p.data.clone() for p in backbone.parameters()]

        # 收集 diff_k 的列表，最后再一次 cat
        diffs_list = []

        # 在 n_query 次迭代中，为同一个模型加噪推理，然后还原参数
        for _ in tqdm(range(self.args.n_query)):
            # 给模型参数加噪
            self.backup_and_add_noise(backbone, backup_params)
            # noisy forward
            outputs_noisy = self.get_all_outputs(backbone, selection_loader, use_feature)
            # 还原
            self.restore_weights(backbone, backup_params)

            # diff_k = outputs_noisy - outputs
            diff_k = outputs_noisy - outputs
            # 各行除以原输出的范数
            diff_k = diff_k / row_norms
            diffs_list.append(diff_k)

        # 一次性拼接 diff_k, 最终形状 [num_unlabeled, n_query * out_dim]
        diffs = torch.cat(diffs_list, dim=1)

        # 用 kcenter_greedy 方法选出 K 个索引
        indsAll = self.kcenter_greedy(diffs, self.args.n_query)
        select_idx = [tensor.item() for tensor in indsAll]
        return select_idx, uncertainty.cpu()

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores


    def kcenter_greedy(self, X, K):
        """
        对矩阵 X (shape = [N, dim]) 做 k-center greedy.
        返回选中的样本索引列表，长度为 K。
        """
        # 初始化
        N = X.shape[0]
        mu = torch.zeros(1, X.shape[1], device=X.device)
        indsAll = []
        with torch.no_grad():
            # 距离缓存
            D2 = torch.cdist(X, mu).squeeze(1)  # (N,)
            while len(indsAll) < K:
                # 重新计算到新中心的距离，并更新最小距离
                newD = torch.cdist(X, mu[-1:].detach()).squeeze(1)  # shape: [N]
                D2 = torch.min(D2, newD)
                # 找到当前距离最大的点
                ind = torch.argmax(D2)  
                # 将该点加入中心集合
                mu = torch.cat((mu, X[ind].unsqueeze(0)), dim=0)
                D2[ind] = 0
                indsAll.append(ind)
        return indsAll


    @torch.no_grad()
    def get_all_outputs(self, model, unlabeled_loader, use_feature=False):
        """
        将传入 model 对所有 unlabeled_loader 里的数据做一次前向推理。
        根据 use_feature 选择输出特征向量或分类概率。
        """
        model.eval()
        outputs = []
        for data in unlabeled_loader:
            if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                # 提取 input_ids 和 attention_mask，并将其移动到指定设备
                input_ids = data['input_ids'].to(self.args.device)
                attention_mask = data['attention_mask'].to(self.args.device)
                # 获取模型输出
                model_output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = model_output.logits
                if use_feature:
                    # 如果需要特征，提取最后一层隐藏状态的第一个标记（通常是 [CLS] 标记）的输出
                    features = model_output.hidden_states[-1][:, 0, :]
                    batch_out = features
                else:
                    # 否则，计算分类概率
                    batch_out = F.softmax(logits, dim=1)
            else:
                # 对于其他数据集，处理方式可能不同，请根据实际情况调整
                inputs = data[0].to(self.args.device)
                logits, features = model(inputs)
                batch_out = features if use_feature else F.softmax(logits, dim=1)
            
            outputs.append(batch_out)
        return torch.cat(outputs, dim=0)

