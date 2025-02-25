from .almethod import ALMethod 
import torch
import numpy as np
from tqdm import tqdm

class Uncertainty(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, selection_method="CONF", **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        
        selection_choices = [
            "CONF", "Entropy", "Margin", 
            "MeanSTD", "BALDDropout", "VarRatio", 
            "MarginDropout", "CONFDropout", "EntropyDropout"
        ]
        if selection_method not in selection_choices:
            raise NotImplementedError(f"Selection algorithm '{selection_method}' unavailable.")
        
        self.selection_method = selection_method
        self.eps = self.args.eps  # for AdversarialBIM, if needed

    def run(self):
        """
        主调函数：计算所有未标记数据的得分，然后根据得分进行排序并选择样本。
        """
        # 计算 scores
        scores = self.rank_uncertainty()
        # 选出分数最低（对应不确定度最高）的前 n_query 个下标
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        """
        根据 self.selection_method 不同，计算未标记集在模型下的“不确定性分数”。
        注意：这里的 scores 越小，表示不确定性越大（因为最终要用 np.argsort(scores)[:n]）。
        """
        model = self.models['backbone'].to(self.args.device)
        model.eval()  # 对于确定性推理（无dropout）的方法先用 eval()

        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size, 
            num_workers=self.args.workers
        )

        # 如果是 MC Dropout 类型的方法，下面会单独调用 self.predict_prob_dropout_split(...)
        # 在这个 for-loop 中什么也不做。反之则在循环中计算 scores。
        scores = np.array([])
        batch_num = len(selection_loader)

        # 针对非 MC Dropout 方法，直接单次前向即可
        print("| Calculating uncertainty of Unlabeled set...")
        if self.selection_method in ["CONF", "Entropy", "Margin", "VarRatio"]:
            # 在单次前向的模式下，模型保持 eval()
            with torch.no_grad():
                for data in tqdm(selection_loader, total=batch_num):
                    # 根据是文本还是图像数据取出对应的输入
                    if self.args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                        input_ids = data['input_ids'].to(self.args.device)
                        attention_mask = data['attention_mask'].to(self.args.device)
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    else:
                        inputs = data[0].to(self.args.device)
                        logits, _ = model(inputs)

                    if self.selection_method == "CONF":
                        # 置信度越低越不确定 → 直接取最大置信度值，然后我们要取它的“反面”。
                        # 最大置信度 = max prob
                        confs = torch.max(torch.softmax(logits, dim=1), dim=1).values
                        # 因为我们 argsort(scores) 是升序，所以把置信度本身作为 scores
                        # 小置信度会被排在前面、优先选中
                        scores = np.append(scores, confs.cpu().numpy())

                    elif self.selection_method == "Entropy":
                        # 计算熵: 通常公式为 -\sum p*log p
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        # 正常熵 = - sum(p * log(p))。下式相当于少了一个负号 → 这是负熵
                        # 如果想让越大越不确定，就手动加上负号使之变成正数再决定是否取负
                        # 这里演示更直观的做法：
                        ent = -(probs * np.log(probs + 1e-6)).sum(axis=1)
                        # 通常我们想要“熵大 -> 不确定性大”，
                        # 为了用 argsort(scores) 升序把最大熵排在最前，可以存 -ent:
                        # 或者干脆存 ent，然后最后使用 np.argsort(-ent)。这里保持和原写法一致：
                        scores = np.append(scores, -ent)  
                        # 这样越大的熵会越小的 -ent，排序会放到前面

                    elif self.selection_method == "Margin":
                        # margin = p(y1) - p(y2), y1是最大概率类别, y2是次大概率类别
                        probs = torch.softmax(logits, dim=1)
                        top1_vals, top1_idxs = probs.max(dim=1)
                        # 暂时将 top1 的位置置为 -1，使得再取 max 时就能拿到次大
                        tmp_probs = probs.clone()
                        tmp_probs[range(len(top1_idxs)), top1_idxs] = -1.0
                        top2_vals, _ = tmp_probs.max(dim=1)
                        margins = (top1_vals - top2_vals).cpu().numpy()
                        # margin 越小越不确定 → 所以我们直接存 margin，argsort 得到最小 margin 排在前
                        scores = np.append(scores, margins)

                    elif self.selection_method == "VarRatio":
                        # VarRatio 一般是 1 - max_prob，越大表示越不确定
                        probs = torch.softmax(logits, dim=1)
                        max_probs = torch.max(probs, dim=1).values
                        uncertainties = 1.0 - max_probs
                        # 越大越不确定，所以为了让它在升序排序里靠前，可以取负
                        scores = np.append(scores, -uncertainties.cpu().numpy())

        # ---------------------
        # 接下来处理 MC Dropout 相关方法
        # ---------------------
        if self.selection_method in [
            "MeanSTD", "BALDDropout", "MarginDropout", 
            "CONFDropout", "EntropyDropout"
        ]:
            # 注意：predict_prob_dropout_split 内部会使 model 处于 train()，
            # 以使dropout生效，从而进行多次采样
            probs_mc = self.predict_prob_dropout_split(
                self.unlabeled_set, 
                selection_loader, 
                n_drop=self.args.n_drop
            )

            # 下面根据不同方法对 MC dropout 的多次采样结果做不确定性度量
            if self.selection_method == "MeanSTD":
                # probs_mc.shape = [n_drop, N, num_classes]
                # 对 n_drop 个采样在最后一维做 std，然后再对类别做平均
                sigma_c = torch.std(probs_mc, dim=0)  # shape=[N, num_classes]
                # 不同类上的 std 再做平均
                uncertainties = sigma_c.mean(dim=1)   # shape=[N]
                # 不确定性越大 → uncertainties 越大
                # 为了升序排序时把它放前，可以存负数
                scores = -uncertainties.cpu().numpy()

            elif self.selection_method == "BALDDropout":
                # pb = E[p(y|x, w)] 做平均
                pb = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                # H(均值) = -\sum pb * log pb
                entropy1 = -(pb * torch.log(pb + 1e-6)).sum(dim=1)
                # E[H(p(y|x,w))] = E[ -\sum p*log p ]
                entropy2 = -(probs_mc * torch.log(probs_mc + 1e-6)).sum(dim=2).mean(dim=0)
                # mutual information = entropy2 - entropy1
                uncertainties = entropy2 - entropy1
                # 越大越不确定，所以最后取负用 argsort
                scores = -uncertainties.cpu().numpy()

            elif self.selection_method == "MarginDropout":
                # 先对 n_drop 个采样做平均，或者先对每个采样都算 top1 - top2?
                # 通常做法：先对 n_drop 个采样做平均 -> 平均概率 -> top1 - top2
                mean_probs = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                sorted_probs, _ = torch.sort(mean_probs, descending=True, dim=1)
                # margin = p1 - p2，越小越不确定
                margin_vals = (sorted_probs[:, 0] - sorted_probs[:, 1])
                # 因为 margin 越小越不确定，直接存 margin，就会选到 margin 最小的样本
                # 不需要再取负
                scores = np.append(scores, margin_vals.cpu().numpy())

            elif self.selection_method == "CONFDropout":
                # 先取 n_drop 个采样的最大置信度，再对它做某种聚合(例如平均)
                # 也可先取 mean_probs 再做最大值
                mean_probs = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                max_conf = torch.max(mean_probs, dim=1).values
                # 置信度越低越不确定，故直接存置信度即可
                scores = np.append(scores, max_conf.cpu().numpy())

            elif self.selection_method == "EntropyDropout":
                # 若 probs_mc 只是 logits，需要先 softmax；不过上面 predict_prob_dropout_split
                # 已经对每次采样都做过 softmax，这里只需要再对平均做熵？
                mean_probs = probs_mc.mean(dim=0)  # shape=[N, num_classes]
                ent = -(mean_probs * torch.log(mean_probs + 1e-6)).sum(dim=1)
                # 越大越不确定，故取负，保证越大熵排在越前
                scores = np.append(scores, -ent.cpu().numpy())

        return scores

    def select(self, **kwargs):
        """
        暴露给外部的方法：返回选中的未标记样本索引，以及每个样本的 scores。
        """
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):
        """
        开启 model.train() 使得 dropout 生效，并进行 n_drop 次采样，
        每次对所有样本做一次前向，最终返回形状为 (n_drop, dataset_size, num_classes) 的概率张量。
        """
        model = self.models['backbone'].to(self.args.device)
        model.train()  # VERY IMPORTANT：这样才能激活 dropout

        n_classes = len(self.args.target_list)
        probs = torch.zeros([n_drop, len(to_predict_dataset), n_classes], device=self.args.device)

        print('Processing Monte Carlo dropout...')
        # 重复 n_drop 次采样
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

                    # 累加存储概率
                    start_slice = evaluated_instances
                    end_slice = start_slice + batch_size
                    probs[i, start_slice:end_slice, :] = pred_probs
                    evaluated_instances = end_slice

        return probs
