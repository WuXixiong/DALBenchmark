from .almethod import ALMethod
import torch
import numpy as np

class Uncertainty(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, selection_method="CONF", **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        selection_choices = ["CONF", "Entropy", "Margin", "MeanSTD", "AdversarialBIM", "BALDDropout", "VarRatio", "MarginDropout",
                             "CONFDropout", "EntropyDropout"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method
        # for AdversarialBIM
        self.eps = self.args.eps

    def run(self):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.models['backbone'] = self.models['backbone'].to(self.args.device)
        self.models['backbone'].eval()
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

        scores = np.array([])
        batch_num = len(selection_loader)
        print("| Calculating uncertainty of Unlabeled set")
        for i, data in enumerate(selection_loader):
            if self.args.dataset in ['AGNEWS']:
                # Extract input_ids, attention_mask, and labels from the dictionary
                input_ids = data['input_ids'].to(self.args.device)
                attention_mask = data['attention_mask'].to(self.args.device)
                # labels = data['labels'].to(self.args.device)
            else:
                inputs = data[0].to(self.args.device)


            if i % self.args.print_freq == 0:
                print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))

            if self.selection_method == "AdversarialBIM":
                self.models['backbone'].train()
                self.models['backbone'] = self.models['backbone'].to(self.args.device)
                scores = np.append(scores, self.cal_dis_bim(inputs).cpu().numpy())
            elif self.selection_method == "Adversarialdeepfool":
                self.models['backbone'].train()
                self.models['backbone'] = self.models['backbone'].to(self.args.device)
                scores = np.append(scores, self.cal_dis_deepfool(inputs).cpu().detach().numpy())
            else:
                with torch.no_grad():
                    if self.selection_method == "CONF":
                        if self.args.dataset in ['AGNEWS']:
                            preds = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                            preds = preds.logits  # Extract logits for AGNEWS
                        else:
                            preds, _ = self.models['backbone'](inputs)
                        confs = preds.max(axis=1).values.cpu().numpy()
                        scores = np.append(scores, confs)
                    elif self.selection_method == "Entropy":
                        if self.args.dataset in ['AGNEWS']:
                            preds = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                            preds = preds.logits  # Extract logits for AGNEWS
                        else:
                            preds, _ = self.models['backbone'](inputs)
                        preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                        entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                        scores = np.append(scores, entropys)
                    elif self.selection_method == "Margin":
                        if self.args.dataset in ['AGNEWS']:
                            preds = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                            preds = preds.logits  # Extract logits for AGNEWS
                        else:
                            preds, _ = self.models['backbone'](inputs)
                        preds = torch.nn.functional.softmax(preds, dim=1)
                        preds_argmax = torch.argmax(preds, dim=1)
                        max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                        preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                        preds_sub_argmax = torch.argmax(preds, dim=1)
                        margins = (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy()
                        scores = np.append(scores, margins)
                    elif self.selection_method == "VarRatio":
                        if self.args.dataset in ['AGNEWS']:
                            preds = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
                            preds = preds.logits  # Extract logits for AGNEWS
                        else:
                            preds, _ = self.models['backbone'](inputs)
                        preds = torch.nn.functional.softmax(preds, dim=1)
                        preds = torch.max(preds, 1)[0]
                        uncertainties = 1.0 - preds
                        uncertainties_np = -uncertainties.cpu().numpy()  # 取负
                        scores= np.append(scores, uncertainties_np)
                    elif self.selection_method == "LossPrediction":
                        _, feature = self.models['backbone'](inputs)

        if self.selection_method == "MeanSTD":
            probs = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            # 将Tensor转移到CPU并转换为NumPy数组
            probs_np = probs.cpu().numpy()
            # 计算标准差
            sigma_c = np.std(probs_np, axis=0)
            # sigma_c = np.std(probs, axis=0)
            uncertainties = np.mean(sigma_c, axis=-1)
            uncertainties_np = -uncertainties # 取负
            scores = np.append(scores, uncertainties_np)
        elif self.selection_method == "BALDDropout":
            probs = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            pb = probs.mean(0)
            entropy1 = (-pb*torch.log(pb)).sum(1)
            entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
            uncertainties = entropy2 - entropy1
            uncertainties_np = -uncertainties.cpu().numpy()  # negative
            scores= np.append(scores, uncertainties_np)
        elif self.selection_method == "MarginDropout":
            probs = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            probs_sorted, _ = probs.sort(descending=True)
            uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
            uncertainties_np = -uncertainties.cpu().numpy()  # negative
            scores= np.append(scores, uncertainties_np)
        elif self.selection_method == "CONFDropout":
            probs = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            confs = probs.max(axis=1).values.cpu().numpy()
            scores = np.append(scores, confs)
        elif self.selection_method == "EntropyDropout":
            preds = self.predict_prob_dropout_split(self.unlabeled_set, selection_loader, n_drop=self.args.n_drop)
            preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
            entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
            scores = np.append(scores, entropys)

        return scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

    def predict_prob_dropout_split(self, to_predict_dataset, to_predict_dataloader, n_drop):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_dataset), len(self.args.target_list)]).to(self.args.device)

        # Create a dataloader object to load the dataset
        # to_predict_dataloader = torch.utils.data.DataLoader(to_predict_dataset, batch_size=self.args['batch_size'], shuffle=False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):

                evaluated_instances = 0
                # for i, data in enumerate(selection_loader):
                # inputs = data[0].to(self.args.device)
                for _, elements_to_predict in enumerate(to_predict_dataloader):
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict[0].to(self.args.device)
                    out = self.models['backbone'](elements_to_predict)[0]
                    # print(out)
                    pred = torch.nn.functional.softmax(out, dim=1)

                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice

        return probs


    def cal_dis_bim(self, x):
        self.models['backbone'].train()
        self.models['backbone'] = self.models['backbone'].to(self.args.device)

        nx = x.to(self.args.device)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape, device=self.args.device)  # Automatically set device during creation


        # First prediction to get the initial class labels
        out, _ = self.models['backbone'](nx)
        initial_pred = out.max(1)[1].detach()

        while True:
            out, _ = self.models['backbone'](nx + eta)
            current_pred = out.max(1)[1]

            if not torch.all(current_pred == initial_pred):
                break

            loss = torch.nn.functional.cross_entropy(out, initial_pred)
            # loss.requires_grad = True
            loss.backward()

            # eta.data += self.eps * torch.sign(nx.grad.data)
            # nx.grad.data.zero_()
            eta.data += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

        return (eta * eta).sum()

    def cal_dis_deepfool(self, nx):
        nx.requires_grad_()
        eta = torch.zeros(nx.shape, device=self.args.device)   # 修改为使用 zeros_like 以自动匹配形状

        out, _ = self.models['backbone'](nx + eta)
        py = out.max(1)[1]  
        ny = out.max(1)[1]  

        i_iter = 0
        max_iter = self.args.max_iter  # 假设max_iter是一个实例变量

        while (py == ny).all() and i_iter < max_iter:  # 修改条件检查，确保所有样本满足条件
            for i in range(nx.size(0)): 
                out[i, py[i]].backward(retain_graph=True) if i == nx.size(0) - 1 else out[i, py[i]].backward(retain_graph=True, create_graph=True)
                grad_np = nx.grad.data.clone()
                nx.grad.data.zero_()

                for j in range(out.shape[1]):  # 遍历所有类别
                    if j == py[i]:
                        continue
                    out[i, j].backward(retain_graph=True) if j == out.shape[1] - 1 else out[i, j].backward(retain_graph=True, create_graph=True)
                    grad_i = nx.grad.data.clone()
                    nx.grad.data.zero_()

                    wi = grad_i - grad_np
                    fi = out[i, j] - out[i, py[i]]
                    value_l = torch.full((nx.size(0),), float('inf'), device=nx.device)
                    value_i = torch.abs(fi) / torch.norm(wi.flatten())

                    if (value_i < value_l).any():
                        ri = (value_i / torch.norm(wi.flatten())) * wi
                        # print(ri.shape)
                        # print(eta.shape)
                        eta += ri.clone()  # update

            nx.grad.data.zero_()
            out, _ = self.models['backbone'](nx + eta)
            py = out.max(1)[1]
            i_iter += 1

        return (eta * eta).sum()
    