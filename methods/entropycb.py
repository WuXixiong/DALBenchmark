from .almethod import ALMethod
import torch
import numpy as np
import cvxpy as cp

class EntropyCB(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.cur_cycle = cur_cycle
        self.I_index = I_index
        self.unlabeled_dst = unlabeled_dst

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

        preds = []
        U = np.array([])
        batch_num = len(selection_loader)
        print("| Calculating uncertainty of Unlabeled set")
        for i, data in enumerate(selection_loader):
            if i % self.args.print_freq == 0:
                print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))

            with torch.no_grad():
            # Extract input based on whether the dataset is text or image
                if self.args.textset:
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    pred = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask).logits
                else:
                    inputs = data[0].to(self.args.device)
                    pred, _ = self.models['backbone'](inputs)

                pred = torch.nn.functional.softmax(pred, dim=1).cpu().numpy()
                # entropys = (np.log(preds + 1e-6) * preds).sum(axis=1) # U = (probs*log_probs).sum(1)
                probs = (np.log(pred + 1e-6) * pred).sum(axis=1)
                preds.append(pred)
                U = np.append(U, probs)

        preds = np.vstack(preds) # convert preds to a 2-d nparray
        b=self.args.n_query # b=n
        # N=len(self.U_index) # N=len(idxs_unlabeled), but here N should be the data (a batch of unlabelled data)
        N=len(U)
        total_label = []
        L1_DISTANCE=[]
        L1_Loss=[]
        ENT_Loss=[]
        # Adaptive counts of samples per cycle
        labelled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        if self.args.textset:
            labelled_classes = [labelled_subset[i]['labels'] for i in range(len(labelled_subset))]
        else:
            labelled_classes = [labelled_subset[i][1] for i in range(len(labelled_subset))]
        # _, counts = np.unique(labelled_classes, return_counts=True)
        labelled_classes = np.array(labelled_classes)
        counts = np.bincount(labelled_classes, minlength=self.args.num_IN_class)
        class_threshold=int((2*self.args.n_query+(self.cur_cycle+1)*self.args.n_query)/int(self.args.num_IN_class))
        class_share=class_threshold-counts
        samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(int(self.args.num_IN_class),1)
        if self.args.dataset == 'CIFAR10':
            lamda=0.6
        elif self.args.dataset == 'CIFAR100':
            lamda=2
        elif self.args.dataset == 'TINYIMAGENET':
            lamda=3
            # maybe need another parameter for imagenet
        else:
            lamda=1

        for lam in [lamda]:

            z=cp.Variable((N,1),boolean=True)
            constraints = [sum(z) == b]
            cost = z.T @ U + lam * cp.norm1(preds.T @ z - samples_share)
            objective = cp.Minimize(cost)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.GUROBI, verbose=True, TimeLimit=1000)
            print('Optimal value with gurobi : ', problem.value)
            print(problem.status)
            print("A solution z is")
            print(z.value.T)
            lb_flag = np.array(z.value.reshape(1, N)[0], dtype=bool)
            indices = np.where(lb_flag == 1)[0]
			# -----------------Stats of optimization---------------------------------
            # reset the variables as mqnet's
            n = self.args.n_query
            num_classes = int(self.args.num_IN_class)

            ENT_Loss.append(np.matmul(z.value.T, U))
            print('ENT LOSS= ', ENT_Loss)
            threshold = (2 * n / num_classes) + (self.cur_cycle + 1) * n / num_classes
            round=self.cur_cycle+1
            selected_subset = torch.utils.data.Subset(self.unlabeled_dst, indices)
            if self.args.textset:
                selected_classes = [selected_subset[i]['labels'] for i in range(len(selected_subset))]
            else:    
                selected_classes = [selected_subset[i][1] for i in range(len(selected_subset))] # self.Y[idxs_unlabeled[lb_flag]]
            # labeled_classes # labeled_classes=self.Y[self.idxs_lb]
            freq = torch.histc(torch.FloatTensor(selected_classes), bins=num_classes)+torch.histc(torch.FloatTensor(labelled_classes), bins=num_classes)
            L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * (2 * n + round * n) * (num_classes - 1))).item()
            print('Lambda = ',lam)
            L1_DISTANCE.append(L1_distance)
            L1_Loss_term=np.linalg.norm(np.matmul(preds.T,z.value) - samples_share, ord=1)
            L1_Loss.append(L1_Loss_term)

            print('L1 Loss = ')
            for i in L1_Loss:
                print('%.3f' %i)
            print('L1_distance = ')
            for j in L1_DISTANCE:
                print('%.3f' % j)
            print('ENT LOSS = ')
            for k in ENT_Loss:
                print('%.3f' % k)

        return indices

    def select(self, **kwargs):
        # selected_indices, scores = self.run()
        selected_indices = self.rank_uncertainty()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        scores = list(np.ones(len(Q_index)))  # equally assign 1 (meaningless)

        return Q_index, scores
