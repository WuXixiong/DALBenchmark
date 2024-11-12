import numpy as np
from .almethod import ALMethod
import torch
from sklearn.random_projection import GaussianRandomProjection
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Subset

class VESSAL(ALMethod):
#     def __init__(self, X, Y, idxs_lb, net, args):
#         super(VESSAL, self).__init__(X, Y, idxs_lb, net, args)
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.skipped = []

        # modify in the load_split_data.py in the function get_dataset(args, trial):
        # if self.args.dataset == 'CLOW' or self.args.dataset == 'clip':
        #     self.transformer = GaussianRandomProjection(n_components=2560)
        self.X = []
        self.Y = []
        self.zeta = self.args.zeta
        subset = Subset(unlabeled_dst, U_index)
        for idx in range(len(subset)):
            data, target, _ = self.unlabeled_dst[idx]  # 获取数据和标签
            self.X.append(data)  # 将数据添加到 pool_data_dropout 列表
            self.Y.append(target)  # 将标签添加到 pool_target_dropout 列表
        self.X = torch.stack(self.X)  # 拼接成形如 (batch_size, channels, height, width) 的张量
        self.Y = torch.tensor(self.Y)  # 转换为形如 (batch_size,) 的张量
        self.idxs_lb = np.zeros(len(unlabeled_dst), dtype=bool)
        self.idxs_lb[I_index] = True

    # just in case values get too big, sometimes happens
    def inf_replace(self, mat):
        mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
        return mat

    def streaming_sampler(self, samps, k, early_stop=False, streaming_method='det', \
                        cov_inv_scaling=100, embs="grad_embs"):
        inds = []
        skipped_inds = []
        if embs == "penultimate":
            samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        dim = samps.shape[-1]
        rank = samps.shape[-2]

        covariance = torch.zeros(dim,dim).cuda()
        covariance_inv = cov_inv_scaling * torch.eye(dim).cuda()
        samps = torch.tensor(samps)
        samps = samps.cuda()

        for i, u in enumerate(samps):
            if i % 1000 == 0: print(i, len(inds), flush=True)
            if rank > 1: u = torch.Tensor(u).t().cuda()
            else: u = u.view(-1, 1)
            
            # get determinantal contribution (matrix determinant lemma)
            if rank > 1:
                norm = torch.abs(torch.det(u.t() @ covariance_inv @ u))
            else:
                norm = torch.abs(u.t() @ covariance_inv @ u)

            ideal_rate = (k - len(inds))/(len(samps) - (i))
            # just average everything together: \Sigma_t = (t-1)/t * A\{t-1}  + 1/t * x_t x_t^T
            covariance = (i/(i+1))*covariance + (1/(i+1))*(u @ u.t())

            self.zeta = (ideal_rate/(torch.trace(covariance @ covariance_inv))).item()

            pu = np.abs(self.zeta) * norm

            if np.random.rand() < pu.item():
                inds.append(i)
                if early_stop and len(inds) >= k:
                    break
                
                # woodbury update to covariance_inv
                inner_inv = torch.inverse(torch.eye(rank).cuda() + u.t() @ covariance_inv @ u)
                inner_inv = self.inf_replace(inner_inv)
                covariance_inv = covariance_inv - covariance_inv @ u @ inner_inv @ u.t() @ covariance_inv
            else:
                skipped_inds.append(i)

        return inds, skipped_inds


    def get_valid_candidates(self):
        # skipped = np.zeros(self.n_pool, dtype=bool)
        skipped = np.zeros(len(self.unlabeled_dst), dtype=bool)
        skipped[self.skipped] = True
        if self.args.single_pass:
            valid = ~self.idxs_lb & ~skipped & self.allowed 
        else:
            valid = ~self.idxs_lb 
        return valid 


    def select(self):#, num_round=0):
        n = self.args.n_query

        # valid = self.get_valid_candidates()
        # idxs_unlabeled = np.arange(self.n_pool)[valid]
        idxs_unlabeled = self.U_index

        rank = self.args.rank
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        if self.args.embs == "penultimate":
            gradEmbedding = self.get_embedding(selection_loader, self.Y).numpy()
            # print('pen embedding shape: {}'.format(gradEmbedding.shape))
        else:
            gradEmbedding = self.get_exp_grad_embedding(selection_loader, self.Y, rank=rank).numpy()
            # print('gradient embedding shape: {}'.format(gradEmbedding.shape))

        early_stop = self.args.early_stop 
        cov_inv_scaling = self.args.cov_inv_scaling
       
        start_time = time.time()
        chosen, skipped = self.streaming_sampler(gradEmbedding, n, early_stop=early_stop, \
            cov_inv_scaling=cov_inv_scaling, embs = self.args.embs)
        print(len(idxs_unlabeled), len(chosen), flush=True)
        print('compute time (sec):', time.time() - start_time, flush=True)
        print('chosen: {}, skipped: {}, n:{}'.format(len(chosen),len(skipped),n), flush=True)

        # If more than n samples were selected, take the first n.
        if len(chosen) > n:
            chosen = chosen[:n]
        self.skipped.extend([idxs_unlabeled[idx] for idx in skipped])
        # self.skipped.extend(idxs_unlabeled[skipped])
        # 将 idxs_unlabeled 转换为 numpy 数组
        idxs_unlabeled_np = np.array(idxs_unlabeled)
        # 使用 chosen 列表进行索引
        result = idxs_unlabeled_np[chosen]
        #result = idxs_unlabeled[chosen]
        if self.args.fill_random:
            # If less than n samples where selected, fill is with random samples.
            if len(chosen) < n:
                labelled = np.copy(self.idxs_lb)
                labelled[idxs_unlabeled[chosen]] = True
                remaining_unlabelled = np.arange(len(self.U_index))[~labelled]
                n_random = n - len(chosen)
                fillers = remaining_unlabelled[np.random.permutation(len(remaining_unlabelled))][:n_random]
                result = np.concatenate([idxs_unlabeled[chosen], fillers], axis=0)

        return result, None # query index, score

    def get_embedding(self, loader_te, Y):
        # loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
        #                     shuffle=False, **self.args['loader_te_args'])
        # self.clf.eval()
        model = self.models['backbone']
        embedding = torch.zeros([len(Y), model.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding

    def get_exp_grad_embedding(self, loader_te, Y, probs=[], model=[], rank=1):
        if type(model) == list:
            model = self.models['backbone']
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        if rank == 0 or rank > nLab: rank = nLab
        embedding = np.zeros([len(Y), rank, embDim * nLab])
        for ind in range(rank):
            # loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
            #                 shuffle=False, **self.args['loader_te_args'])
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    cout, out = model(x)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()

                    for j in range(len(y)):
                        order = np.argsort(batchProbs[j])[::-1]
                        probs = batchProbs[j][order]
                        fakeLab = order[ind]
                        if idxs[j] >= len(Y): # prevent to beyond bound
                                break
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - probs[c])
                            else:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * probs[c])
                        probs = probs / np.sum(probs)
                        embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(probs[ind])
        return torch.Tensor(embedding)