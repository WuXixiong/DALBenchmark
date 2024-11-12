import numpy as np
import torch
from math import log
from finch import FINCH
import torch.nn.functional as F
from .almethod import  ALMethod

class EOAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, dataloaders, cluster_centers, cluster_labels, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.trainloader_C = dataloaders['ood']
        self.cluster_centers = cluster_centers
        # print(self.trainloader_C)
    
    def select(self, **kwargs):
        self.models['ood_detection'].eval()
        self.models['model_bc'].eval()
        labelArr, queryIndex, entropy_list, y_pred, unk_entropy_list = [], [], [], [], []
        feat_all = torch.zeros([1, 512]).cuda() # original it was [1, 128]
        # precision, recall = 0, 0
        selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        batch_num = len(selection_loader)
        # get cluster_centers
        # if len(queryIndex) > self.args.n_query:
        #     None,None,None,None = cluster_centers, embeddings, cluster_labels, queryIndex
        # cluster_centers, embeddings, cluster_labels, queryIndex = unknown_clustering(self.args, self.models['backbone'], self.models['model_bc'], self.trainloader_C, self.args.target_list)
        # select
        for i, data in enumerate(selection_loader):
            inputs = data[0].to(self.args.device)
            if i % self.args.print_freq == 0:
                print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                
            labels = data[1]
            index = data[2]
            labels = lab_conv(self.args.target_list, labels)
            # if use_gpu:
            #     data, labels = data.cuda(), labels.cuda()
            outputs, features = self.models['backbone'](inputs)
            softprobs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(softprobs, 1)
            queryIndex += index
            y_pred += list(np.array(predicted.cpu().data))
            labelArr += list(np.array(labels.cpu().data))
            feat_all = torch.cat([feat_all, features.data],0)
            out_open = self.models['model_bc'](features)
            out_open = out_open.view(outputs.size(0), 2, -1)

            ####### closed-set entropy score
            entropy_data = open_entropy(out_open)
            entropy_list.append(entropy_data.cpu().data)
             ####### distance-based entropy score
            if self.cluster_centers is not None:
                # cluster_centers = torch.tensor(self.cluster_centers)
                dists = torch.cdist(features, self.cluster_centers)
                similarity_scores_cj = torch.softmax(-dists, dim=1)
                pred_ent = -torch.sum(similarity_scores_cj*torch.log(similarity_scores_cj+1e-20), 1)
                unk_entropy_list.append(pred_ent.cpu().data)

        entropy_list = torch.cat(entropy_list).cpu()
        entropy_list = entropy_list / log(2)

        y_pred = np.array(y_pred)
        labelArr = torch.tensor(labelArr)
        labelArr_k = labelArr[y_pred < self.args.num_IN_class]

        if self.cluster_centers is not None:
            unk_entropy_list = torch.cat(unk_entropy_list).cpu()
            unk_entropy_list = unk_entropy_list / log(len(self.cluster_centers))
            entropy_list = entropy_list - unk_entropy_list

        embeddings = feat_all[1:].cpu().numpy()
        embeddings_k = embeddings[y_pred < self.args.num_IN_class]

        uncertaintyArr_k = entropy_list[y_pred < self.args.num_IN_class]
        queryIndex = torch.tensor(queryIndex)
        queryIndex_k = queryIndex[y_pred < self.args.num_IN_class]

        if not self.args.eoal_diversity:
            sorted_idx = uncertaintyArr_k.sort()[1][:self.args.n_query]
            selected_idx = queryIndex_k[sorted_idx]
            selected_gt = labelArr_k[sorted_idx]
            selected_gt = selected_gt.numpy()
            selected_idx = selected_idx.numpy()

        else:        
            labels_c, num_clust, _ = FINCH(embeddings_k, req_clust= len(self.args.target_list), verbose=True)
            tmp_var = 0
            while num_clust[tmp_var] > self.args.n_query:
                tmp_var += 1
            cluster_labels = labels_c[:, tmp_var]
            num_clusters = num_clust[tmp_var]

            rem = min(self.args.n_query, len(queryIndex_k))
            num_per_cluster = int(rem/num_clusters)
            selected_idx = []
            selected_gt = []

            ax = [0 for i in range(num_clusters)]
            while rem > 0:
                print("Remaining Budget to Sample:  ", rem)
                for cls in range(num_clusters):
                    temp_ent = uncertaintyArr_k[cluster_labels == cls]
                    temp_index = queryIndex_k[cluster_labels == cls]
                    temp_gt = labelArr_k[cluster_labels == cls]
                    if rem >= num_per_cluster:
                        sorted_idx = temp_ent.sort()[1][ax[cls]:ax[cls]+min(num_per_cluster, len(temp_ent))]
                        ax[cls] += len(sorted_idx)
                        rem -= len(sorted_idx)
                    else:
                        sorted_idx = temp_ent.sort()[1][ax[cls]:ax[cls]+min(rem, len(temp_ent))]
                        ax[cls] += len(sorted_idx)
                        rem -= len(sorted_idx)
                    q_idxs = temp_index[sorted_idx.cpu()]
                    selected_idx.extend(list(q_idxs.numpy()))
                    gt_cls = temp_gt[sorted_idx.cpu()]
                    selected_gt.extend(list(gt_cls.numpy()))
            print("clustering finished")
            selected_gt = np.array(selected_gt)
            selected_idx = np.array(selected_idx)

        if len(selected_gt) < self.args.n_query:
            rem_budget = self.args.n_query - len(set(selected_idx))
            print("Not using all the budget...")
            uncertaintyArr_u = entropy_list[y_pred >= self.args.num_IN_class]
            queryIndex_u = queryIndex[y_pred >= self.args.num_IN_class]
            queryIndex_u = np.array(queryIndex_u)
            labelArr_u = labelArr[y_pred >= self.args.num_IN_class]
            labelArr_u = np.array(labelArr_u)
            tmp_data = np.vstack((queryIndex_u, labelArr_u)).T
            print("Choosing from the K+1 classifier's rejected samples...")
            sorted_idx_extra = uncertaintyArr_u.sort()[1][:rem_budget]
            tmp_data = tmp_data.T
            rand_idx = tmp_data[0][sorted_idx_extra.cpu().numpy()]
            rand_LabelArr = tmp_data[1][sorted_idx_extra.cpu().numpy()]
            selected_gt = np.concatenate((selected_gt, rand_LabelArr))
            selected_idx = np.concatenate((selected_idx, rand_idx))

        # precision = len(np.where(selected_gt < args.known_class)[0]) / len(selected_gt)
        # recall = (len(np.where(selected_gt < args.known_class)[0]) + Len_labeled_ind_train) / (
        #             len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
        # queryIndex, invalidIndex, Precision[query], Recall[query]
        # return selected_idx[np.where(selected_gt < args.known_class)[0]], selected_idx[np.where(selected_gt >= args.known_class)[0]]
        return selected_idx, None # if use tmp_data as query score, then it sometimes return local variable error

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1)
    return ent_open

def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label), dtype=int)
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = int(len(knownclass))     
    return label_convert

def unknown_clustering(args, model, model_bc, trainloader_C, knownclass):
    model.eval()
    model_bc.eval()
    feat_all = torch.zeros([1, 512], device=args.device) # original 128
    labelArr, labelArr_true, queryIndex, y_pred = [], [], [], []
    
    for i, data in enumerate(trainloader_C):
        inputs = data[0]
        labels = data[1]
        index = data[2]
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        index = index.to(args.device)

        labels_true = labels
        labelArr_true += list(labels_true.cpu().data.numpy())
        labels = lab_conv(knownclass, labels)
        # if use_gpu:
        #     data, labels = data.cuda(), labels.cuda()
        outputs, features = model(inputs)
        softprobs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(softprobs, 1)
        y_pred += list(predicted.cpu().data.numpy())
        feat_all = torch.cat([feat_all, features.data], 0)
        queryIndex += index
        labelArr += list(labels.cpu().data.numpy())

    queryIndex = np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in queryIndex])
    queryIndex = np.array(queryIndex)
    y_pred = np.array(y_pred)

    embeddings = feat_all[1:].cpu().numpy()
    _, _, req_c = FINCH(embeddings, req_clust= args.w_unk_cls * len(knownclass), verbose=False)
    cluster_labels = req_c
    # Convert back to tensors after clustering
    embeddings = torch.tensor(embeddings, device='cuda')
    labelArr_true = torch.tensor(labelArr_true)
    queryIndex = torch.tensor(queryIndex)
    cluster_labels = torch.tensor(cluster_labels)
    cluster_centers = calculate_cluster_centers(embeddings, cluster_labels)
    return cluster_centers, embeddings, cluster_labels, queryIndex

def calculate_cluster_centers(features, cluster_labels):
    unique_clusters = torch.unique(cluster_labels)
    cluster_centers = torch.zeros((len(unique_clusters), features.shape[1])).cuda()
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = torch.where(cluster_labels == cluster_id)[0]
        cluster_features = features[cluster_indices]
        # Calculate the center of the cluster using the mean of features
        cluster_center = torch.mean(cluster_features, dim=0)
        cluster_centers[i] = cluster_center
    return cluster_centers
