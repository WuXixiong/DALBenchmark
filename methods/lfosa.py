"""
This implementation is primarily based on the official open-source code of:
GitHub: https://github.com/ningkp/LfOSA
Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Ning_Active_Learning_for_Open-Set_Annotation_CVPR_2022_paper.html

@inproceedings{ning2022active,
  title={Active learning for open-set annotation},
  author={Ning, Kun-Peng and Zhao, Xun and Li, Yu and Huang, Sheng-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={41--49},
  year={2022}
}
"""
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from .almethod import ALMethod


class LFOSA(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
    
    # def AV_sampling_temperature(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    def select(self, **kwargs):
        Len_labeled_ind_train = len(self.I_index)
        self.models['ood_detection'].eval()
        # scores = np.array([])
        with torch.no_grad():
            selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            queryIndex = []
            labelArr = []
            uncertaintyArr = []
            S_ij = {}
            batch_num = len(selection_loader)
            for i, data in enumerate(selection_loader):
                if self.args.textset:
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    outputs = self.models['ood_detection'](input_ids=input_ids, attention_mask=attention_mask)
                    outputs = outputs.logits # logists
                    labels = data['labels']
                    index = data['index']
                    # hidden_states = outputs.hidden_states
                    # last_hidden_state = hidden_states[-1]
                    # features = last_hidden_state[:, 0, :]

                else: # for images
                    inputs = data[0].to(self.args.device)
                    if i % self.args.print_freq == 0:
                        print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                    
                    labels = data[1]
                    index = data[2]

                    outputs, _ = self.models['ood_detection'](inputs)

                labelArr += list(np.array(labels.cpu().data))
                # activation value based
                v_ij, predicted = outputs.max(1)
                for j in range(len(predicted.data)):
                    tmp_class = np.array(predicted.data.cpu())[j]
                    tmp_index = index[j]
                    tmp_label = np.array(labels.data.cpu())[j]
                    tmp_value = np.array(v_ij.data.cpu())[j]
                    if tmp_class not in S_ij:
                        S_ij[tmp_class] = []
                    S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])


        # fit a two-component GMM for each class
        tmp_data = []
        for tmp_class in S_ij:
            S_ij[tmp_class] = np.array(S_ij[tmp_class])
            activation_value = S_ij[tmp_class][:, 0]
            if len(activation_value) < 2:
                continue
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(np.array(activation_value).reshape(-1, 1))
            prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
            # The probability of getting for the 'known' category
            prob = prob[:, gmm.means_.argmax()]
            # If the category is UNKNOWN, it is 0
            if tmp_class == self.args.num_IN_class:
                prob = [0]*len(prob)
                prob = np.array(prob)

            if len(tmp_data) == 0:
                tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
            else:
                tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))
        
        # tmp_data = tmp_data.T
        # scores = tmp_data[2]
        # return scores


        tmp_data = tmp_data[np.argsort(tmp_data[:, 0])] # scores
        tmp_data = tmp_data.T
        queryIndex = tmp_data[2][-self.args.n_query:].astype(int)
        return queryIndex, tmp_data
        # labelArr = tmp_data[3].astype(int)
        # queryLabelArr = tmp_data[3][-args.query_batch:]
        # precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
        # recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
        #     len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
        # return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall