import numpy as np
# from sklearn.mixture import GaussianMixture
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import copy
# from utils import ova_ent, compute_roc,compute_S_ID
# from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from .almethod import ALMethod

def compute_S_ID(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    logits_open, _ = torch.max(logits_open, dim=2)
    logits_open  = logits_open[:,0]
    L_c = 2.5*(1+logits_open *torch.log(logits_open + 1e-8))
    return L_c

def get_sequence(args, tmp_data, query_num):
    ood_list = []
    tmp_seq = np.argsort(tmp_data)[0][:query_num]
    Un_Value = tmp_data[0]
    query_Index = tmp_data[1]
    labelArr = tmp_data[2]
    
    Un_Value = [Un_Value[i] for i in tmp_seq]
    query_Index = [query_Index[i] for i in tmp_seq]
    query_Index = list(map(int, query_Index))
    labelArr = [labelArr[i] for i in tmp_seq]

    for i in range(len(labelArr)):
        if labelArr[i] >= args.num_IN_class:
            ood_list.append(int(query_Index[i]))
    
    return ood_list, query_Index


def get_sequence_back(args, tmp_data, query_num):
    ood_list = []
    tmp_seq = np.argsort(tmp_data)[0][-query_num:]
    Un_Value = tmp_data[0]
    query_Index = tmp_data[1]
    labelArr = tmp_data[2]
    
    Un_Value = [Un_Value[i] for i in tmp_seq]
    query_Index = [query_Index[i] for i in tmp_seq]
    query_Index = list(map(int, query_Index))
    labelArr = [labelArr[i] for i in tmp_seq]

    for i in range(len(labelArr)):
        if labelArr[i] >= args.num_IN_class:
            ood_list.append(int(query_Index[i]))
    
    return ood_list, query_Index


class PAL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, wnet, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.cycle = cur_cycle
        self.wnet = wnet

    # def Compute_un(self, args, unlabeledloader, Len_labeled_ind_train, model, wnet, query, ID_need):
    def select(self, **kwargs):
        ID_need = self.args.need_ID
        # print('-'*40 + ' Start Sampling ' + '-'*40)
        OOD_need = self.args.n_query - ID_need
        self.wnet.eval()
        # model.eval()
        query = self.cycle
        Len_labeled_ind_train = len(self.I_index)
        self.models['ood_detection'].eval()
        model = self.models['ood_detection']

        temp_label = []
        Un_Value = []
        queryIndex = []
        labelArr = []
        ood_list = []
        unlabeledloader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        for batch_idx, data in enumerate(unlabeledloader):
            fea, labels, index = data
            labels = labels.tolist()
            labels = torch.Tensor(labels)
            inputs = fea
            index = index.cpu().tolist()
            if self.args.device:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
            outputs, outputs_open = model(inputs, method='PAL')

            # get the predicted label of label
            batch_size = inputs.shape[0]
            v_ij, predicted = outputs.max(1)
            predicted = predicted
            weight = self.wnet(outputs_open).cpu()
            s_ID = compute_S_ID(outputs_open)
            s_ID = torch.reshape(s_ID, (len(s_ID),1)).cpu()
            # else:
                # s_ID = torch.reshape(1-s_ID, (len(s_ID),1))
            Un = weight + self.args.miu*s_ID
            Un = Un.squeeze(1)
            Un = Un.cpu().tolist()

            if query > 0:
                tmp_lab = predicted.cpu().tolist()
                weight_list = weight.detach().numpy().tolist()
                tmp_sd = s_ID.tolist()
                # print(tmp_lab)
                # print()
                for ind, lab in enumerate(tmp_lab):
                    if lab == self.args.num_IN_class:
                        un_val = weight_list[ind][0] + self.args.miu*(1.0-tmp_sd[ind][0])
                        Un[ind] = un_val
            temp_label += predicted.cpu()
            Un_Value += Un
            queryIndex += index
            labelArr += list(np.array(labels.cpu()))
        
        tmp_data = np.vstack((Un_Value, queryIndex, labelArr, temp_label))

        Unlabel_ID = len(np.where(np.array(labelArr) < self.args.num_IN_class)[0])

        if query == 0:
            tmp_seq = np.argsort(tmp_data)[0][:self.args.n_query]
            Un_Value = tmp_data[0]
            query_Index = tmp_data[1]
            labelArr = tmp_data[2]
            temp_Label = tmp_data[3]
            Un_Value = [Un_Value[i] for i in tmp_seq]
            query_Index = [query_Index[i] for i in tmp_seq]
            Q_index = [int(num) for num in query_Index]
            return Q_index, Un_Value
        
        else:
            tmp_seq = np.argsort(tmp_data)[0][:self.args.n_query]
            Un_Value = tmp_data[0]
            query_Index = tmp_data[1]
            labelArr = tmp_data[2]
            temp_Label = tmp_data[3]
            Un_Value = [Un_Value[i] for i in tmp_seq]
            ori_query_Index = [query_Index[i] for i in tmp_seq]
            ori_Q_index = [int(num) for num in ori_query_Index]
        
            ood_back, q_back = get_sequence_back(self.args, tmp_data, self.args.n_query)
            prec_back = len(ood_back) / len(q_back)
            print(len(q_back))
            print(len(ood_back))
            print('back {self.args.n_query} ood_numer precision is:' + str(prec_back))
            Un_Value = tmp_data[0]
            query_Index = tmp_data[1]
            labelArr = tmp_data[2]
            temp_Label = tmp_data[3]

            targets_unk = temp_Label >= int(self.args.num_IN_class)
            targets_know = temp_Label < int(self.args.num_IN_class)
        
            ID_tmps = np.vstack((Un_Value[targets_know],query_Index[targets_know],labelArr[targets_know],temp_Label[targets_know]))
            OOD_tmps = np.vstack((Un_Value[targets_unk],query_Index[targets_unk],labelArr[targets_unk],temp_Label[targets_unk]))
            print('the pseudo label of ID is '+str(len(ID_tmps[0])))
            print('the pseudo label of OOD is '+str(len(OOD_tmps[0])))

            ood_back_ID, ID_back = get_sequence_back(self.args, ID_tmps, ID_need)
            ID_ID_back = list(set(ID_back)-set(ood_back_ID))
            prec_back_ID = len(ood_back_ID) / len(ID_back)
            print('Last OOD ood_numer precision is:' + str(prec_back_ID) + " and the number of OOD is " + str(len(ood_back_ID))+ " and the number of ID is " + str(len(ID_ID_back)))

            if OOD_need > 0:
                if len(OOD_tmps[0]) >= OOD_need:
                    ood_back_OOD, OOD_back = get_sequence(self.args, OOD_tmps, OOD_need)
                    OOD_ID_back = list(set(OOD_back)-set(ood_back_OOD))
                    prec_back_OOD = len(ood_back_OOD) / len(OOD_back)
                elif len(OOD_tmps[0]) > 0:
                    ood_back_OOD, OOD_back = get_sequence(self.args, OOD_tmps, len(OOD_tmps[0]))
                    OOD_ID_back = list(set(OOD_back)-set(ood_back_OOD))
                    prec_back_OOD = len(ood_back_OOD) / len(OOD_back)
                else:
                    OOD_ID_back = []
                    OOD_back = []
                    prec_back_OOD = 0
                    ood_back_OOD = []
                print('OOD ood_numer precision is:' + str(prec_back_OOD) + " and the number of OOD is " + str(len(ood_back_OOD)))
            else:
                OOD_back = []
                ood_back_OOD = []
                OOD_ID_back = []
            quey_back = ID_back + OOD_back

            precision = len(ID_ID_back) / int(ID_need)
            recall = (Len_labeled_ind_train + len(ID_ID_back))/(Len_labeled_ind_train + Unlabel_ID + len(OOD_ID_back))
            # print('-'*40 + ' Finished Sampling ' + '-'*40)
            # return ID_ID_back, quey_back, precision, recall # for now
        
            Q_index = [int(num) for num in quey_back]
            return ori_Q_index, Un_Value
