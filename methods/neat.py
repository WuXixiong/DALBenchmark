import numpy as np
import torch
from .almethod import ALMethod


class NEAT(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)


    def select(args, model, query, unlabeledloader, Len_labeled_ind_train, use_gpu, labeled_ind_train, invalidList,
                 unlabeled_ind_train, ordered_feature, ordered_label, labeled_index_to_label):
        index_knn = CIFAR100_EXTRACT_FEATURE_CLIP_new(labeled_ind_train + invalidList, unlabeled_ind_train, args,
                                                  ordered_feature, ordered_label)

        labelArr = []

        model.eval()
    #################################################################
        S_index = {}

        for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            _, outputs = model(data)

            v_ij, predicted = outputs.max(1)

            labelArr += list(np.array(labels.cpu().data))

            for i in range(len(data.data)):
                predict_class = predicted[i].detach()

                predict_value = np.array(v_ij.data.cpu())[i]

                predict_prob = outputs[i, :]

            # print(index[i])
            # tmp_index = index[i].item()
                tmp_index = index[i]

                true_label = np.array(labels.data.cpu())[i]

                S_index[tmp_index] = [true_label, predict_class, predict_value, predict_prob.detach().cpu()]

    #################################################################

    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # queryIndex 存放known class的地方
        queryIndex = []

        neighbor_unknown = {}

        detected_unknown = 0.0
        detected_known = 0.0

        for current_index in S_index:

            index_Neighbor, values = index_knn[current_index]

            true_label = S_index[current_index][0]

            count_known = 0.0
            count_unknown = 0.0

            for k in range(len(index_Neighbor)):

                n_index = index_Neighbor[k]

                if n_index in set(labeled_ind_train):
                    count_known += 1

                elif n_index in set(invalidList):
                    count_unknown += 1

            if count_unknown < count_known:

                queryIndex.append([current_index, count_known, true_label])

            else:
                detected_unknown += 1

        print("detected_unknown: ", detected_unknown)
        print("\n")

        queryIndex = sorted(queryIndex, key=lambda x: x[-2], reverse=True)

    #################################################################

        queryIndex = queryIndex[:2 * args.query_batch]

    #################################################################

    # if args.active_5 or args.active_5_reverse:

        queryIndex = active_learning_5(args, query, index_knn, queryIndex, S_index, labeled_index_to_label)

    # elif args.active_4:

    # queryIndex = active_learning_4(args, query, index_knn, queryIndex, S_index, labeled_index_to_label)

    #################################################################

        print(queryIndex[:20])

        final_chosen_index = []
        invalid_index = []

        for item in queryIndex[:args.query_batch]:

            num = item[0]

            num3 = item[-2]

            if num3 < args.known_class:

                final_chosen_index.append(int(num))

            elif num3 >= args.known_class:

                invalid_index.append(int(num))

    #################################################################

        precision = len(final_chosen_index) / args.query_batch

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

        recall = (len(final_chosen_index) + Len_labeled_ind_train) / (

            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

        return final_chosen_index, invalid_index, precision, recall
####ABOVE ARE METHOD ORIGINAL CODE


def cosDistance_two(unlabeled_features, labeled_features):
    # features: N*M matrix. N features, each features is M-dimension.
    unlabeled_features = F.normalize(unlabeled_features, dim=1)  # each feature's l2-norm should be 1

    labeled_features = F.normalize(labeled_features, dim=1)  # each feature's l2-norm should be 1

    similarity_matrix = torch.matmul(unlabeled_features, labeled_features.T)

    distance_matrix = 1.0 - similarity_matrix

    return distance_matrix

def CIFAR100_EXTRACT_FEATURE_CLIP_new(labeled_index, unlabeled_index, args, ordered_feature, ordered_label):
    ###########################################################

    labeled_final_feat, labled_labels = get_features(ordered_feature, ordered_label, labeled_index)

    unlabeled_final_feat, unlabled_labels = get_features(ordered_feature, ordered_label, unlabeled_index)
    ###########################################################

    order_to_index = {}

    index_to_order = {}
    for i in range(len(labeled_index)):
        order_to_index[i] = labeled_index[i]

        index_to_order[labeled_index[i]] = i

    ###################################################

    Dist = cosDistance_two(unlabeled_final_feat, labeled_final_feat)

    values, indices = torch.topk(Dist, k=args.k, dim=1, largest=False, sorted=True)

    print(indices)

    for k in range(indices.size()[0]):
        for j in range(indices.size()[1]):
            indices[k][j] = order_to_index[indices[k][j].item()]

    ###################################################

    index_knn = {}

    for i in range(len(unlabeled_index)):
        index_knn[unlabeled_index[i]] = (indices[i, :].cpu().numpy(), values[i, :])

    return index_knn


def get_features(ordered_feature, ordered_label, indices):
    features = ordered_feature[indices, :]

    labels = ordered_label[indices]

    return features, labels


def active_learning_5(args, query, index_knn, queryIndex, S_index, labeled_index_to_label):
    print("active learning 5")
    # S_index[n_index][0][1]

    new_query_index = []

    for i in range(len(queryIndex)):

        # all the indices for neighbors
        neighbors, values = index_knn[queryIndex[i][0]]

        predicted_prob = F.softmax(S_index[queryIndex[i][0]][-1], dim=-1).cuda()

        predicted_label = S_index[queryIndex[i][0]][-3]

        knn_labels_cnt = torch.zeros(args.known_class).cuda()

        for idx, neighbor in enumerate(neighbors):

            neighbor_labels = labeled_index_to_label[neighbor]

            test_variable_1 = 1.0 - values[idx]

            if neighbor_labels < args.known_class:
                knn_labels_cnt[neighbor_labels] += 1.0
        score = F.cross_entropy(knn_labels_cnt.unsqueeze(0), predicted_prob.unsqueeze(0), reduction='mean')

        score_np = score.cpu().item()

        # entropy = Categorical(probs = predicted_prob ).entropy().cpu().item()

        new_query_index.append(queryIndex[i] + [score_np])

    new_query_index = sorted(new_query_index, key=lambda x: x[-1], reverse=True)

    return new_query_index