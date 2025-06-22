import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from .almethod import ALMethod

class LFOSA(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.is_multilabel = getattr(args, 'is_multilabel', False)

    def select(self, **kwargs):
        Len_labeled_ind_train = len(self.I_index)
        self.models['ood_detection'].eval()
        
        with torch.no_grad():
            selection_loader = torch.utils.data.DataLoader(
                self.unlabeled_set, 
                batch_size=self.args.test_batch_size, 
                num_workers=self.args.workers
            )
            queryIndex = []
            labelArr = []
            uncertaintyArr = []
            S_ij = {}
            batch_num = len(selection_loader)
            
            print(f"LFOSA processing ({'multi-label' if self.is_multilabel else 'single-label'}) samples...")
            
            for i, data in enumerate(selection_loader):
                if self.args.textset:
                    input_ids = data['input_ids'].to(self.args.device)
                    attention_mask = data['attention_mask'].to(self.args.device)
                    outputs = self.models['ood_detection'](input_ids=input_ids, attention_mask=attention_mask)
                    outputs = outputs.logits  # logits
                    labels = data['labels']
                    index = data['index']
                else:  # for images
                    inputs = data[0].to(self.args.device)
                    if i % self.args.print_freq == 0:
                        print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                    
                    labels = data[1]
                    index = data[2]
                    outputs, _ = self.models['ood_detection'](inputs)

                labelArr += list(np.array(labels.cpu().data))
                
                # Process activation values based on task type
                if self.is_multilabel:
                    # For multi-label: use sigmoid and process each class independently
                    probs = torch.sigmoid(outputs)
                    
                    # Get activation values and predictions for each class
                    for class_idx in range(outputs.shape[1]):
                        class_probs = probs[:, class_idx]
                        class_logits = outputs[:, class_idx]
                        
                        # Use logit values as activation values for multi-label
                        # This preserves the confidence information before sigmoid
                        v_ij = class_logits
                        
                        # Consider samples with probability > threshold as positive predictions
                        threshold = getattr(self.args, 'lfosa_multilabel_threshold', 0.5)
                        predicted_positive = (class_probs > threshold).long()
                        
                        for j in range(len(predicted_positive)):
                            tmp_index = index[j]
                            tmp_label = np.array(labels.data.cpu())[j]
                            tmp_value = np.array(v_ij.data.cpu())[j]
                            
                            # Create composite class key for multi-label
                            # Format: "class_idx_prediction" (e.g., "0_1" for class 0 predicted as positive)
                            tmp_class = f"{class_idx}_{predicted_positive[j].item()}"
                            
                            if tmp_class not in S_ij:
                                S_ij[tmp_class] = []
                            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])
                else:
                    # For single-label: original logic
                    v_ij, predicted = outputs.max(1)
                    for j in range(len(predicted.data)):
                        tmp_class = np.array(predicted.data.cpu())[j]
                        tmp_index = index[j]
                        tmp_label = np.array(labels.data.cpu())[j]
                        tmp_value = np.array(v_ij.data.cpu())[j]
                        if tmp_class not in S_ij:
                            S_ij[tmp_class] = []
                        S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

        # Fit a two-component GMM for each class/prediction combination
        tmp_data = []
        print(f"LFOSA GMM fitting for {len(S_ij)} class groups...")
        
        for tmp_class in S_ij:
            S_ij[tmp_class] = np.array(S_ij[tmp_class])
            activation_value = S_ij[tmp_class][:, 0]
            
            if len(activation_value) < 2:
                continue
                
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(np.array(activation_value).reshape(-1, 1))
            prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
            
            # The probability of getting for the 'known' category
            prob = prob[:, gmm.means_.argmax()]
            
            if self.is_multilabel:
                # For multi-label: handle OOD detection differently
                # Parse the composite class key
                if '_' in str(tmp_class):
                    class_parts = str(tmp_class).split('_')
                    class_idx = int(class_parts[0])
                    prediction = int(class_parts[1])
                    
                    # If the class index is beyond in-distribution classes, treat as unknown
                    if class_idx >= self.args.num_IN_class:
                        prob = [0] * len(prob)
                        prob = np.array(prob)
                    # For negative predictions (prediction=0), reduce probability
                    elif prediction == 0:
                        negative_scale = getattr(self.args, 'lfosa_negative_scale', 0.5)
                        prob = prob * negative_scale
                else:
                    # Fallback for non-composite keys
                    if int(tmp_class) >= self.args.num_IN_class:
                        prob = [0] * len(prob)
                        prob = np.array(prob)
            else:
                # For single-label: original logic
                # If the category is UNKNOWN, it is 0
                if tmp_class == self.args.num_IN_class:
                    prob = [0] * len(prob)
                    prob = np.array(prob)

            if len(tmp_data) == 0:
                tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
            else:
                tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

        if len(tmp_data) == 0:
            print("Warning: No valid data for LFOSA selection, using random selection")
            # Fallback to random selection if no valid data
            all_indices = list(range(len(self.unlabeled_set)))
            queryIndex = np.random.choice(all_indices, size=min(self.args.n_query, len(all_indices)), replace=False)
            scores = np.ones(len(queryIndex))
            return queryIndex, scores

        # Sort by scores (probability of being known)
        tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]  # scores
        tmp_data = tmp_data.T
        
        # Select samples with lowest probability (highest uncertainty)
        queryIndex = tmp_data[2][-self.args.n_query:].astype(int)
        scores = tmp_data[0][-self.args.n_query:]
        
        print(f"LFOSA selection completed:")
        print(f"  Selected {len(queryIndex)} samples")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Task type: {'Multi-label' if self.is_multilabel else 'Single-label'}")
        
        return queryIndex, scores

    def analyze_selection(self, S_ij, queryIndex, scores):
        """
        Optional method to analyze the selection quality.
        """
        print(f"LFOSA selection analysis:")
        print(f"  Number of class groups: {len(S_ij)}")
        
        if self.is_multilabel:
            # Analyze class distribution for multi-label
            class_dist = {}
            for class_key in S_ij.keys():
                if '_' in str(class_key):
                    class_parts = str(class_key).split('_')
                    class_idx = int(class_parts[0])
                    prediction = int(class_parts[1])
                    
                    if class_idx not in class_dist:
                        class_dist[class_idx] = {'positive': 0, 'negative': 0}
                    
                    if prediction == 1:
                        class_dist[class_idx]['positive'] += len(S_ij[class_key])
                    else:
                        class_dist[class_idx]['negative'] += len(S_ij[class_key])
            
            print(f"  Class prediction distribution: {class_dist}")
            
        else:
            # Analyze class distribution for single-label
            class_sizes = {k: len(v) for k, v in S_ij.items()}
            print(f"  Class sizes: {class_sizes}")
        
        print(f"  Average selection score: {scores.mean():.4f}")
        print(f"  Selection score std: {scores.std():.4f}")

    def get_class_statistics(self, S_ij):
        """
        Get statistics about class distribution in the data.
        """
        stats = {
            'total_samples': sum(len(samples) for samples in S_ij.values()),
            'num_classes': len(S_ij),
            'samples_per_class': {k: len(v) for k, v in S_ij.items()}
        }
        
        if self.is_multilabel:
            # Additional statistics for multi-label
            positive_samples = sum(len(samples) for k, samples in S_ij.items() 
                                 if '_1' in str(k))
            negative_samples = sum(len(samples) for k, samples in S_ij.items() 
                                 if '_0' in str(k))
            stats['positive_predictions'] = positive_samples
            stats['negative_predictions'] = negative_samples
            stats['positive_ratio'] = positive_samples / (positive_samples + negative_samples) if (positive_samples + negative_samples) > 0 else 0
        
        return stats