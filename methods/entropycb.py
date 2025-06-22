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
        self.is_multilabel = getattr(args, 'is_multilabel', False)

    def compute_entropy(self, probs):
        """
        Compute entropy for both single-label and multi-label cases.
        
        Args:
            probs: Probability matrix of shape (N, num_classes)
            
        Returns:
            entropies: Array of entropy values for each sample
        """
        if self.is_multilabel:
            # For multi-label: compute binary entropy for each class and sum
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            eps = 1e-6
            binary_entropy = -(probs * np.log(probs + eps) + 
                             (1 - probs) * np.log(1 - probs + eps))
            # Sum entropy across all classes
            entropies = binary_entropy.sum(axis=1)
        else:
            # For single-label: standard categorical entropy
            # H = -sum(p * log(p))
            entropies = -(np.log(probs + 1e-6) * probs).sum(axis=1)
        
        return entropies

    def get_labeled_class_counts(self):
        """
        Get class distribution in the current labeled set.
        Modified to support both single-label and multi-label tasks.
        """
        num_classes = self.args.num_IN_class
        labelled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        
        if self.is_multilabel:
            # For multi-label: count each label occurrence
            label_counts = np.zeros(num_classes)
            for i in range(len(labelled_subset)):
                if self.args.textset:
                    labels = labelled_subset[i]['labels']
                else:
                    labels = labelled_subset[i][1]
                
                # Handle different label formats
                if isinstance(labels, torch.Tensor):
                    if labels.dim() > 0 and labels.shape[0] > 1:
                        # Multi-hot encoded
                        active_labels = torch.nonzero(labels, as_tuple=True)[0].cpu().numpy()
                        label_counts[active_labels] += 1
                    else:
                        # Single label in tensor format
                        label_counts[labels.item()] += 1
                elif isinstance(labels, (list, np.ndarray)):
                    # List of active labels
                    for label in labels:
                        if 0 <= label < num_classes:
                            label_counts[label] += 1
                else:
                    # Single label
                    if 0 <= labels < num_classes:
                        label_counts[labels] += 1
            
            counts = label_counts
        else:
            # For single-label: standard approach
            if self.args.textset:
                labelled_classes = [labelled_subset[i]['labels'] for i in range(len(labelled_subset))]
            else:
                labelled_classes = [labelled_subset[i][1] for i in range(len(labelled_subset))]
            labelled_classes = np.array(labelled_classes)
            counts = np.bincount(labelled_classes, minlength=num_classes)
            
        return counts

    def get_selected_class_counts(self, indices):
        """
        Get class distribution in the selected samples.
        """
        num_classes = self.args.num_IN_class
        selected_subset = torch.utils.data.Subset(self.unlabeled_dst, indices)
        
        if self.is_multilabel:
            # For multi-label: count each label occurrence
            label_counts = np.zeros(num_classes)
            for i in range(len(selected_subset)):
                if self.args.textset:
                    labels = selected_subset[i]['labels']
                else:
                    labels = selected_subset[i][1]
                
                # Handle different label formats
                if isinstance(labels, torch.Tensor):
                    if labels.dim() > 0 and labels.shape[0] > 1:
                        # Multi-hot encoded
                        active_labels = torch.nonzero(labels, as_tuple=True)[0].cpu().numpy()
                        label_counts[active_labels] += 1
                    else:
                        label_counts[labels.item()] += 1
                elif isinstance(labels, (list, np.ndarray)):
                    for label in labels:
                        if 0 <= label < num_classes:
                            label_counts[label] += 1
                else:
                    if 0 <= labels < num_classes:
                        label_counts[labels] += 1
            
            selected_classes = label_counts
        else:
            # For single-label: standard approach
            if self.args.textset:
                selected_classes = [selected_subset[i]['labels'] for i in range(len(selected_subset))]
            else:    
                selected_classes = [selected_subset[i][1] for i in range(len(selected_subset))]
            selected_classes = np.array(selected_classes)
            
        return selected_classes

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size, 
            num_workers=self.args.workers
        )

        preds = []
        U = np.array([])
        batch_num = len(selection_loader)
        print(f"| Calculating uncertainty of Unlabeled set ({'multi-label' if self.is_multilabel else 'single-label'})")
        
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

                # Convert logits to probabilities based on task type
                if self.is_multilabel:
                    # For multi-label: use sigmoid probabilities
                    pred = torch.sigmoid(pred)
                    
                    # Optional: normalize probabilities for better optimization
                    if getattr(self.args, 'entropycb_normalize_multilabel', True):
                        pred = pred / (pred.sum(dim=-1, keepdim=True) + 1e-10)
                else:
                    # For single-label: use softmax probabilities
                    pred = torch.nn.functional.softmax(pred, dim=1)
                
                pred = pred.cpu().numpy()
                
                # Compute entropy based on task type
                entropies = self.compute_entropy(pred)
                
                preds.append(pred)
                U = np.append(U, entropies)

        preds = np.vstack(preds)  # convert preds to a 2-d nparray
        b = self.args.n_query  # b=n
        N = len(U)
        
        # Initialize tracking lists
        total_label = []
        L1_DISTANCE = []
        L1_Loss = []
        ENT_Loss = []
        
        # Get current labeled class counts
        counts = self.get_labeled_class_counts()
        
        # Calculate class balance targets
        class_threshold = int((2 * self.args.n_query + (self.cur_cycle + 1) * self.args.n_query) / int(self.args.num_IN_class))
        class_share = class_threshold - counts
        
        if self.is_multilabel:
            # For multi-label: adjust sharing strategy
            samples_share = np.array([max(0, c) for c in class_share]).reshape(int(self.args.num_IN_class), 1)
            
            # Scale down for multi-label to account for multiple labels per sample
            multilabel_scale = getattr(self.args, 'entropycb_multilabel_scale', 0.5)
            samples_share = samples_share * multilabel_scale
        else:
            # For single-label: standard sharing
            samples_share = np.array([0 if c < 0 else c for c in class_share]).reshape(int(self.args.num_IN_class), 1)

        # Set lambda parameter based on dataset and task type
        if self.args.dataset == 'CIFAR10':
            lamda = 0.6
        elif self.args.dataset == 'CIFAR100':
            lamda = 2
        elif self.args.dataset == 'TINYIMAGENET':
            lamda = 3
        elif self.args.dataset == 'RCV1' and self.is_multilabel:
            # Special handling for RCV1 multi-label dataset
            lamda = 1.5
        else:
            lamda = 1

        # Adjust lambda for multi-label tasks
        if self.is_multilabel:
            multilabel_lambda_scale = getattr(self.args, 'entropycb_multilabel_lambda_scale', 0.8)
            lamda = lamda * multilabel_lambda_scale

        print(f"EntropyCB optimization setup:")
        print(f"  Task type: {'Multi-label' if self.is_multilabel else 'Single-label'}")
        print(f"  Samples to select: {b}")
        print(f"  Current class counts: {counts}")
        print(f"  Target class threshold: {class_threshold}")
        print(f"  Class sharing needs: {class_share}")
        print(f"  Lambda parameter: {lamda}")
        print(f"  Samples share shape: {samples_share.shape}")

        for lam in [lamda]:
            # Define optimization variables
            z = cp.Variable((N, 1), boolean=True)
            constraints = [sum(z) == b]
            
            # Define objective: minimize entropy + class balance penalty
            if self.is_multilabel:
                # For multi-label: use L2 norm for smoother optimization
                cost = z.T @ U + lam * cp.norm(preds.T @ z - samples_share, 2)
            else:
                # For single-label: use L1 norm as in original
                cost = z.T @ U + lam * cp.norm1(preds.T @ z - samples_share)
            
            objective = cp.Minimize(cost)
            problem = cp.Problem(objective, constraints)
            
            # Solve optimization problem
            try:
                problem.solve(solver=cp.GUROBI, verbose=True, TimeLimit=1000)
                print('Optimal value with gurobi: ', problem.value)
                print(problem.status)
                print("A solution z is")
                print(z.value.T)
            except:
                print("GUROBI solver failed, trying ECOS_BB...")
                try:
                    problem.solve(solver=cp.ECOS_BB, verbose=True)
                    print('Optimal value with ECOS_BB: ', problem.value)
                    print(problem.status)
                except:
                    print("All solvers failed, using relaxed solution...")
                    # Fallback: use relaxed solution and round
                    z_relaxed = cp.Variable((N, 1))
                    constraints_relaxed = [sum(z_relaxed) == b, z_relaxed >= 0, z_relaxed <= 1]
                    cost_relaxed = z_relaxed.T @ U + lam * cp.norm(preds.T @ z_relaxed - samples_share, 2)
                    objective_relaxed = cp.Minimize(cost_relaxed)
                    problem_relaxed = cp.Problem(objective_relaxed, constraints_relaxed)
                    problem_relaxed.solve(verbose=True)
                    
                    # Round the relaxed solution
                    z_values = z_relaxed.value.flatten()
                    indices = np.argsort(-z_values)[:b]  # Select top b samples
                    z_solution = np.zeros(N)
                    z_solution[indices] = 1
                    z.value = z_solution.reshape(-1, 1)

            lb_flag = np.array(z.value.reshape(1, N)[0], dtype=bool)
            indices = np.where(lb_flag == 1)[0]

            # -----------------Stats of optimization---------------------------------
            n = self.args.n_query
            num_classes = int(self.args.num_IN_class)

            ENT_Loss.append(np.matmul(z.value.T, U))
            print('ENT LOSS= ', ENT_Loss)
            
            threshold = (2 * n / num_classes) + (self.cur_cycle + 1) * n / num_classes
            round_num = self.cur_cycle + 1
            
            # Get selected class distribution
            selected_classes = self.get_selected_class_counts(indices)
            labelled_classes = counts
            
            if self.is_multilabel:
                # For multi-label: compute frequency based on label counts
                freq = selected_classes + labelled_classes
                total_labels = (2 * n + round_num * n)
                L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * total_labels * (num_classes - 1)))
            else:
                # For single-label: use histogram
                freq = torch.histc(torch.FloatTensor(selected_classes), bins=num_classes) + torch.histc(torch.FloatTensor(labelled_classes), bins=num_classes)
                L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * (2 * n + round_num * n) * (num_classes - 1))).item()
            
            print('Lambda = ', lam)
            L1_DISTANCE.append(L1_distance)
            
            # Compute L1 loss based on task type
            if self.is_multilabel:
                L1_Loss_term = np.linalg.norm(np.matmul(preds.T, z.value) - samples_share, ord=2)
            else:
                L1_Loss_term = np.linalg.norm(np.matmul(preds.T, z.value) - samples_share, ord=1)
            L1_Loss.append(L1_Loss_term)

            print('L1 Loss = ')
            for i in L1_Loss:
                print('%.3f' % i)
            print('L1_distance = ')
            for j in L1_DISTANCE:
                print('%.3f' % j)
            print('ENT LOSS = ')
            for k in ENT_Loss:
                print('%.3f' % k)

        return indices

    def select(self, **kwargs):
        """
        Main selection method that uses convex optimization to balance entropy and class distribution.
        """
        selected_indices = self.rank_uncertainty()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        scores = list(np.ones(len(Q_index)))  # equally assign 1 (meaningless)

        print(f"EntropyCB selection completed:")
        print(f"  Selected {len(Q_index)} samples")
        print(f"  Task type: {'Multi-label' if self.is_multilabel else 'Single-label'}")

        return Q_index, scores

    def analyze_selection(self, selected_indices, preds):
        """
        Optional method to analyze the quality of selection.
        """
        if len(selected_indices) == 0:
            return
            
        selected_probs = preds[selected_indices]
        selected_entropies = self.compute_entropy(selected_probs)
        
        print(f"EntropyCB selection analysis:")
        print(f"  Average entropy of selected samples: {selected_entropies.mean():.4f}")
        print(f"  Entropy std of selected samples: {selected_entropies.std():.4f}")
        
        if self.is_multilabel:
            # Analyze label distribution
            label_coverage = (selected_probs > 0.5).sum(axis=0)
            print(f"  Label coverage: {label_coverage}")
            print(f"  Average labels per selected sample: {(selected_probs > 0.5).sum(axis=1).mean():.2f}")
        else:
            # Analyze class distribution
            predicted_classes = selected_probs.argmax(axis=1)
            class_counts = np.bincount(predicted_classes, minlength=self.args.num_IN_class)
            print(f"  Predicted class distribution: {class_counts}")
            print(f"  Class balance variance: {class_counts.var():.2f}")