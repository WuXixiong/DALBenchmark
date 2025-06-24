# Python
import time
import random

# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Utils
from utils import *
from trainers import *

# Custom
from arguments import parser
from loadData import get_dataset, get_sub_train_dataset, get_sub_test_dataset
import nets
import methods as methods
from collections import Counter
from logger import initialize_log, log_cycle_info, save_logs, log_trial_timing_summary

def get_query_class_info(args, train_dst, Q_index):
    """
    Extract class information from query indices, supporting both single-label and multi-label.
    
    Args:
        args: Arguments containing dataset configuration
        train_dst: Training dataset
        Q_index: Query indices
        
    Returns:
        class_counts: Counter object with class distribution
    """
    is_multilabel = getattr(args, 'is_multilabel', False)
    
    if args.textset:
        if is_multilabel:
            # Multi-label: labels is a tensor/array with multiple labels
            Q_classes = []
            for idx in Q_index:
                labels = train_dst[idx]['labels']
                if isinstance(labels, torch.Tensor):
                    # If it's a multi-hot vector, get indices of positive labels
                    if len(labels.shape) > 0 and labels.shape[0] > 1:
                        active_labels = torch.nonzero(labels, as_tuple=True)[0].tolist()
                        Q_classes.extend(active_labels)
                    else:
                        # Single label in tensor format
                        Q_classes.append(labels.item())
                elif isinstance(labels, (list, np.ndarray)):
                    # If it's a list of active label indices
                    Q_classes.extend(labels)
                else:
                    # Single label
                    Q_classes.append(labels)
        else:
            # Single-label: labels is a single value
            Q_classes = [train_dst[idx]['labels'].item() if isinstance(train_dst[idx]['labels'], torch.Tensor) 
                        else train_dst[idx]['labels'] for idx in Q_index]
    else:
        # Image dataset
        if is_multilabel:
            Q_classes = []
            for idx in Q_index:
                labels = train_dst[idx][1]
                if isinstance(labels, torch.Tensor):
                    if len(labels.shape) > 0 and labels.shape[0] > 1:
                        active_labels = torch.nonzero(labels, as_tuple=True)[0].tolist()
                        Q_classes.extend(active_labels)
                    else:
                        Q_classes.append(labels.item())
                elif isinstance(labels, (list, np.ndarray)):
                    Q_classes.extend(labels)
                else:
                    Q_classes.append(labels)
        else:
            Q_classes = [train_dst[idx][1] for idx in Q_index]
    
    class_counts = Counter(Q_classes)
    return class_counts

def setup_criterion(args):
    """
    Setup appropriate loss functions based on task type.
    
    Args:
        args: Arguments containing dataset configuration
        
    Returns:
        Dictionary containing different loss functions
    """
    is_multilabel = getattr(args, 'is_multilabel', False)
    
    # Main criterion for training
    if is_multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Additional criterions for specific methods
    criterion_xent = torch.nn.CrossEntropyLoss()  # Always cross-entropy for some methods
    
    # Center loss configuration
    if args.textset:  # text dataset
        feat_dim = 768  # BERT feature dimension
    else:  # image dataset
        feat_dim = 512  # Typical image feature dimension
        
    criterion_cent = CenterLoss(
        num_classes=args.num_IN_class + 1, 
        feat_dim=feat_dim, 
        use_gpu=True
    )
    
    if args.textset:
        optimizer_centloss = torch.optim.AdamW(criterion_cent.parameters(), lr=0.005)
    else:
        optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
    
    return criterion, criterion_xent, criterion_cent, optimizer_centloss

# Main
if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    args = get_more_args(args)
    print("args: ", args)

    # Add global time statistics
    all_select_times = []  # Store selection times for all trials and cycles

    # Runs on Different Class-splits
    for trial in range(args.trial):
        print("=============================Trial: {}=============================".format(trial + 1))

        # Initialize time statistics for each trial
        trial_select_times = []  # Store selection times for the current trial

        # Set random seed
        random_seed = args.seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        train_dst, unlabeled_dst, test_dst = get_dataset(args, trial)
        if args.method == 'TIDAL':
            train_dst.init_tidal_params(len(args.target_list))
            unlabeled_dst.init_tidal_params(len(args.target_list))

        # Initialize a labeled dataset by randomly sampling K=1,000 points from the entire dataset.
        I_index, O_index, U_index, Q_index = [], [], [], []
        I_index, O_index, U_index = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=True)
        test_I_index = get_sub_test_dataset(args, test_dst)

        # DataLoaders
        sampler_labeled = SubsetRandomSampler(I_index)
        sampler_test = SubsetSequentialSampler(test_I_index)
        train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
        test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            ood_detection_index = I_index + O_index
            sampler_ood = SubsetRandomSampler(O_index)
            sampler_query = SubsetRandomSampler(ood_detection_index)
            query_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            ood_dataloader = DataLoader(train_dst, sampler=sampler_ood, batch_size=args.batch_size, num_workers=args.workers)
            sampler_unlabeled = SubsetRandomSampler(U_index)
            unlabeled_loader = DataLoader(train_dst, sampler=sampler_unlabeled, batch_size=args.batch_size, num_workers=args.workers)
        dataloaders = {'train': train_loader, 'test': test_loader}

        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            dataloaders = {'train': train_loader, 'query': query_loader, 'test': test_loader, 'ood': ood_dataloader, 'unlabeled': unlabeled_loader}

        # Initialize logs
        logs = initialize_log(args, trial)

        models = None

        for cycle in range(args.cycle):
            print("====================Cycle: {}====================".format(cycle + 1))
            # Model (re)initialization
            random_seed = args.seed + trial
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True

            print("| Training on model %s" % args.model)
            models = get_models(args, nets, args.model, models)
            torch.backends.cudnn.benchmark = False

            # Loss, criterion and scheduler (re)initialization - Modified for multi-label support
            criterion, optimizers, schedulers = get_optim_configurations(args, models)

            # Setup additional criterions with multi-label support
            criterion, criterion_xent, criterion_cent, optimizer_centloss = setup_criterion(args)

            # PAL wnet
            ood_num = (args.num_IN_class+1)*2
            wnet, optimizer_wnet = set_Wnet(args, ood_num)

            # Self-supervised Training (for CCAL and MQ-Net with CSI)
            if cycle == 0:
                models = self_sup_train(args, trial, models, optimizers, schedulers, train_dst, I_index, O_index, U_index)

            # EOAL
            cluster_centers, cluster_labels, cluster_indices = [], [], []
            if args.method == 'EOAL':
                cluster_centers, _, cluster_labels, cluster_indices = unknown_clustering(args, models['ood_detection'], models['model_bc'], dataloaders['ood'], args.target_list)

            # Training
            t = time.time()
            train_model(args, trial + 1, models, criterion, optimizers, schedulers, dataloaders, 
                      criterion_xent, criterion_cent, optimizer_centloss, 
                      O_index, cluster_centers, cluster_labels, cluster_indices)
            print("cycle: {}, elapsed time: {}".format(cycle, (time.time() - t)))

            # Test
            print('Trial {}/{} || Cycle {}/{} || Labeled IN size {}: '.format(
                    trial + 1, args.trial, cycle + 1, args.cycle, len(I_index)), flush=True)
            acc, prec, recall, f1  = evaluate_model(args, models, dataloaders)

            #### AL Query #### - Modified to add timing statistics
            print("==========Start Querying==========")
            selection_args = dict(I_index=I_index,
                                  O_index=O_index,
                                  selection_method=args.uncertainty,
                                  dataloaders=dataloaders,
                                  cur_cycle=cycle,
                                  cluster_centers=cluster_centers,
                                  cluster_labels=cluster_labels,
                                  cluster_indices=cluster_indices,
                                  wnet=wnet)
            ALmethod = methods.__dict__[args.method](args, models, unlabeled_dst, U_index, **selection_args)

            # Add timing statistics
            select_start_time = time.time()
            Q_index, Q_scores = ALmethod.select()
            select_end_time = time.time()
            select_duration = select_end_time - select_start_time

            # Record time
            trial_select_times.append(select_duration)
            all_select_times.append(select_duration)

            print(f"Trial {trial+1}, Cycle {cycle+1} - ALmethod.select() time: {select_duration:.4f}s")

            # Get query data class - Modified for multi-label support
            class_counts = get_query_class_info(args, train_dst, Q_index)

            # Print class distribution
            if getattr(args, 'is_multilabel', False):
                print(f"Query label distribution (multi-label): {dict(class_counts)}")
            else:
                print(f"Query class distribution: {dict(class_counts)}")

            # Update Indices
            I_index, O_index, U_index, in_cnt = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=False)
            print("# Labeled_in: {}, # Labeled_ood: {}, # Unlabeled: {}".format(
                len(set(I_index)), len(set(O_index)), len(set(U_index)))
            )

            # Meta-training MQNet
            if args.method == 'MQNet':
                models, optimizers, schedulers = init_mqnet(args, nets, models, optimizers, schedulers)
                unlabeled_loader = DataLoader(unlabeled_dst, sampler=SubsetRandomSampler(U_index), batch_size=args.test_batch_size, num_workers=args.workers)
                delta_loader = DataLoader(train_dst, sampler=SubsetRandomSampler(Q_index), batch_size=max(1, args.csi_batch_size), num_workers=args.workers)
                models = meta_train(args, models, optimizers, schedulers, criterion, dataloaders['train'], unlabeled_loader, delta_loader)

            # Update trainloader
            sampler_labeled = SubsetRandomSampler(I_index)
            dataloaders['train'] = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            if args.method in ['LFOSA', 'EOAL', 'PAL']:
                query_Q = I_index + O_index
                sampler_query = SubsetRandomSampler(query_Q)
                dataloaders['query'] = DataLoader(train_dst, sampler=sampler_query, batch_size=args.batch_size, num_workers=args.workers)
                ood_query = SubsetRandomSampler(O_index)
                dataloaders['ood'] = DataLoader(train_dst, sampler=ood_query, batch_size=args.batch_size, num_workers=args.workers)

            # Log cycle information - Modified to include selection time
            log_cycle_info(logs, cycle, acc, prec, recall, f1, in_cnt, class_counts, select_duration)

        # Record timing summary for each trial after it ends
        log_trial_timing_summary(logs, trial, trial_select_times)

        # Print time statistics for the current trial
        if trial_select_times:
            avg_time = sum(trial_select_times) / len(trial_select_times)
            total_time = sum(trial_select_times)
            print(f"Trial {trial+1} Summary:")
            print(f"  - Average ALmethod.select() time: {avg_time:.4f}s")
            print(f"  - Total ALmethod.select() time: {total_time:.4f}s")
            print(f"  - Min time: {min(trial_select_times):.4f}s")
            print(f"  - Max time: {max(trial_select_times):.4f}s")

        # Save logs after each trial (including time statistics)
        save_logs(logs, args, trial, all_select_times)

    # Print overall statistics after all trials
    if all_select_times:
        print(f"\n========== Overall ALmethod.select() Time Statistics ==========")
        print(f"Total selection calls: {len(all_select_times)}")
        print(f"Average time per call: {sum(all_select_times)/len(all_select_times):.4f}s")
        print(f"Total selection time: {sum(all_select_times):.4f}s")
        print(f"Min time: {min(all_select_times):.4f}s")
        print(f"Max time: {max(all_select_times):.4f}s")
