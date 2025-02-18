# logger.py
import datetime
import os

def initialize_log(args, trial):
    logs = []
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(["This experiment time is:"])
    logs.append([current_time])
    logs.append([])
    logs.append(["Experiment settings:"])
    if args.method == 'Uncertainty':
        logs.append(["AL method:" + args.uncertainty])
    else:
        logs.append(["AL method:" + args.method])
    logs.append(["Dataset:" + args.dataset])
    logs.append(["Model random seed:" + str(args.seed + trial)])
    logs.append(["Backbone:" + args.model])
    logs.append(["Optimizer:" + args.optimizer])
    logs.append(["Learning rate:" + str(args.lr)])
    logs.append(["Weight decay:" + str(args.weight_decay)])
    logs.append(["Gamma value for StepLR:" + str(args.gamma)])
    logs.append(["Step size for StepLR:" + str(args.step_size)])
    logs.append(["Number of total epochs each cycle:" + str(args.epochs)])
    logs.append(["Initial training set size:" + str(args.n_initial)])
    logs.append([])
    logs.append(['Cycle | ', 'Test Accuracy | ', 'Number of in-domain query data | ', 'Queried Classes'])
    return logs

def log_cycle_info(logs, cycle, acc, in_cnt, class_counts):
    from collections import Counter
    my_dict = dict(Counter(class_counts))
    sorted_dict = dict(sorted(my_dict.items()))
    logs.append([cycle + 1, acc, in_cnt, sorted_dict])

def save_logs(logs, args, trial):
    file_name = f'logs/{args.dataset}/{args.n_query}/open_set/r{args.ood_rate}_t{trial}_{args.method}'

    if not args.openset and not args.imbalanceset:
        file_name = f'logs/{args.dataset}/{args.n_query}/close_balance_set_t{trial}_{args.method}'
    if args.imbalanceset:
        file_name = f'logs/{args.dataset}/{args.n_query}/imbalance_set/_t{trial}_{args.method}'

    if args.method == 'MQNet':
        file_name = f'{file_name}_{args.mqnet_mode}_v3_b64'

    if args.method == 'Uncertainty':
        file_name = f'{file_name}_{args.uncertainty}'

    # Ensure the directory exists before saving the file
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'w') as file:
        for entry in logs:
            if len(entry) == 0:
                continue
            if isinstance(entry[-1], dict):
                entry_main = entry[:-1]
                queried_classes = entry[-1]
                queried_classes_str = ', '.join([f'{k}: {v}' for k, v in queried_classes.items()])
                file.write(' | '.join(str(x) for x in entry_main) + f' | {queried_classes_str}\n')
            elif all(isinstance(sub, list) for sub in entry):
                for sub_entry in entry:
                    if all(isinstance(x, (int, float)) for x in sub_entry):
                        file.write(' | '.join(f'{x:.4f}' for x in sub_entry) + '\n')
                    else:
                        file.write(' | '.join(str(x) for x in sub_entry) + '\n')
            else:
                file.write(' '.join(str(x) for x in entry) + '\n')
