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
    
    # 修改表头，添加选择时间列
    logs.append(['Cycle | ', 'Test Accuracy | ', 'Precision | ', 'Recall | ', 'F1-Score | ', 
                'Number of in-domain query data | ', 'Selection Time (s) | ', 'Queried Classes'])
    
    # 添加时间统计初始化
    logs.append([])  # 空行分隔
    logs.append(["=== Selection Time Statistics ==="])
    
    return logs

def log_cycle_info(logs, cycle, acc, prec, recall, f1, in_cnt, class_counts, select_time=None):
    """
    记录每个cycle的信息，包括选择时间
    
    Args:
        logs: 日志列表
        cycle: 当前cycle编号
        acc: 准确率
        prec: 精确率
        recall: 召回率
        f1: F1分数
        in_cnt: 域内查询数据数量
        class_counts: 类别分布
        select_time: ALmethod.select()执行时间（秒）
    """
    from collections import Counter
    my_dict = dict(Counter(class_counts))
    sorted_dict = dict(sorted(my_dict.items()))
    
    # 如果没有提供选择时间，设置为N/A
    time_str = f"{select_time:.4f}" if select_time is not None else "N/A"
    
    logs.append([cycle + 1, acc, prec, recall, f1, in_cnt, time_str, sorted_dict])

def finalize_timing_stats(logs, all_select_times):
    """
    在日志末尾添加时间统计摘要
    
    Args:
        logs: 日志列表
        all_select_times: 所有选择时间的列表
    """
    if not all_select_times:
        return
    
    logs.append([])
    logs.append(["=== ALmethod.select() Time Statistics Summary ==="])
    logs.append([f"Total selection calls: {len(all_select_times)}"])
    logs.append([f"Average time per call: {sum(all_select_times)/len(all_select_times):.4f} seconds"])
    logs.append([f"Minimum time: {min(all_select_times):.4f} seconds"])
    logs.append([f"Maximum time: {max(all_select_times):.4f} seconds"])
    logs.append([f"Total selection time: {sum(all_select_times):.4f} seconds"])
    logs.append([f"Standard deviation: {(sum([(t - sum(all_select_times)/len(all_select_times))**2 for t in all_select_times])/len(all_select_times))**0.5:.4f} seconds"])

def save_logs(logs, args, trial, all_select_times=None):
    """
    保存日志到文件
    
    Args:
        logs: 日志列表
        args: 参数
        trial: 试验编号
        all_select_times: 所有选择时间列表（可选）
    """
    # 如果提供了时间统计，添加到日志末尾
    if all_select_times:
        finalize_timing_stats(logs, all_select_times)
    
    # Base directory structure
    base_dir = f'logs/{args.dataset}/{args.n_query}'
         
    # Determine the dataset type part of the path
    if args.openset:
        dataset_type = f'open_set/r{args.ood_rate}'
    elif args.imb_factor:
        dataset_type = f'imbalance_set/r{args.imb_factor}'
    else:
        dataset_type = 'close_balance_set'
         
    # Construct the base filename
    file_name = f'{base_dir}/{dataset_type}_t{trial+1}_{args.method}'
         
    # Add method-specific suffixes
    if args.method == 'MQNet':
        file_name = f'{file_name}_{args.mqnet_mode}_v3_b64'
    elif args.method == 'Uncertainty':
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

def log_trial_timing_summary(logs, trial, trial_select_times):
    """
    记录单个trial的时间统计摘要
    
    Args:
        logs: 日志列表
        trial: 试验编号
        trial_select_times: 当前trial的选择时间列表
    """
    if not trial_select_times:
        return
        
    logs.append([])
    logs.append([f"=== Trial {trial+1} Selection Time Summary ==="])
    logs.append([f"Cycles in this trial: {len(trial_select_times)}"])
    logs.append([f"Average time per cycle: {sum(trial_select_times)/len(trial_select_times):.4f} seconds"])
    logs.append([f"Total time for this trial: {sum(trial_select_times):.4f} seconds"])
    logs.append([f"Min time: {min(trial_select_times):.4f} seconds"])
    logs.append([f"Max time: {max(trial_select_times):.4f} seconds"])