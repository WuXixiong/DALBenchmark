import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to extract test accuracy from a text file
def extract_test_accuracy(file_path):
    cycles = []
    test_accuracies = []
    query_datas = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Use regular expression to find lines with "Test Accuracy"
            match0 = re.search(r'(\d+\.\d+)\|(\d+\.\d+)\|(\d+\.\d+)', line)
            match = re.search(r'(\d+)\|([\d\.]+)\|(\d+)\|\{([\d\D]+)\}', line)
            if match:
                cycle = float(match.group(1))  # Cycle number
                test_accuracy = float(match.group(2))  # Test Accuracy
                query_data = float(match.group(3))  # Number of in-domain query data
                test_accuracies.append(test_accuracy)
                cycles.append(int(cycle))
                query_datas.append(int(query_data))

            if match0:
                cycle = float(match0.group(1))  # Cycle number
                test_accuracy = float(match0.group(2))  # Test Accuracy
                query_data = float(match0.group(3))  # Number of in-domain query data
                test_accuracies.append(test_accuracy)
                cycles.append(int(cycle))
                query_datas.append(int(query_data))
    
    return cycles, test_accuracies, query_datas

# Directory containing your log files
log_dir = 'logs/SVHN/200'

# Dictionary to store results for each method
methods = {
    'CONF': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'Random': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'Entropy': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'LearningLoss': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'TrainingDynamics': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'Margin': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'LfOSA': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'Coreset': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'AlphaMix': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'BADGE': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'CoresetCB': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'EntropyCB': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'NoiseStability': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'SAAL': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'MeanSTD': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
    'VarRatio': {'test_accuracies': [], 'cycles': [], 'query_datas': []},
}

# Iterate through all files in the specified directory
for filename in os.listdir(log_dir):
    file_path = os.path.join(log_dir, filename)
    
    # Check if it's a file (not a directory) and filename contains the method name
    if os.path.isfile(file_path):
        if 'CONF' in filename:
            method = 'CONF'
        elif 'Random' in filename:
            method = 'Random'
        elif 'EntropyCB' in filename:
            method = 'EntropyCB'
        elif 'Entropy' in filename:
            method = 'Entropy'
        elif 'LL' in filename:
            method = 'LearningLoss'
        elif 'TIDAL' in filename:
            method = 'TrainingDynamics'
        elif 'Margin' in filename:
            method = 'Margin'
        elif 'LFOSA' in filename:
            method = 'LfOSA'
        elif 'Coreset' in filename and 'CoresetCB' not in filename:
            method = 'Coreset'
        elif 'AlphaMixSampling' in filename:
            method = 'AlphaMix'
        elif 'BADGE' in filename:
            method = 'BADGE'
        elif 'SAAL' in filename:
            method = 'SAAL'
        elif 'MeanSTD' in filename:
            method = 'MeanSTD'
        elif 'VarRatio' in filename:
            method = 'VarRatio'
        else:
            continue
        
        # Extract test accuracies from the file
        cycles, test_accuracies, query_datas = extract_test_accuracy(file_path)
        
        # Append the results to the corresponding method
        methods[method]['test_accuracies'].append(test_accuracies)
        methods[method]['cycles'].append(cycles)
        methods[method]['query_datas'].append(query_datas)
# Number of cycles, assuming all methods have the same number of cycles
num_cycles = len(cycles)  # or use a fixed number if you know it in advance

# Create a dictionary to map methods to specific colors
method_colors = {
    'CONF': 'blue',
    'Random': 'green',
    'Entropy': 'red',
    'LearningLoss': 'orange',
    'TrainingDynamics': 'purple',
    'Margin': 'brown',
    'LfOSA': 'cyan',
    'Coreset': 'magenta',
    'AlphaMix': 'lime',
    'BADGE': 'olive',
    'EntropyCB': 'gold',
    'NoiseStability': 'indigo',
    'SAAL': 'darkblue',
    'MeanSTD': 'darkgreen',
    'VarRatio': 'darkorange'
}

# Create the plot
plt.figure(figsize=(12, 6))

# Dictionary to store AUC and final accuracy for each method
auc_results = {}
final_accuracy_results = {}

final_accuracys = {}
all_aubc = {}
# Plot for each method
for method, data in methods.items():
    if data['test_accuracies']:  # Ensure there's data for the method
        # Extract the number of cycles and the accuracies as a NumPy array
        num_cycles = len(data['test_accuracies'][0])
        accuracies_array = np.array(data['test_accuracies'])  # Shape: (num_runs, num_cycles)
        
        # Calculate the mean and standard deviation across the experiments for each cycle
        mean_accuracies = np.mean(accuracies_array, axis=0)
        std_accuracies = np.std(accuracies_array, axis=0)
        
        # Plot the mean accuracy as a line using the fixed color
        cycles = np.arange(1, num_cycles + 1)
        plt.plot(cycles, mean_accuracies, label=f'{method}', linewidth=2, color=method_colors.get(method, 'black'))
        
        # Fill between the mean ± std to indicate variance
        plt.fill_between(cycles, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies, 
                         color=method_colors.get(method, 'black'), alpha=0.1)
        
        # Calculate the AUC using the trapezoidal rule, with accuracy scaled to percentages
        auc = np.trapz(mean_accuracies / 100, dx=1)  # Divide accuracies by 100 to convert to fractions
        auc_normalized = auc / num_cycles  # Normalize by number of cycles
        auc_results[method] = auc_normalized  # Store the normalized AUC for this method

        # calculate AUBC for each trial
        current_aubc = []
        for accuracy_array in accuracies_array:
            aubc = np.trapz(accuracy_array / 100, dx=1)
            aubc_normalized = aubc / num_cycles
            current_aubc.append(aubc_normalized)
        all_aubc[method] = current_aubc

        # Extract the final accuracies from each run
        final_accuracies = accuracies_array[:, -1]  # Get the last value from each run
        final_accuracy_mean = np.mean(final_accuracies)
        final_accuracy_std = np.std(final_accuracies)
        final_accuracy_results[method] = (final_accuracy_mean, final_accuracy_std)
        final_accuracys[method] = final_accuracies

# Add labels and title
plt.xlabel('Cycle')
plt.ylabel('Test Accuracy (%)')
# plt.legend(loc='lower right')  # Adjust the legend location if needed
# Adjust the legend to be outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside
# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rectangle to make room for the legend
plt.grid(True)

# Save the plot to a file
plt.savefig('plots/svhn_200.png')  # Saves the plot as a PNG file
plt.show()

# Optionally, print the AUC results in a sorted order for better readability
sorted_auc_results = sorted(auc_results.items(), key=lambda x: x[1], reverse=True)
print("\nAUBC Results:")
for method, auc in sorted_auc_results:
    print(f"{method}: {auc:.4f}")  # AUBC values should now be closer to 0-1 range

# Print final accuracy results sorted by mean final accuracy
sorted_final_accuracy = sorted(final_accuracy_results.items(), key=lambda x: x[1][0], reverse=True)
print("\nFinal Accuracy Results:")
for method, (mean_acc, std_acc) in sorted_final_accuracy:
    print(f"{method}: {mean_acc:.2f}%")
