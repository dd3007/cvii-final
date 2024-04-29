#  Code to generate training loss and evaluation metrics plots

# Plot 1 - Distillation Loss for each experiment
# Plot 2 - Finetuning Train and Evaluation Loss for each experiment on ChestXray14
# Plot 3 - Finetuning Evaluation mAUC for each experiment on ChestXray14
# Plot 4 - Finetuning Train and Evaluation Loss for each experiment on CheXpert
# Plot 5 - Finetuning Evaluation mAUC for each experiment on CheXpert

import matplotlib.pyplot as plt
import os

def plot_loss(data, dataset_name, output_dir):
    for key, finetune_log in data.items():
        train_loss = [line['train_loss'] for line in finetune_log]
        test_loss = [line['test_loss'] for line in finetune_log]
        epochs = [line['epoch'] for line in finetune_log]

        plt.plot(epochs, train_loss, label=f'{key} (train)')
        plt.plot(epochs, test_loss, label=f'{key} (eval)')

    plt.title(f'Finetuning Train and Evaluation Loss on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'finetune_loss_{dataset_name.lower()}.png'))
    plt.clf()

def plot_auc(data, dataset_name, output_dir):
    for key, finetune_log in data.items():
        eval_auc = [line['test_auc_avg'] for line in finetune_log]
        epochs = [line['epoch'] for line in finetune_log]

        plt.plot(epochs, eval_auc, label=key)

    plt.title(f'Evaluation mAUC on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('mAUC')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'finetune_auc_{dataset_name.lower()}.png'))
    plt.clf()

def truncate_data(data):
    min_epochs = min(len(d) for d in data.values())
    return {k: v[:min_epochs] for k, v in data.items()}

base_dir = '/mnt/home/mpaez/cvii-final/med_mae/results'
output_dir = '/mnt/home/mpaez/cvii-final/med_mae/plots'  # Changed to reflect the base directory

# Define file paths for the datasets
finetune_chestxray14_files = {
    'Baseline': os.path.join(base_dir, 'finetune_baseline_small', 'finetuned_baseline_small_chestxray14_50epoch.txt'),
    'New': os.path.join(base_dir, 'finetune_new_small', 'finetuned_new_small_chestxray14_50epoch.txt'),
}
finetune_chexpert_files = {
    'Baseline': os.path.join(base_dir, 'finetune_baseline_small', 'finetuned_baseline_small_chexpert_100epoch.txt'),
    'New': os.path.join(base_dir, 'finetune_new_small', 'finetuned_new_small_chexpert_50epoch.txt'),
}

# Read and parse files
finetune_chestxray14_data = {}
for key, finetune_file in finetune_chestxray14_files.items():
    with open(finetune_file, 'r') as file:
        finetune_log = file.readlines()
        finetune_log = [eval(line) for line in finetune_log]
        finetune_chestxray14_data[key] = finetune_log

finetune_chexpert_data = {}
for key, finetune_file in finetune_chexpert_files.items():
    with open(finetune_file, 'r') as file:
        finetune_log = file.readlines()
        finetune_log = [eval(line) for line in finetune_log]
        finetune_chexpert_data[key] = finetune_log

# Ensure all datasets have the same number of epochs
finetune_chestxray14_data = truncate_data(finetune_chestxray14_data)
finetune_chexpert_data = truncate_data(finetune_chexpert_data)

# Plotting
plot_loss(finetune_chestxray14_data, 'ChestXray14', output_dir)
plot_auc(finetune_chestxray14_data, 'ChestXray14', output_dir)
plot_loss(finetune_chexpert_data, 'CheXpert', output_dir)
plot_auc(finetune_chexpert_data, 'CheXpert', output_dir)