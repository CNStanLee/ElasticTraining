import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def parse_results_from_h5(h5_file='results/training_trace.h5'):
    """
    Load epoch, accuracy, and EBOPs from training_trace.h5
    """
    if not os.path.exists(h5_file):
        print(f"H5 file {h5_file} not found")
        return None
    
    with h5py.File(h5_file, 'r') as f:
        epochs = np.array(f['epochs'][:]).astype(int)
        accuracies = np.array(f['val_accuracy'][:]).astype(float)
        ebops_list = np.array(f['ebops'][:]).astype(float)
    
    return epochs, accuracies, ebops_list

def plot_acc_ebops_vs_epochs(h5_file='results/training_trace.h5', output_file='results/acc_ebops_plot.png'):
    """
    Plot accuracy and EBOPs vs epochs on a dual-axis figure.
    """
    data = parse_results_from_h5(h5_file)
    if data is None:
        return
    
    epochs, accuracies, ebops_list = data
    
    # Calculate acc/ebops ratio
    acc_ebops_ratio = accuracies / ebops_list
    
    # Create figure with triple y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot accuracy on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    line1 = ax1.plot(epochs, accuracies, color=color, marker='o', markersize=4, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot EBOPs on first right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('EBOPs', color=color)
    line2 = ax2.plot(epochs, ebops_list, color=color, marker='s', markersize=4, label='EBOPs')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Plot acc/ebops ratio on second right y-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color = 'tab:green'
    ax3.set_ylabel('Acc/EBOPs', color=color)
    line3 = ax3.plot(epochs, acc_ebops_ratio, color=color, marker='^', markersize=4, label='Acc/EBOPs')
    ax3.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Accuracy, EBOPs, and Efficiency (Acc/EBOPs) vs Epochs')
    fig.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    # plt.show()

if __name__ == '__main__':
    result = 'results/baseline/'
    h5_file = os.path.join(result, 'training_trace.h5')  # Update this path if needed
    output_file = os.path.join(result, 'acc_ebops_plot.png')
    plot_acc_ebops_vs_epochs(h5_file, output_file=output_file)

    # result = 'results/ram/'
    # h5_file = os.path.join(result, 'training_trace.h5')  # Update this path if needed
    # output_file = os.path.join(result, 'acc_ebops_plot.png')
    # plot_acc_ebops_vs_epochs(h5_file, output_file=output_file)