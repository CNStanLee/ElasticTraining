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

def parse_bw_from_h5(h5_file='results/training_trace.h5'):
    """
    Load epoch and bitwidth percentage matrix from training_trace.h5.

    Returns:
      epochs: [N]
      bits: [B] (typically 0..8)
      bw_pct: [N, B]
    """
    if not os.path.exists(h5_file):
        print(f"H5 file {h5_file} not found")
        return None

    with h5py.File(h5_file, 'r') as f:
        if 'epochs' not in f or 'bw_pct' not in f:
            print("Required datasets ('epochs', 'bw_pct') not found in H5 file")
            return None

        epochs = np.array(f['epochs'][:]).astype(int)
        bw_pct = np.array(f['bw_pct'][:]).astype(float)

        if 'bits' in f:
            bits = np.array(f['bits'][:]).astype(int)
        else:
            bits = np.arange(bw_pct.shape[1], dtype=int)

    return epochs, bits, bw_pct

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

def plot_pareto_frontier(
    h5_file='results/training_trace.h5',
    output_file='results/pareto_frontier.png',
    show_all_points=True,
):
    """
    Plot Pareto frontier for (val_accuracy ↑, ebops ↓).
    """
    data = parse_results_from_h5(h5_file)
    if data is None:
        return

    epochs, accuracies, ebops_list = data

    valid_mask = np.isfinite(accuracies) & np.isfinite(ebops_list)
    if not np.any(valid_mask):
        print('No valid points found for Pareto frontier')
        return

    epochs = epochs[valid_mask]
    accuracies = accuracies[valid_mask]
    ebops_list = ebops_list[valid_mask]

    # Sort by EBOPs ascending; keep running best accuracy as Pareto frontier
    order = np.argsort(ebops_list)
    ebops_sorted = ebops_list[order]
    acc_sorted = accuracies[order]
    epochs_sorted = epochs[order]

    pareto_idx_sorted = []
    best_acc = -np.inf
    for idx, acc in enumerate(acc_sorted):
        if acc > best_acc:
            pareto_idx_sorted.append(idx)
            best_acc = acc

    pareto_ebops = ebops_sorted[pareto_idx_sorted]
    pareto_acc = acc_sorted[pareto_idx_sorted]
    pareto_epochs = epochs_sorted[pareto_idx_sorted]

    plt.figure(figsize=(10, 6))

    if show_all_points:
        plt.scatter(
            ebops_list,
            accuracies,
            s=18,
            alpha=0.35,
            color='tab:gray',
            label='All epochs',
        )

    plt.plot(
        pareto_ebops,
        pareto_acc,
        color='tab:orange',
        marker='o',
        linewidth=2,
        markersize=5,
        label='Pareto frontier',
    )

    if len(pareto_ebops) > 0:
        plt.scatter(
            pareto_ebops,
            pareto_acc,
            s=35,
            color='tab:red',
            label='Pareto points',
        )

    # Annotate a few points (start, middle, end) for readability
    if len(pareto_epochs) >= 1:
        key_ids = [0, len(pareto_epochs) // 2, len(pareto_epochs) - 1]
        key_ids = sorted(set(key_ids))
        for i in key_ids:
            plt.annotate(
                f"e{int(pareto_epochs[i])}",
                (pareto_ebops[i], pareto_acc[i]),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
            )

    plt.xlabel('EBOPs (lower is better)')
    plt.ylabel('Validation Accuracy (higher is better)')
    plt.title('Pareto Frontier: Validation Accuracy vs EBOPs')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Pareto frontier plot saved to {output_file}")
    # plt.show()

def plot_bw_vs_epochs(
    h5_file='results/training_trace.h5',
    output_file='results/bw_vs_epochs.png',
    bits_to_plot=range(9),
):
    """
    Plot bitwidth percentage trajectories (e.g., bw 0..8) versus epochs.
    """
    data = parse_bw_from_h5(h5_file)
    if data is None:
        return

    epochs, bits, bw_pct = data

    plt.figure(figsize=(14, 6))

    bits_set = set(int(b) for b in bits)
    plotted_any = False

    for bit in bits_to_plot:
        bit = int(bit)
        if bit not in bits_set:
            continue

        bit_idx = int(np.where(bits == bit)[0][0])
        plt.plot(
            epochs,
            bw_pct[:, bit_idx],
            linewidth=1.5,
            label=f'bw_{bit}',
        )
        plotted_any = True

    if not plotted_any:
        print('No matching bitwidth channels found to plot')
        return

    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.title('Bitwidth Distribution (bw 0..8) vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Bitwidth plot saved to {output_file}")
    # plt.show()

if __name__ == '__main__':
    result = 'results/baseline/'
    h5_file = os.path.join(result, 'training_trace.h5')  # Update this path if needed
    output_acc_ebops = os.path.join(result, 'aaa_acc_ebops_plot.png')
    output_pareto = os.path.join(result, 'aaa_pareto_frontier.png')
    output_bw = os.path.join(result, 'aaa_bw_vs_epochs.png')
    plot_acc_ebops_vs_epochs(h5_file, output_file=output_acc_ebops)
    plot_pareto_frontier(h5_file, output_file=output_pareto)
    plot_bw_vs_epochs(h5_file, output_file=output_bw)

    # result = 'results/ram/'
    # h5_file = os.path.join(result, 'training_trace.h5')  # Update this path if needed
    # output_file = os.path.join(result, 'acc_ebops_plot.png')
    # plot_acc_ebops_vs_epochs(h5_file, output_file=output_file)