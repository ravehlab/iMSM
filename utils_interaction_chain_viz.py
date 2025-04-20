import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(data, labels=[], label_sizes=[], cbar_label=r'$Log_{10} (Pr[i \to j])$', 
                      title="", suptitle="", first_and_last_factor=1, figsize=(10, 8), dpi=300):
    # Expand first and last rows (keeping your existing logic)
    first_row_repeated = np.tile(data[0, :], (first_and_last_factor-1, 1))
    data = np.append(first_row_repeated, data, axis=0)
    first_col_repeated = np.tile(data[:, 0], (first_and_last_factor-1, 1))
    data = np.append(first_col_repeated.T, data, axis=1)
    last_row_repeated = np.tile(data[-1, :], (first_and_last_factor-1, 1))
    data = np.append(data, last_row_repeated, axis=0)
    last_col_repeated = np.tile(data[:, -1], (first_and_last_factor-1, 1))
    data = np.append(data, last_col_repeated.T, axis=1)
    
    # Create figure with customizable size
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create the heatmap with improved colormap
    im = ax.imshow(data, cmap='jet', interpolation='nearest', vmin=-5, vmax=0)
    
    # Add colorbar with better formatting
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=10, fontweight='bold')
    
    # Calculate tick positions (center of each group)
    tick_positions = []
    current_pos = 0
    
    for size in label_sizes:
        tick_positions.append(current_pos + size / 2 - 0.5)
        current_pos += size
    
    # Set ticks and labels with better formatting
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    # Add gridlines at label boundaries
    if len(label_sizes) > 1:
        grid_lines = [sum(label_sizes[:i]) - 0.5 for i in range(1, len(label_sizes))]
        for pos in grid_lines:
            ax.axhline(y=pos, color='white', linestyle='-', linewidth=1.0)
            ax.axvline(x=pos, color='white', linestyle='-', linewidth=1.0)
    
    # Set title and axis labels with better formatting
    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    if title:
        ax.set_title(title, fontsize=12, pad=10)
        
    ax.set_xlabel("To", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("From", fontsize=11, fontweight='bold', labelpad=10)
    
    # Add a light grid to help with visual alignment
    ax.tick_params(axis='both', which='major', length=0)
    
    # Tight layout to optimize spacing
    plt.tight_layout()
    
    # Return the figure for potential further modifications
    return fig, ax

def labels_and_sizes(states, first_and_last_factor=5):

    labels = []
    label_sizes = []
    prev_label = ""
    cur_size = first_and_last_factor
    for i in range(len(states)):
        cur_label = states[i]    
        if cur_label == "nuc":
            labels.append(cur_label)
            prev_label = cur_label
            continue
        if cur_label == "cyt":
            labels.append(cur_label)
            label_sizes.append(cur_size)
            label_sizes.append(first_and_last_factor)
            prev_label = cur_label
            break
        
        cur_label = cur_label[:-3]
        
        if cur_label == prev_label:
            cur_size += 1
            continue
        label_sizes.append(cur_size)
        cur_size = 1
        labels.append(cur_label)
        prev_label = cur_label
    
    return labels, label_sizes

def labels_and_sizes_same(first_and_last_factor=5):
    labels = ["Nuc", "Nup2", "Nup60", "Nup2", "Nup60", "Nup1", "Nup145", 
                "Nup49", "Nsp1", "Nup49", "Nsp1", "Nup57", "Nup145", 
                "Nup57", "Nup57", "Nup57", "Nsp1", "Nup49", "Nsp1", 
                "Nup49", "Nup100", "Nup159", "Nup100", "Nup159", 
                "Nsp1", "Nsp1", "Nup116", "Nup116", "Cyt"]
    label_sizes = [first_and_last_factor] + [8] * (len(labels) - 2) + [first_and_last_factor]
    return labels, label_sizes