import os
import sys
from pathlib import Path

_fig_dir: Path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_repo_root: Path = _fig_dir.parent if _fig_dir.name == "figure_scripts" else _fig_dir
for _p in [str(_fig_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.exists("data") and (_repo_root / "data").exists():
    os.chdir(_repo_root)

import numpy as np
import pickle
import importlib
from scipy.stats import multivariate_normal, chi2, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
# --
import iMSM.extensions.npc.npc_utils
importlib.reload(iMSM.extensions.npc.npc_utils)
from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates_np, get_sorted_anchor_coordinates, infinitesimal_generator, stationary_distribution, radius_a_to_kda
# --
import iMSM.extensions.npc.npc_graph_figure
importlib.reload(iMSM.extensions.npc.npc_graph_figure)
from iMSM.extensions.npc.npc_graph_figure import comparison_plot, master_plot, load_cluster_data, sample_from_cluster, PIE_COLORS_DICT, PIE_COLORS, PIE_COLORS_LABELS
# -- 
import iMSM.extensions.npc.npc
importlib.reload(iMSM.extensions.npc.npc)
from iMSM.extensions.npc.npc import get_top_bottom_states


def calc_non_spoke_cluster_makeup_layered(cluster):
    """
    Calculate the makeup of the non-spoke cluster. (but yes layered, i.e. different layers of the same nup are seperated)
    Returns a dictionary with keys as microstate indices and values as probabilities.
    """
    # Initialize the makeup dictionary
    nups = ['Nup2_08', 'Nup2_09', 'Nup2_10', 'Nup2_11', 'Nup2_12', 'Nup2_13', 'Nup2_14', 'Nup2_15', 'Nup60_08', 'Nup60_09', 'Nup60_10', 'Nup60_11', 'Nup60_12', 'Nup60_13', 'Nup60_14', 'Nup60_15', 'Nup2_00', 'Nup2_01', 'Nup2_02', 'Nup2_03', 'Nup2_04', 'Nup2_05', 'Nup2_06', 'Nup2_07', 'Nup60_00', 'Nup60_01', 'Nup60_02', 'Nup60_03', 'Nup60_04', 'Nup60_05', 'Nup60_06', 'Nup60_07', 'Nup1_00', 'Nup1_01', 'Nup1_02', 'Nup1_03', 'Nup1_04', 'Nup1_05', 'Nup1_06', 'Nup1_07', 'Nup145_00', 'Nup145_01', 'Nup145_02', 'Nup145_03', 'Nup145_04', 'Nup145_05', 'Nup145_06', 'Nup145_07', 'Nup49_24', 'Nup49_25', 'Nup49_26', 'Nup49_27', 'Nup49_28', 'Nup49_29', 'Nup49_30', 'Nup49_31', 'Nsp1_40', 'Nsp1_41', 'Nsp1_42', 'Nsp1_43', 'Nsp1_44', 'Nsp1_45', 'Nsp1_46', 'Nsp1_47', 'Nup49_16', 'Nup49_17', 'Nup49_18', 'Nup49_19', 'Nup49_20', 'Nup49_21', 'Nup49_22', 'Nup49_23', 'Nsp1_32', 'Nsp1_33', 'Nsp1_34', 'Nsp1_35', 'Nsp1_36', 'Nsp1_37', 'Nsp1_38', 'Nsp1_39', 'Nup57_24', 'Nup57_25', 'Nup57_26', 'Nup57_27', 'Nup57_28', 'Nup57_29', 'Nup57_30', 'Nup57_31', 'Nup145_08', 'Nup145_09', 'Nup145_10', 'Nup145_11', 'Nup145_12', 'Nup145_13', 'Nup145_14', 'Nup145_15', 'Nup57_16', 'Nup57_17', 'Nup57_18', 'Nup57_19', 'Nup57_20', 'Nup57_21', 'Nup57_22', 'Nup57_23', 'Nup57_00', 'Nup57_01', 'Nup57_02', 'Nup57_03', 'Nup57_04', 'Nup57_05', 'Nup57_06', 'Nup57_07', 'Nup57_08', 'Nup57_09', 'Nup57_10', 'Nup57_11', 'Nup57_12', 'Nup57_13', 'Nup57_14', 'Nup57_15', 'Nsp1_16', 'Nsp1_17', 'Nsp1_18', 'Nsp1_19', 'Nsp1_20', 'Nsp1_21', 'Nsp1_22', 'Nsp1_23', 'Nup49_00', 'Nup49_01', 'Nup49_02', 'Nup49_03', 'Nup49_04', 'Nup49_05', 'Nup49_06', 'Nup49_07', 'Nsp1_24', 'Nsp1_25', 'Nsp1_26', 'Nsp1_27', 'Nsp1_28', 'Nsp1_29', 'Nsp1_30', 'Nsp1_31', 'Nup49_08', 'Nup49_09', 'Nup49_10', 'Nup49_11', 'Nup49_12', 'Nup49_13', 'Nup49_14', 'Nup49_15', 'Nup100_00', 'Nup100_01', 'Nup100_02', 'Nup100_03', 'Nup100_04', 'Nup100_05', 'Nup100_06', 'Nup100_07', 'Nup159_00', 'Nup159_01', 'Nup159_02', 'Nup159_03', 'Nup159_04', 'Nup159_05', 'Nup159_06', 'Nup159_07', 'Nup100_08', 'Nup100_09', 'Nup100_10', 'Nup100_11', 'Nup100_12', 'Nup100_13', 'Nup100_14', 'Nup100_15', 'Nup159_08', 'Nup159_09', 'Nup159_10', 'Nup159_11', 'Nup159_12', 'Nup159_13', 'Nup159_14', 'Nup159_15', 'Nsp1_00', 'Nsp1_01', 'Nsp1_02', 'Nsp1_03', 'Nsp1_04', 'Nsp1_05', 'Nsp1_06', 'Nsp1_07', 'Nsp1_08', 'Nsp1_09', 'Nsp1_10', 'Nsp1_11', 'Nsp1_12', 'Nsp1_13', 'Nsp1_14', 'Nsp1_15', 'Nup116_00', 'Nup116_01', 'Nup116_02', 'Nup116_03', 'Nup116_04', 'Nup116_05', 'Nup116_06', 'Nup116_07', 'Nup116_08', 'Nup116_09', 'Nup116_10', 'Nup116_11', 'Nup116_12', 'Nup116_13', 'Nup116_14', 'Nup116_15']

    # defaultdict starting 0
    makeup = {'nuc': 0.0, 'nuc_channel': 0.0, 'Nup2_1': 0.0, 'Nup60_1': 0.0, 'Nup2_0': 0.0, 'Nup60_0': 0.0, 'Nup1_0': 0.0, 'Nup145_0': 0.0, 'Nup49_3': 0.0, 'Nsp1_5': 0.0, 'Nup49_2': 0.0, 'Nsp1_4': 0.0, 'Nup57_3': 0.0, 'Nup145_1': 0.0, 'Nup57_2': 0.0, 'mid_channel': 0.0, 'Nup57_0': 0.0, 'Nup57_1': 0.0, 'Nsp1_2': 0.0, 'Nup49_0': 0.0, 'Nsp1_3': 0.0, 'Nup49_1': 0.0, 'Nup100_0': 0.0, 'Nup159_0': 0.0, 'Nup100_1': 0.0, 'Nup159_1': 0.0, 'Nsp1_0': 0.0, 'Nsp1_1': 0.0, 'Nup116_0': 0.0, 'Nup116_1': 0.0, 'cyt_channel' : 0.0, 'cyt': 0.0}

    # Iterate through each microstate index
    for i in range(458):
        if i == 0:
            makeup['nuc'] += cluster[i]
            continue
        if i == 457:
            makeup['cyt'] += cluster[i]
            continue
        if i in list(range(1,9)): # nuc channel
            makeup['nuc_channel'] += cluster[i]
            continue
        if i in list(range(449, 457)): # cyt channel
            makeup['cyt_channel'] += cluster[i]
            continue
        if i in list(range(225, 233)): # mid channel
            makeup['mid_channel'] += cluster[i]
            continue
        new_i = i
        if i < 225:
            new_i -= 9  # Adjust index for fgs (i.e. remove nuc and nuc_channel)
        else: 
            new_i -= 17 # Adjust index for fgs (i.e. remove nuc, nuc_channel, mid_channel)
        new_i //= 2 # adjust for nc
        nup = nups[new_i]
        split = nup.split("_")
        makeup[f"{split[0]}_{int(split[1])//8}"] += cluster[i]
        
    temp_makeup = np.array(list(makeup.values()))
    temp_makeup = temp_makeup / np.sum(temp_makeup)  # Normalize to sum to 1
    for key, value in zip(makeup.keys(), temp_makeup):
        makeup[key] = value
    
    return makeup

def cross_entropy(p, q, epsilon=1e-10):
    """
    Calculate cross entropy between two normalized histograms.
    
    Args:
        p: True distribution (numpy array)
        q: Predicted distribution (numpy array)
        epsilon: Small value to avoid log(0)
    
    Returns:
        Cross entropy value
    """
    # Add epsilon to avoid log(0)
    q_safe = np.clip(q, epsilon, 1.0)
    
    # Calculate cross entropy
    return -np.sum(p * np.log(q_safe))

def get_empirical_per_layer(clustered, clustering, tm):
    centroids = clustering.get_inverse_centers()
    counts = np.bincount(clustered.flatten(), minlength=tm.shape[0])
    empirical = counts / np.sum(counts)
    empirical = np.average(centroids, axis=0, weights=empirical)
    empirical_per_layer = calc_non_spoke_cluster_makeup_layered(empirical)
    return empirical_per_layer

def get_stationary_per_layer(tm, clustering):
    centroids = clustering.get_inverse_centers()
    stationary = stationary_distribution(tm)
    stationary = np.average(centroids, axis=0, weights=stationary)
    stationary_per_layer = calc_non_spoke_cluster_makeup_layered(stationary)
    return stationary_per_layer

def msm_md_js_divergence(tm_path, clustering_path, clustered_path):
    with open(tm_path, "rb") as f:
        tm = pickle.load(f)
    with open(clustering_path, "rb") as f:
        clustering = pickle.load(f)
    with open(clustered_path, "rb") as f:
        clustered = pickle.load(f)

    empirical_per_layer = get_empirical_per_layer(clustered, clustering, tm)
    stationary_per_layer = get_stationary_per_layer(tm, clustering)

    jsd = jensenshannon(
        np.array(list(empirical_per_layer.values())),
        np.array(list(stationary_per_layer.values()))
    ) ** 2

    return jsd

def msm_md_wasserstein(tm_path, clustering_path, clustered_path):
    """
    bin_centers: a 1D array/list of the x-axis positions of each state (in nm).
    Example: [0.5, 1.2, 5.0, 10.1, ...]
    """
    with open(tm_path, "rb") as f:
        tm = pickle.load(f)
    with open(clustering_path, "rb") as f:
        clustering = pickle.load(f)
    with open(clustered_path, "rb") as f:
        clustered = pickle.load(f)
        
    layer_zs = {'nuc': -40, 'nuc_channel': -20, 'Nup2_1': -17.22742004, 'Nup60_1': -17.22742004, 'Nup2_0': -15.30036926, 'Nup60_0': -15.30036926, 'Nup1_0': -7.48079376, 'Nup145_0': -7.2528282, 'Nup49_3': -3.39458046, 'Nsp1_5': -3.29364357, 'Nup49_2': -2.97223129, 'Nsp1_4': -2.90567799, 'Nup57_3': -2.06220741, 'Nup145_1': -1.72072506, 'Nup57_2': -1.60570507, 'mid_channel': 0.0, 'Nup57_0': 1.60570507, 'Nup57_1': 2.06220741, 'Nsp1_2': 2.90567799, 'Nup49_0': 2.97223148, 'Nsp1_3': 3.29364357, 'Nup49_1': 3.39458046, 'Nup100_0': 6.36628075, 'Nup159_0': 11.77414322, 'Nup100_1': 12.66063614, 'Nup159_1': 13.04617462, 'Nsp1_0': 15.12388916, 'Nsp1_1': 17.27483673, 'Nup116_0': 17.32117004, 'Nup116_1': 23.47518616, 'cyt_channel' : 20, 'cyt': 40}


    empirical_per_layer = get_empirical_per_layer(clustered, clustering, tm)
    stationary_per_layer = get_stationary_per_layer(tm, clustering)

    p_empirical = np.array(list(empirical_per_layer.values()))
    p_stationary = np.array(list(stationary_per_layer.values()))
    bin_centers = np.array(list(layer_zs.values()))

    # Calculate 1D Wasserstein distance using the specific bin locations
    wd = wasserstein_distance(
        u_values=bin_centers, 
        v_values=bin_centers, 
        u_weights=p_empirical, 
        v_weights=p_stationary
    )

    return wd

LAYER_BEAD_AMOUNTS = {
    'nuc': 0, 'nuc_channel': 0, 'Nup2_1': 160, 'Nup60_1': 96, 'Nup2_0': 160, 'Nup60_0': 96,
    'Nup1_0': 352, 'Nup145_0': 104, 'Nup49_3': 112, 'Nsp1_5': 256, 'Nup49_2': 112, 'Nsp1_4': 256,
    'Nup57_3': 120, 'Nup145_1': 104, 'Nup57_2': 120, 'mid_channel': 0, 'Nup57_0': 120, 'Nup57_1': 120,
    'Nsp1_2': 256, 'Nup49_0': 112, 'Nsp1_3': 256, 'Nup49_1': 112, 'Nup100_0': 320, 'Nup159_0': 272,
    'Nup100_1': 320, 'Nup159_1': 272, 'Nsp1_0': 256, 'Nsp1_1': 256, 'Nup116_0': 384, 'Nup116_1': 384,
    'cyt_channel': 0, 'cyt': 0
}

def visualize_stationary_distribution(base_tm_path, base_clustering_path, base_clustered_path,
                                      r, n, title, x_limits=(0, 0.05), use_energy=False, add_number_labels=True, add_bead_amounts=False, include_stationary=True, include_unbound_states=True, include_mid_channel=True, right_to_left=False, center_labels=False, nup_label_fontsize=28, nup_label_fontfamily="Roboto Condensed"):
    tm_path = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n))
    clustering_path = base_clustering_path.replace("#r#", str(r)).replace("#n#", str(n))
    clustered_path = base_clustered_path.replace("#r#", str(r)).replace("#n#", str(n))

    with open(tm_path, "rb") as f:
        tm = pickle.load(f)
    with open(clustering_path, "rb") as f:
        clustering = pickle.load(f)
    with open(clustered_path, "rb") as f:
        clustered = pickle.load(f)

    empirical_per_layer = get_empirical_per_layer(clustered, clustering, tm)
    stationary_per_layer = get_stationary_per_layer(tm, clustering)
    if not include_unbound_states:
        for key in ['nuc', 'cyt', 'nuc_channel', 'cyt_channel', 'mid_channel']:
            if key in empirical_per_layer:
                del empirical_per_layer[key]
            if key in stationary_per_layer:
                del stationary_per_layer[key]
        empirical_total = sum(empirical_per_layer.values())
        stationary_total = sum(stationary_per_layer.values())
        for key in empirical_per_layer.keys():
            empirical_per_layer[key] /= empirical_total
            stationary_per_layer[key] /= stationary_total
    if not include_mid_channel:
        if 'mid_channel' in empirical_per_layer:
            del empirical_per_layer['mid_channel']
        if 'mid_channel' in stationary_per_layer:
            del stationary_per_layer['mid_channel']
        empirical_total = sum(empirical_per_layer.values())
        stationary_total = sum(stationary_per_layer.values())
        for key in empirical_per_layer.keys():
            empirical_per_layer[key] /= empirical_total
            stationary_per_layer[key] /= stationary_total
    
    colors, labels = [], []
    for key in (stationary_per_layer.keys() if include_stationary else empirical_per_layer.keys()):
        split = key.split("_")
        nup = split[0] if split[0] not in ["nuc", "cyt", "mid"] else key
        color_key = key
        if key in ["Nsp1_0", "Nsp1_1"]:
            color_key = "Nsp1_cyt"
        elif key in ["Nsp1_2", "Nsp1_3", "Nsp1_4", "Nsp1_5"]:
            color_key = "Nsp1_inner"
        if color_key in PIE_COLORS_DICT:
            colors.append(PIE_COLORS_DICT[color_key])
        else:
            colors.append(PIE_COLORS_DICT[nup])
        labels.append(nup)

    stationary_values = list(stationary_per_layer.values())
    empirical_values = list(empirical_per_layer.values())
    y_positions = np.arange(len(labels))
    bar_thickness = 0.25 if (add_bead_amounts and include_stationary) else 0.35
    
    # reverse everything
    stationary_values = stationary_values[::-1]
    empirical_values = empirical_values[::-1]
    colors = colors[::-1]
    labels = labels[::-1]
    labels = [label.replace("_", " ") for label in labels]
    
    # 1. Define custom replacements
    label_map = {
        "nuc": "Nucleus",
        "nuc channel": "Nuc (partial)",
        "cyt": "Cytoplasm",
        "cyt channel": "Cyt (partial)",
        "mid channel": "Mid (partial)"
    }

    # 2. Apply the custom map, falling back to default formatting if no match is found
    labels = [label_map.get(label, label.replace("_", " ")) for label in labels]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 18))
    ax.set_xlim(*x_limits)
    
    # --- NEW RIGHT-TO-LEFT LOGIC ---
    if right_to_left:
        ax.invert_xaxis()
        ax.yaxis.tick_right()  # Move the tick labels to the right (where x=0 is)
        ax.yaxis.set_label_position("right") # Move the axis label to the right
    # -------------------------------
    
    ax.set_yticks(y_positions)
    
    # Logic for centrally aligning labels and preventing axis overlap
    if center_labels:
        h_align = 'center'
        y_pad = 75
    else:
        h_align = 'left' if right_to_left else 'right'
        y_pad = 4
        
    ax.set_yticklabels(labels, fontsize=nup_label_fontsize, fontfamily=nup_label_fontfamily, ha=h_align)
    ax.tick_params(axis='y', pad=y_pad)
    ax.tick_params(axis='x', labelsize=24)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    if use_energy:
        stationary_values = -np.log(np.clip(stationary_values, 1e-10, 1.0))
        empirical_values = -np.log(np.clip(empirical_values, 1e-10, 1.0))
        
    if add_bead_amounts:
        beads_raw = np.array([LAYER_BEAD_AMOUNTS.get(k, 0) for k in empirical_per_layer.keys()], dtype=float)
        bead_sum = np.sum(beads_raw)
        n_beads_norm = beads_raw / bead_sum if bead_sum > 0 else beads_raw
        if not include_stationary:
            jsd = jensenshannon(
                np.array(list(empirical_per_layer.values())),
                n_beads_norm
            ) ** 2
        else:
            jsd = jensenshannon(
                np.array(list(empirical_per_layer.values())),
                np.array(list(stationary_per_layer.values()))
            ) ** 2
        n_beads_per_layer = n_beads_norm[::-1]
        if use_energy:
            n_beads_per_layer = -np.log(np.clip(n_beads_per_layer, 1e-10, 1.0))
    else:
        jsd = jensenshannon(
            np.array(list(empirical_per_layer.values())),
            np.array(list(stationary_per_layer.values()))
        ) ** 2

    legend_handles = []
    legend_titles = []

    if include_stationary:
        stat_y = y_positions - bar_thickness / 2 if not add_bead_amounts else y_positions
        stationary_bars = ax.barh(
            stat_y,
            np.minimum(stationary_values, x_limits[1]),
            bar_thickness,
            label='Stationary',
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        legend_handles.append(stationary_bars[0])
        legend_titles.append('Stationary')

    emp_y = (
        y_positions + bar_thickness / 2
        if not add_bead_amounts
        else (y_positions - bar_thickness if include_stationary else y_positions + bar_thickness / 2)
    )
    empirical_bars = ax.barh(
        emp_y,
        np.minimum(empirical_values, x_limits[1]),
        bar_thickness,
        label='Empirical',
        color=colors,
        alpha=0.5,
        edgecolor='black',
        linewidth=1.5,
        hatch='///'
    )
    legend_handles.append(empirical_bars[0])
    legend_titles.append('Empirical')

    if add_bead_amounts:
        bead_y = y_positions + bar_thickness if include_stationary else y_positions - bar_thickness / 2
        bead_bars = ax.barh(
            bead_y,
            np.minimum(n_beads_per_layer, x_limits[1]),
            bar_thickness,
            label='Bead Proportion',
            color=colors,
            alpha=0.5,
            edgecolor='black',
            linewidth=1.5,
            hatch='|'
        )
        legend_handles.append(bead_bars[0])
        legend_titles.append('Bead Proportion')

    ax.invert_yaxis()
    ax.set_ylabel('Nucleoporin layer by Z of anchor', fontsize=20)
    if use_energy:
        ax.set_xlabel('Free Energy (kT)', fontsize=24)
    else:
        ax.set_xlabel('Probability', fontsize=24)

    fig.suptitle(
        f'{title}\nJensen-Shannon Divergence: {jsd:.4f}',
        fontsize=18,
        y=0.97
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ax.legend(
        legend_handles,
        legend_titles,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.04),
        fontsize=22,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        shadow=True
    )

    def add_labels(axis, bars, values):
            x_max = x_limits[1]
            x_offset = (x_max - x_limits[0]) * 0.01
            ha_align = 'right' if right_to_left else 'left'
            
            for bar, val in zip(bars, values):
                if val > x_max:
                    continue
                y_pos = bar.get_y() + bar.get_height() / 2
                xpos = min(bar.get_width() + x_offset, x_max - x_offset)
                axis.text(xpos, y_pos, f'{val:.3f}', ha=ha_align, va='center', fontsize=10)

    def annotate_overflows(axis, bars, values):
            limit = x_limits[1]
            span = x_limits[1] - x_limits[0]
            text_dx = span * 0.04
            cap_dx = span * 0.005
            ha_align = 'right' if right_to_left else 'left'

            for bar, val in zip(bars, values):
                if val <= limit:
                    continue
                y_pos = bar.get_y() + bar.get_height() / 2
                axis.plot(
                    [limit - cap_dx, limit + cap_dx],
                    [y_pos, y_pos],
                    color='gray',
                    linewidth=1.2,
                    linestyle='-',
                    clip_on=False
                )
                axis.annotate(
                    f'{val:.3f}',
                    xy=(limit, y_pos),
                    xytext=(limit + text_dx, y_pos),
                    ha=ha_align,
                    va='center',
                    fontsize=9,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle='<|-', lw=1, color='gray')
                )
                axis.plot([limit + 0.0003, limit + 0.0003], [y_pos - 0.5, y_pos + 0.5], linestyle='--', color='gray', clip_on=False)

    if add_number_labels:
        if include_stationary:
            add_labels(ax, stationary_bars, stationary_values)
        add_labels(ax, empirical_bars, empirical_values)
        if add_bead_amounts:
            add_labels(ax, bead_bars, n_beads_per_layer)
    if include_stationary:
        annotate_overflows(ax, stationary_bars, stationary_values)
    annotate_overflows(ax, empirical_bars, empirical_values)
    if add_bead_amounts:
        annotate_overflows(ax, bead_bars, n_beads_per_layer)

    plt.show()
    return fig
    
def get_variant_distributions(n, r, base_tm_path, base_clustering_path, base_clustered_path, 
                              include_unbound_states=True, include_mid_channel=True):
    """Fetches and formats empirical and stationary distributions for a specific variant."""
    tm_path = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n))
    clustering_path = base_clustering_path.replace("#r#", str(r)).replace("#n#", str(n))
    clustered_path = base_clustered_path.replace("#r#", str(r)).replace("#n#", str(n))

    with open(tm_path, "rb") as f:
        tm = pickle.load(f)
    with open(clustering_path, "rb") as f:
        clustering = pickle.load(f)
    with open(clustered_path, "rb") as f:
        clustered = pickle.load(f)

    # Note: These helper functions must be defined in your environment
    empirical_per_layer = get_empirical_per_layer(clustered, clustering, tm)
    stationary_per_layer = get_stationary_per_layer(tm, clustering)

    # Filter states if requested
    keys_to_remove = []
    if not include_unbound_states:
        keys_to_remove.extend(['nuc', 'cyt', 'nuc_channel', 'cyt_channel'])
    if not include_mid_channel:
        keys_to_remove.append('mid_channel')
        
    for key in keys_to_remove:
        if key in empirical_per_layer:
            del empirical_per_layer[key]
            del stationary_per_layer[key]

    # Renormalize
    emp_total = sum(empirical_per_layer.values())
    stat_total = sum(stationary_per_layer.values())
    for key in empirical_per_layer.keys():
        empirical_per_layer[key] /= emp_total
        stationary_per_layer[key] /= stat_total

    labels = list(stationary_per_layer.keys())[::-1]
    stat_vals = list(stationary_per_layer.values())[::-1]
    emp_vals = list(empirical_per_layer.values())[::-1]
    
    jsd = jensenshannon(emp_vals, stat_vals) ** 2
    return np.array(emp_vals), np.array(stat_vals), labels, jsd

def plot_free_energy_grid(n_sites_list, r_list, paths, include_mid_channel=False, title='Free Energy Distributions Across Nup Layers'):
    """
    Creates the grid of free energy plots.
    
    Args:
        paths (dict): Dictionary containing 'tm', 'clustering', and 'clustered' base paths.
        radius_to_kda_func (callable): Function to convert radius 'r' to kDa.
    """
    fig, axes = plt.subplots(nrows=len(n_sites_list), ncols=len(r_list), 
                             figsize=(12, 12), sharex=True, sharey=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    kdas_list = [radius_a_to_kda(r) for r in r_list]

    for i, n_sites in enumerate(n_sites_list):
        for j, r in enumerate(r_list):
            ax = axes[i, j]
            
            # Fetch data
            emp_vals, stat_vals, labels, jsd = get_variant_distributions(
                n_sites, r, paths['tm'], paths['clustering'], paths['clustered'], 
                include_mid_channel=include_mid_channel
            )
            
            # Energy calculation (-log(p))
            stat_energy = -np.log(np.clip(stat_vals, 1e-10, 1.0))
            emp_energy = -np.log(np.clip(emp_vals, 1e-10, 1.0))
            
            # Convert to Delta Free Energy (subtract the minimum to make the lowest value 0)
            stat_energy -= np.min(stat_energy)
            emp_energy -= np.min(emp_energy)
            
            # Step profile coordinates
            y_edges = np.arange(len(labels) + 1) - 0.5
            y_steps = np.repeat(y_edges, 2)[1:-1]
            stat_steps = np.repeat(stat_energy, 2)
            emp_steps = np.repeat(emp_energy, 2)
            
            # Plotting
            ax.plot(emp_steps, y_steps, color='#ff7f0e', alpha=0.9, linewidth=2, linestyle='--',
                    zorder=3, label='MD' if (i==0 and j==0) else "")
            ax.plot(stat_steps, y_steps, color='#1f77b4', alpha=0.9, linewidth=2.5,
                    zorder=2, label='iMSM' if (i==0 and j==0) else "")
            
            ax.fill_betweenx(y_steps, stat_steps, emp_steps, color='red', alpha=0.25, zorder=1)
            
            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i == 0:
                ax.set_title(f'{kdas_list[j]:.1f} kDa', fontsize=25, pad=10)

            if j == 0:
                ax.set_ylabel(f'{n_sites} Sites', fontsize=30, labelpad=10)
            
            # Formatting ticks
            ax.tick_params(axis='y', left=False, labelsize=0)
            
            if i == len(n_sites_list) - 1:
                # Set xlabel on the middle-ish plot of the bottom row
                if j == len(r_list) // 2: 
                    ax.set_xlabel(r'$\Delta$ Free Energy (kT)', fontsize=28, labelpad=10)
                ax.tick_params(axis='x', labelsize=25)
            else:
                ax.tick_params(axis='x', bottom=False, labelsize=25)
                
            ax.set_xlim(0, 11)  # You may need to adjust this max limit now that the curve starts at 0
            ax.set_xticks([0, 3, 6, 9])

    # Global elements
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), 
               ncol=2, fontsize=25, frameon=False)
    fig.suptitle(title, fontsize=25, y=1.02)
    
    return fig, axes
    
def plot_js(js_divergences, kdas_list, n_sites_list):    
    fig, ax = plt.subplots(figsize=(11, 7))
    js_labels = np.empty_like(js_divergences, dtype=object)
    for i in range(js_divergences.shape[0]):
        for j in range(js_divergences.shape[1]):
            val = js_divergences[i, j]
            if val < 0.01:
                js_labels[i, j] = "<0.01"
            else:
                js_labels[i, j] = f"{val:.2f}"
                
    sns.heatmap(js_divergences, 
                annot=js_labels, 
                fmt='', 
                cmap='Reds',
                vmin = 0,
                vmax = 0.5,
                xticklabels=[f'{kda:.1f}' for kda in kdas_list],
                yticklabels=[f'{n} Sites' for n in n_sites_list],
                cbar_kws={'label': 'JS Divergence'},
                annot_kws={'fontsize': 24},
                ax=ax)

    ax.set_xlabel('Molecular Mass (kDa)', fontsize=35, labelpad=20)
    # ax.set_ylabel('Number of interaction sites', fontsize=26, labelpad=20)

    # Set tick label fontsize
    ax.tick_params(axis='both', labelsize=27)

    # Set colorbar label fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=27)
    cbar.set_label('JS Divergence', fontsize=30, labelpad=20)

    plt.tight_layout()
    # plt.savefig('js_divergence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig
    
def plot_wd(wassersteins, kdas_list, n_sites_list):
    fig, ax = plt.subplots(figsize=(11, 7))
    wd_labels = np.empty_like(wassersteins, dtype=object)
    for i in range(wassersteins.shape[0]):
        for j in range(wassersteins.shape[1]):
            val = wassersteins[i, j]
            wd_labels[i, j] = f"{val:.1f}"
            
    sns.heatmap(wassersteins, 
                annot=wd_labels, 
                fmt='', 
                cmap='Reds',
                # vmin = 0,
                # vmax = 0.5,
                xticklabels=[f'{kda:.1f}' for kda in kdas_list],
                yticklabels=[f'{n} Sites' for n in n_sites_list],
                cbar_kws={'label': 'Wasserstein Distance'},
                annot_kws={'fontsize': 24},
                ax=ax)

    ax.set_xlabel('Molecular Mass (kDa)', fontsize=35, labelpad=20)
    # ax.set_ylabel('Number of interaction sites', fontsize=26, labelpad=20)

    # Set tick label fontsize
    ax.tick_params(axis='both', labelsize=27)

    # Set colorbar label fontsize
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=27)
    cbar.set_label('Wasserstein\nDistance (nm)', fontsize=30, labelpad=20)

    plt.tight_layout()
    # plt.savefig('js_divergence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig
    
def plot_js_wd():
    n_clusters = 320
    r = 10
    js_divergences = np.zeros(shape=(3, 5))
    wassersteins = np.zeros(shape=(3, 5))
    base_tm_path=f"data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle"
    base_clustering_path=f"data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle"
    base_clustered_path=f"data/ntr_variants/#n#_#r#_more/5_clustered_subsets/1.00fraction_simulations/0index/{n_clusters}clusters.pickle"
    n_sites_list = [2, 4, 6]
    r_list = [10, 14, 18, 22, 26]
    kdas_list = [radius_a_to_kda(r) for r in r_list]
    for i, n_sites in enumerate(n_sites_list):
        for j, r in enumerate(r_list):
            tm_path = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n_sites))
            clustering_path = base_clustering_path.replace("#r#", str(r)).replace("#n#", str(n_sites))
            clustered_path = base_clustered_path.replace("#r#", str(r)).replace("#n#", str(n_sites))
            jsd = msm_md_js_divergence(tm_path, clustering_path, clustered_path)
            js_divergences[i, j] = jsd
            wd = msm_md_wasserstein(tm_path, clustering_path, clustered_path)
            wassersteins[i, j] = wd
    js_fig = plot_js(js_divergences, kdas_list, n_sites_list)
    wd_fig = plot_wd(wassersteins, kdas_list, n_sites_list)
    return js_fig, wd_fig

def plot_initiator_nups(base_tm_path: str, base_clustering_path: str, normalize_by_mass: bool = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import os

    n_sites_list = [4, 6]
    r = 26
    n_sims = 10  # Number of successful transport trajectories to simulate per variant

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)

    nup_keys = ['Nup2', 'Nup60', 'Nup1', 'Nup145', 'Nup49', 'Nsp1_inner', 'Nup57', 'Nup100', 'Nup159', 'Nsp1_cyt', 'Nup116']
    display_labels = ['Nsp1 (cyt)' if k == 'Nsp1_cyt' else ('Nsp1 (inner)' if k == 'Nsp1_inner' else k) for k in nup_keys]

    nup_masses = {
        'Nup2': LAYER_BEAD_AMOUNTS['Nup2_0'] + LAYER_BEAD_AMOUNTS['Nup2_1'],
        'Nup60': LAYER_BEAD_AMOUNTS['Nup60_0'] + LAYER_BEAD_AMOUNTS['Nup60_1'],
        'Nup1': LAYER_BEAD_AMOUNTS['Nup1_0'],
        'Nup145': LAYER_BEAD_AMOUNTS['Nup145_0'] + LAYER_BEAD_AMOUNTS['Nup145_1'],
        'Nup49': sum(LAYER_BEAD_AMOUNTS[f'Nup49_{i}'] for i in range(4)),
        'Nsp1_inner': sum(LAYER_BEAD_AMOUNTS[f'Nsp1_{i}'] for i in range(2, 6)),
        'Nup57': sum(LAYER_BEAD_AMOUNTS[f'Nup57_{i}'] for i in range(4)),
        'Nup100': LAYER_BEAD_AMOUNTS['Nup100_0'] + LAYER_BEAD_AMOUNTS['Nup100_1'],
        'Nup159': LAYER_BEAD_AMOUNTS['Nup159_0'] + LAYER_BEAD_AMOUNTS['Nup159_1'],
        'Nsp1_cyt': LAYER_BEAD_AMOUNTS['Nsp1_0'] + LAYER_BEAD_AMOUNTS['Nsp1_1'],
        'Nup116': LAYER_BEAD_AMOUNTS['Nup116_0'] + LAYER_BEAD_AMOUNTS['Nup116_1'],
    }

    for col_idx, n_sites in enumerate(n_sites_list):
        tm_path = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n_sites))
        clustering_path = base_clustering_path.replace("#r#", str(r)).replace("#n#", str(n_sites))

        with open(tm_path, "rb") as f:
            P = pickle.load(f)
        with open(clustering_path, "rb") as f:
            clustering = pickle.load(f)

        centers = clustering.get_inverse_centers() # (320, 458)
        
        # Calculate Nup layered makeup for all clusters
        cluster_makeups = []
        for c in range(P.shape[0]):
            makeup = calc_non_spoke_cluster_makeup_layered(centers[c])
            cluster_makeups.append(makeup)

        # Find reservoirs
        nuc_comps = [m['nuc'] for m in cluster_makeups]
        cyt_comps = [m['cyt'] for m in cluster_makeups]
        nuc_res = np.argmax(nuc_comps)
        cyt_res = np.argmax(cyt_comps)

        # We also define the FG compositions of all clusters
        # Mapping layered states to FG Nups with separated Nsp1 cyt/inner
        fg_compositions = []
        for makeup in cluster_makeups:
            fg_comp = {nup: 0.0 for nup in nup_keys}
            # Sum layers for each Nup type
            fg_comp['Nup2'] = makeup.get('Nup2_0', 0.0) + makeup.get('Nup2_1', 0.0)
            fg_comp['Nup60'] = makeup.get('Nup60_0', 0.0) + makeup.get('Nup60_1', 0.0)
            fg_comp['Nup1'] = makeup.get('Nup1_0', 0.0)
            fg_comp['Nup145'] = makeup.get('Nup145_0', 0.0) + makeup.get('Nup145_1', 0.0)
            fg_comp['Nup49'] = (makeup.get('Nup49_0', 0.0) + makeup.get('Nup49_1', 0.0) +
                                makeup.get('Nup49_2', 0.0) + makeup.get('Nup49_3', 0.0))
            fg_comp['Nsp1_cyt'] = makeup.get('Nsp1_0', 0.0) + makeup.get('Nsp1_1', 0.0)
            fg_comp['Nsp1_inner'] = (makeup.get('Nsp1_2', 0.0) + makeup.get('Nsp1_3', 0.0) +
                                     makeup.get('Nsp1_4', 0.0) + makeup.get('Nsp1_5', 0.0))
            fg_comp['Nup57'] = (makeup.get('Nup57_0', 0.0) + makeup.get('Nup57_1', 0.0) +
                                makeup.get('Nup57_2', 0.0) + makeup.get('Nup57_3', 0.0))
            fg_comp['Nup100'] = makeup.get('Nup100_0', 0.0) + makeup.get('Nup100_1', 0.0)
            fg_comp['Nup159'] = makeup.get('Nup159_0', 0.0) + makeup.get('Nup159_1', 0.0)
            fg_comp['Nup116'] = makeup.get('Nup116_0', 0.0) + makeup.get('Nup116_1', 0.0)
            
            # Renormalize to sum to 1
            total = sum(fg_comp.values())
            if total > 0.0:
                for nup in fg_comp:
                    fg_comp[nup] /= total
            fg_compositions.append(fg_comp)

        # Define reservoir sets using 0.95 threshold
        nuc_states = {c for c, m in enumerate(cluster_makeups) if m['nuc'] >= 0.95}
        cyt_states = {c for c, m in enumerate(cluster_makeups) if m['cyt'] >= 0.95}

        # Simulation function
        def run_sim_set(start_states, target_states, initial_state):
            initiator_profiles = []
            while len(initiator_profiles) < n_sims:
                curr = initial_state
                path_first_interacting = None
                while True:
                    # Transition to next state
                    curr = np.random.choice(P.shape[0], p=P[curr, :])
                    if curr in target_states:
                        if path_first_interacting is not None:
                            initiator_profiles.append(fg_compositions[path_first_interacting])
                        break
                    elif curr in start_states:
                        path_first_interacting = None  # Reset path
                    else:
                        # Check if it is an interacting state (has FG component)
                        if path_first_interacting is None:
                            # if sum of FG components in the makeup of curr is significant
                            fg_sum = sum(cluster_makeups[curr][k] for k in cluster_makeups[curr] if k not in ['nuc', 'cyt', 'nuc_channel', 'cyt_channel', 'mid_channel'])
                            if fg_sum > 0.1:
                                path_first_interacting = curr
            # Average the profiles
            avg_profile = {nup: 0.0 for nup in nup_keys}
            for profile in initiator_profiles:
                for nup in nup_keys:
                    avg_profile[nup] += profile[nup]
            for nup in nup_keys:
                avg_profile[nup] /= len(initiator_profiles)

            if normalize_by_mass:
                norm_profile = {nup: avg_profile[nup] / nup_masses[nup] for nup in nup_keys}
                total_norm = sum(norm_profile.values())
                if total_norm > 0.0:
                    avg_profile = {nup: norm_profile[nup] / total_norm for nup in nup_keys}

            return avg_profile

        # Nuclear Entry (Nuc -> Cyt transport)
        nuc_to_cyt_profile = run_sim_set(nuc_states, cyt_states, nuc_res)
        # Cytoplasmic Entry (Cyt -> Nuc transport)
        cyt_to_nuc_profile = run_sim_set(cyt_states, nuc_states, cyt_res)

        # Plot Nuc -> Cyt in row 0 (Nuclear Entry)
        ax_n2c = axes[0, col_idx]
        vals_n2c = [nuc_to_cyt_profile[nup] for nup in nup_keys]
        colors_n2c = [PIE_COLORS_DICT.get(nup, '#7f7f7f') for nup in nup_keys]
        ax_n2c.bar(display_labels, vals_n2c, color=colors_n2c, edgecolor='black', linewidth=0.5)
        ax_n2c.set_title(f"{n_sites} Sites - Nuclear Entry", fontsize=25)
        ax_n2c.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=22.5)
        ax_n2c.tick_params(axis='y', labelsize=22.5)
        ax_n2c.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot Cyt -> Nuc in row 1 (Cytoplasmic Entry)
        ax_c2n = axes[1, col_idx]
        vals_c2n = [cyt_to_nuc_profile[nup] for nup in nup_keys]
        colors_c2n = [PIE_COLORS_DICT.get(nup, '#7f7f7f') for nup in nup_keys]
        ax_c2n.bar(display_labels, vals_c2n, color=colors_c2n, edgecolor='black', linewidth=0.5)
        ax_c2n.set_title(f"{n_sites} Sites - Cytoplasmic Entry", fontsize=25)
        ax_c2n.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=22.5)
        ax_c2n.tick_params(axis='y', labelsize=22.5)
        ax_c2n.grid(axis='y', linestyle='--', alpha=0.7)

    y_label = "Fraction (Mass normalized)" if normalize_by_mass else "Initiator Nup Fraction"
    axes[0, 0].set_ylabel(y_label, fontsize=26.25)
    axes[1, 0].set_ylabel(y_label, fontsize=26.25)

    suptitle_text = (
        "Comparison of Initiator Nup Composition Normalized by Mass (r = 26 Å)"
        if normalize_by_mass
        else "Comparison of Initiator Nup Composition (r = 26 Å)"
    )
    plt.suptitle(suptitle_text, fontsize=32, y=1.02)
    plt.tight_layout()
    return fig