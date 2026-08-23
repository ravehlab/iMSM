import numpy as np
import sys
from typing import List, Optional, Union, Tuple
import pickle
from scipy.spatial import ConvexHull
# import pygpcca
from scipy.stats import multivariate_normal, chi2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
# from adjustText import adjust_text
from matplotlib.patches import Wedge, FancyBboxPatch
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import networkx as nx
import itertools as it
from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates, get_sorted_anchor_coordinates_np, infinitesimal_generator, stationary_distribution, radius_a_to_kda
import iMSM.extensions.npc.npc_embed_cluster
sys.modules['_013_PRC_clustering'] = iMSM.extensions.npc.npc_embed_cluster # fix for old file name
import os

############################
# CLUSTER ANALYSIS HELPERS #
############################

def calc_z_min(microstate_index, anchor_coordinates):
    cur_z = anchor_coordinates[microstate_index][2]
    for i in range(microstate_index - 1, -1, -1):
        if anchor_coordinates[i][2] < cur_z:
            return ((anchor_coordinates[i][2] + cur_z) / 2) - 5
        

def calc_z_max(microstate_index, anchor_coordinates):
    cur_z = anchor_coordinates[microstate_index][2]
    for i in range(microstate_index + 1, 215):
        if anchor_coordinates[i][2] > cur_z:
            return ((anchor_coordinates[i][2] + cur_z) / 2) + 5

def microstate_index_to_coordinate_edges(microstate_index: int, anchor_coordinates: np.array) -> np.array:
    """
    return (z_min, z_max, rmin, rmax, phi_min, phi_max)
    """
    if microstate_index < 0 or microstate_index >= 458:
        raise ValueError("Microstate index must be between 0 and 457 inclusive.")

    # nuc and cyt
    if microstate_index == 0:
        return np.array([-35, -25, 0, 20, 0, 2*np.pi])
    if microstate_index == 457:
        return np.array([25, 35, 0, 20, 0, 2*np.pi])
    if microstate_index in list(range(1, 9)): # nuc channel
        min_angle = (microstate_index - 1) * (np.pi / 4)
        max_angle = min_angle + (np.pi / 4)
        return np.array([-15, -5, 0, 20, min_angle, max_angle])
    if microstate_index in list(range(449, 457)): # cyt channel
        min_angle = (microstate_index - 449) * (np.pi / 4)
        max_angle = min_angle + (np.pi / 4)
        return np.array([5, 15, 0, 20, min_angle, max_angle])
    if microstate_index in list(range(225, 233)): # mid channel
        min_angle = (microstate_index - 225) * (np.pi / 4)
        max_angle = min_angle + (np.pi / 4)
        return np.array([-5, 5, 0, 20, min_angle, max_angle])
     
    # if microstate_index == 1:
    #     return np.array([-15, 0, 0, 20, 0, 2*np.pi])
    # if microstate_index == 218:
    #     return np.array([0, 15, 0, 20, 0, 2*np.pi])
    
    if microstate_index < 225:
        microstate_index -= 9  # Adjust index for fgs (i.e. remove nuc and nuc_channel)
    else: 
        microstate_index -= 17 # Adjust index for fgs (i.e. remove nuc, nuc_channel, mid_channel)
    microstate_index //= 2 # adjust for nc
    
    if microstate_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        # bottom fgs 
        z_min = -20
        z_max = calc_z_max(microstate_index, anchor_coordinates)
    elif microstate_index in [208, 209, 210, 211, 212, 213, 214, 215]:
        # top fgs 
        z_min = calc_z_min(microstate_index, anchor_coordinates)
        z_max = 25
    else: 
        # other fgs
        z_min = calc_z_min(microstate_index, anchor_coordinates)
        z_max = calc_z_max(microstate_index, anchor_coordinates)
    
    anchor_x = anchor_coordinates[microstate_index][0]
    anchor_y = anchor_coordinates[microstate_index][1]
    rmin = 0
    rmax = np.sqrt(anchor_x**2 + anchor_y**2)

    index_mod_8 = microstate_index % 8
    phi_min = (index_mod_8 * np.pi / 4)
    phi_max = phi_min + (np.pi / 4)
    
    return np.array([z_min, z_max, rmin, rmax, phi_min, phi_max])
    
    
def calc_coordinate_edges_dict(anchor_coordinates):
    """
    Calculate the coordinate edge dictionary for all microstates.
    Returns a dictionary where keys are microstate indices and values are coordinate edges.
    """
    coordinate_edges = {}
    for i in range(458):
        coordinate_edges[i] = microstate_index_to_coordinate_edges(i, anchor_coordinates)
    
    return coordinate_edges


def sample_from_cluster(cluster: np.array, n_samples: int, coordinate_edges, ignore_nuc_cyt=False) -> np.array:
    """
    cluster - makeup of the cluster, i.e. probabilities of each microstate (nuc/fgs/cyt) - size (218)
    n_samples - number of samples to draw from the cluster
    returns a sample of size n_samples from the cluster, but in spatial space
    """
    # Normalize the cluster to ensure it sums to 1
    cluster = cluster.astype(np.float64)  # Ensure float64 for precision
    cluster = cluster / np.sum(cluster)
    # Draw samples from the multinomial distribution
    sample = np.random.multinomial(n=n_samples, pvals=cluster) # size (218,)
    # Convert sample indices to spatial coordinates
    spatial_coordinates = np.zeros((n_samples, 3))  # Initialize spatial coordinates array
    idx = 0  # Running index to track position in spatial_coordinates array

    for microstate_index, count in enumerate(sample):
        if count > 0:
            edges = coordinate_edges[microstate_index]
            z_min, z_max, rmin, rmax, phi_min, phi_max = edges
            
            # Sample uniformly within the defined edges
            z_samples = np.random.uniform(z_min, z_max, count)
            phi_samples = np.random.uniform(phi_min, phi_max, count)
            # Sample r with uniform density (proportional to area)
            # Use inverse transform sampling: r = sqrt(u*(rmax^2 - rmin^2) + rmin^2)
            u = np.random.uniform(0, 1, count)
            r_samples = np.sqrt(u * (rmax**2 - rmin**2) + rmin**2)
            
            
            # Convert polar coordinates to Cartesian coordinates
            x_samples = r_samples * np.cos(phi_samples)
            y_samples = r_samples * np.sin(phi_samples)
            
            # Fill the spatial coordinates for this microstate
            spatial_coordinates[idx:idx+count] = np.column_stack((x_samples, y_samples, z_samples))
            
            # Update the index
            idx += count  
    return spatial_coordinates
    
def calc_smooth_convex_hull(scatter, smoothing_factor=0.1, num_points=100):
    """
    Calculate the convex hull of the scatter points and smooth it using spline interpolation.
    Returns the vertices of the smooth convex hull.
    
    Parameters:
    - scatter: array of points
    - smoothing_factor: controls smoothness (0 = interpolation, higher = more smoothing)
    - num_points: number of points in the final smooth curve
    """
    # Calculate the convex hull
    hull = ConvexHull(scatter)
    
    # Extract the vertices of the convex hull
    vertices = scatter[hull.vertices]
    
    # Close the hull by adding the first vertex at the end for periodicity
    closed_vertices = np.vstack([vertices, vertices[0]])
    
    # Parametric spline interpolation for smooth curve
    # s parameter controls smoothing: 0 = interpolation, higher = more smoothing
    tck, u = splprep([closed_vertices[:, 0], closed_vertices[:, 1]], 
                     s=smoothing_factor * len(vertices), 
                     per=True)  # per=True for periodic (closed) curve
    
    # Generate smooth curve with more points
    u_new = np.linspace(0, 1, num_points)
    smooth_x, smooth_y = splev(u_new, tck)
    
    # Combine into vertices array
    smoothed_vertices = np.column_stack([smooth_x, smooth_y])
    
    return smoothed_vertices
    
def calc_non_spoke_cluster_makeup(cluster):
    """
    Calculate the makeup of the non-spoke cluster.
    Returns a dictionary with keys as microstate indices and values as probabilities.
    """
    # Initialize the makeup dictionary
    nups = ['Nup2_08', 'Nup2_09', 'Nup2_10', 'Nup2_11', 'Nup2_12', 'Nup2_13', 'Nup2_14', 'Nup2_15', 'Nup60_08', 'Nup60_09', 'Nup60_10', 'Nup60_11', 'Nup60_12', 'Nup60_13', 'Nup60_14', 'Nup60_15', 'Nup2_00', 'Nup2_01', 'Nup2_02', 'Nup2_03', 'Nup2_04', 'Nup2_05', 'Nup2_06', 'Nup2_07', 'Nup60_00', 'Nup60_01', 'Nup60_02', 'Nup60_03', 'Nup60_04', 'Nup60_05', 'Nup60_06', 'Nup60_07', 'Nup1_00', 'Nup1_01', 'Nup1_02', 'Nup1_03', 'Nup1_04', 'Nup1_05', 'Nup1_06', 'Nup1_07', 'Nup145_00', 'Nup145_01', 'Nup145_02', 'Nup145_03', 'Nup145_04', 'Nup145_05', 'Nup145_06', 'Nup145_07', 'Nup49_24', 'Nup49_25', 'Nup49_26', 'Nup49_27', 'Nup49_28', 'Nup49_29', 'Nup49_30', 'Nup49_31', 'Nsp1_40', 'Nsp1_41', 'Nsp1_42', 'Nsp1_43', 'Nsp1_44', 'Nsp1_45', 'Nsp1_46', 'Nsp1_47', 'Nup49_16', 'Nup49_17', 'Nup49_18', 'Nup49_19', 'Nup49_20', 'Nup49_21', 'Nup49_22', 'Nup49_23', 'Nsp1_32', 'Nsp1_33', 'Nsp1_34', 'Nsp1_35', 'Nsp1_36', 'Nsp1_37', 'Nsp1_38', 'Nsp1_39', 'Nup57_24', 'Nup57_25', 'Nup57_26', 'Nup57_27', 'Nup57_28', 'Nup57_29', 'Nup57_30', 'Nup57_31', 'Nup145_08', 'Nup145_09', 'Nup145_10', 'Nup145_11', 'Nup145_12', 'Nup145_13', 'Nup145_14', 'Nup145_15', 'Nup57_16', 'Nup57_17', 'Nup57_18', 'Nup57_19', 'Nup57_20', 'Nup57_21', 'Nup57_22', 'Nup57_23', 'Nup57_00', 'Nup57_01', 'Nup57_02', 'Nup57_03', 'Nup57_04', 'Nup57_05', 'Nup57_06', 'Nup57_07', 'Nup57_08', 'Nup57_09', 'Nup57_10', 'Nup57_11', 'Nup57_12', 'Nup57_13', 'Nup57_14', 'Nup57_15', 'Nsp1_16', 'Nsp1_17', 'Nsp1_18', 'Nsp1_19', 'Nsp1_20', 'Nsp1_21', 'Nsp1_22', 'Nsp1_23', 'Nup49_00', 'Nup49_01', 'Nup49_02', 'Nup49_03', 'Nup49_04', 'Nup49_05', 'Nup49_06', 'Nup49_07', 'Nsp1_24', 'Nsp1_25', 'Nsp1_26', 'Nsp1_27', 'Nsp1_28', 'Nsp1_29', 'Nsp1_30', 'Nsp1_31', 'Nup49_08', 'Nup49_09', 'Nup49_10', 'Nup49_11', 'Nup49_12', 'Nup49_13', 'Nup49_14', 'Nup49_15', 'Nup100_00', 'Nup100_01', 'Nup100_02', 'Nup100_03', 'Nup100_04', 'Nup100_05', 'Nup100_06', 'Nup100_07', 'Nup159_00', 'Nup159_01', 'Nup159_02', 'Nup159_03', 'Nup159_04', 'Nup159_05', 'Nup159_06', 'Nup159_07', 'Nup100_08', 'Nup100_09', 'Nup100_10', 'Nup100_11', 'Nup100_12', 'Nup100_13', 'Nup100_14', 'Nup100_15', 'Nup159_08', 'Nup159_09', 'Nup159_10', 'Nup159_11', 'Nup159_12', 'Nup159_13', 'Nup159_14', 'Nup159_15', 'Nsp1_00', 'Nsp1_01', 'Nsp1_02', 'Nsp1_03', 'Nsp1_04', 'Nsp1_05', 'Nsp1_06', 'Nsp1_07', 'Nsp1_08', 'Nsp1_09', 'Nsp1_10', 'Nsp1_11', 'Nsp1_12', 'Nsp1_13', 'Nsp1_14', 'Nsp1_15', 'Nup116_00', 'Nup116_01', 'Nup116_02', 'Nup116_03', 'Nup116_04', 'Nup116_05', 'Nup116_06', 'Nup116_07', 'Nup116_08', 'Nup116_09', 'Nup116_10', 'Nup116_11', 'Nup116_12', 'Nup116_13', 'Nup116_14', 'Nup116_15']

    # defaultdict starting 0
    makeup = {'nuc': 0.0, 'nuc_channel': 0.0, 'mid_channel': 0.0, 'Nup2': 0.0, 'Nup60': 0.0, 'Nup1': 0.0, 'Nup145': 0.0, 'Nup49': 0.0, 'Nup57': 0.0, 'Nsp1_cyt': 0.0, 'Nsp1_inner': 0.0, 'Nsp1': 0.0, 'Nup100': 0.0, 'Nup159': 0.0, 'Nup116': 0.0, 'cyt_channel' : 0.0, 'cyt': 0.0}

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
        nup_name = split[0]
        if nup_name == "Nsp1":
            layer = int(split[1]) // 8
            if layer in [0, 1]:
                makeup['Nsp1_cyt'] += cluster[i]
            elif layer in [2, 3, 4, 5]:
                makeup['Nsp1_inner'] += cluster[i]
        else:
            makeup[nup_name] += cluster[i]
        
    makeup = np.array(list(makeup.values()))
    makeup = makeup / np.sum(makeup)  # Normalize to sum to 1
    
    return makeup

#################
# VISUALIZATION #
#################

def draw_curved_vertical_line(ax, p1, p2, rad=0.4, **kwargs):
    """
    Draws a curved vertical line on a given matplotlib axes.

    Parameters:
    - ax: The matplotlib axes to draw on.
    - p1: The starting point (x, y).
    - p2: The ending point (x, y).
    - rad: The radius of the curve. Positive values curve one way,
           negative values the other.
    - **kwargs: Additional keyword arguments to pass to FancyArrowPatch,
                e.g., color, lw.
    """
    # Set default style to a simple line if not provided by the user
    kwargs.setdefault('arrowstyle', '-')
    
    conn = patches.FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        **kwargs
    )
    ax.add_patch(conn)

def align_points_to_grid(points, grid_step=5):
    aligned_points = {}
    for point_id, (x, y) in points.items():
        aligned_x = round(x / grid_step) * grid_step
        aligned_y = round(y / grid_step) * grid_step
        aligned_points[point_id] = (aligned_x, aligned_y)
    return aligned_points

def stretch_points_to_frame(points, frame_min = -35, frame_max = 35):
    coords = list(points.values())
    if len(coords) == 1:
        # Center the single point in the new frame
        center = (frame_min + frame_max) / 2.0
        return {k: (center, center) for k in points}
    min_x = min(c[0] for c in coords)
    max_x = max(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    max_y = max(c[1] for c in coords)
    range_x = max_x - min_x
    range_y = max_y - min_y
    frame_range = frame_max - frame_min
    new_points = {}
    for point_id, (x, y) in points.items():
        # Handle cases where all points lie on a vertical or horizontal line
        if range_x == 0:
            new_x = (frame_min + frame_max) / 2.0
        else:
            # Scale and translate x coordinate
            new_x = frame_min + ((x - min_x) * frame_range / range_x)
        if range_y == 0:
            new_y = (frame_min + frame_max) / 2.0
        else:
            # Scale and translate y coordinate
            new_y = frame_min + ((y - min_y) * frame_range / range_y)
        new_points[point_id] = (new_x, new_y)
    return new_points

def draw_labeled_multigraph(G, pos, sizes, arrow_widths, holding_rates, ax=None):
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    # pos = nx.shell_layout(G)
    # print(f"Number of nodes in graph: {len(G.nodes())}")
    # print(f"Length of sizes array: {len(sizes)}")
    # print(f"Sizes array: {sizes}")
    if (len(G.nodes) != len(sizes)):
        print("Could not draw graph: number of nodes does not match length of sizes array.")
        return
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color="grey")
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax, font_color="blue", font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax, arrowsize=20, node_size=sizes, width=arrow_widths
    )

    # Separate self-edges from other edges
    edge_labels = {}
    for *edge, attrs in G.edges(keys=True, data=True):
        edge_tuple = tuple(edge)
        label = f"{attrs['weight']}"
        edge_labels[edge_tuple] = label

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        connectionstyle=connectionstyle,
        label_pos=0.5,
        font_color="black",
        font_weight="bold",
        bbox={"alpha": 0},
        node_size=sizes,
        ax=ax,
    )
    
    # annotate holding rates above the nodes
    for i, (node, (x, y)) in enumerate(pos.items()):
        holding_rate = holding_rates[i]
        ax.text(x, y + 2, f"{holding_rate}", fontsize=10, ha='center', va='bottom', color='Red', fontweight='bold')

def adjust_distances(l,n,m):
    l.sort()
    r=[l[0]-m]
    for i in range(1,len(l)):
        v=max(r[-1]+n,l[i]-m)
        if v>l[i]+m:return None
        r.append(v)
    return r

def visualize_coarse_grained(n_valid_macrostates, ax3, coarse_grained_P, course_grained_stationary_distribution, meta_mus):
    course_grained_Q = infinitesimal_generator(coarse_grained_P, dt=1) # rate at 1 / us
    G = nx.MultiDiGraph()
    holding_rates = [] # represents how frequently the system "decides" to leave state i per unit time.
    for i in range(n_valid_macrostates):
        for j in range(n_valid_macrostates):
            prob = round(coarse_grained_P[i, j], 3)
            if prob > 0.001:
                if i == j:
                    holding_rates.append(round(np.log10(-course_grained_Q[i, j]), 2))
                else:
                    G.add_edge(i+1, j+1, weight=round(np.log10(course_grained_Q[i, j]), 2))
    pos = {}
    for i in range(n_valid_macrostates):
        pos[i+1] = meta_mus[i]
    # pos = stretch_points_to_frame(pos, frame_min=-30, frame_max=30)
    # pos = align_points_to_grid(pos, grid_step=5)
    weights = [d['weight'] * 10 for u, v, d in G.edges(data=True)]
    draw_labeled_multigraph(G,
                            pos,
                            sizes=course_grained_stationary_distribution * 5000,
                            arrow_widths=1,
                            holding_rates=holding_rates,
                            ax=ax3)

def visualize_nup_annotations(ax, pie_colors=None, pie_colors_labels=None, add_scatters=False):
    anchor_coordinates = get_sorted_anchor_coordinates()
    labels = []
    zs = []
    texts = []
    
    for nup, coords in anchor_coordinates.items():
        if nup[-2:] in ['06',  '14', '22', '30', '38', '46']:
            z = coords[2] + (-0.1 if nup[3:-3] == '2' else 0.0)
            label = nup[:-3]
            zs.append(z)
            labels.append(label)
    
    zs[:25] = adjust_distances(zs[:25], n=1, m=5)
    left = False
    for i in range(len(zs)):
        x = -30 if left else -32
        ha = 'left' if left else 'right'
        if pie_colors is not None:
            texts.append(ax.text(x, zs[i], labels[i], fontsize=11, color=pie_colors[pie_colors_labels.index(labels[i])], ha=ha, va='center', fontweight='bold'))
        else:
            texts.append(ax.text(x, zs[i], labels[i], fontsize=11, ha=ha, va='center', fontweight='bold'))
        left = not left
    if pie_colors is not None:
        cyt_text = ax.text(-30, 30, "Cytoplasm", fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Cytoplasm')], fontfamily='Roboto Condensed')
        cyt_channel_text = ax.text(-30, 28, 'Unbound Channel (Cyt)', fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Unbound Channel (Cyt)')], fontfamily='Roboto Condensed')
        nuc_text = ax.text(-30, -30, "Nucleus", fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Nucleus')], fontfamily='Roboto Condensed')
        nuc_channel_text = ax.text(-30, -28, 'Unbound Channel (Nuc)', fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Unbound Channel (Nuc)')], fontfamily='Roboto Condensed')
    else:
        cyt_text = ax.text(-30, 30, "Cytoplasm", fontsize=11, fontweight='bold', fontfamily='Roboto Condensed')
        nuc_text = ax.text(-30, -30, "Nucleus", fontsize=11, fontweight='bold', fontfamily='Roboto Condensed')
    # texts, _ = adjust_text(texts, ax = ax)
    
    # add outlines
    for text in texts + [cyt_text, nuc_text, cyt_channel_text, nuc_channel_text]:
        text.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
            ])  
    
    if pie_colors is not None and add_scatters:
        for text in texts + [cyt_text, nuc_text, cyt_channel_text, nuc_channel_text]:
            x = text.get_position()[0]
            y = text.get_position()[1] + 0.3
            text = text.get_text()
            # add color legend
            i = pie_colors_labels.index(text)
            color = pie_colors[i]
            ax.scatter(-31, y, color=color, s=50, edgecolor='black', linewidth=0.5, zorder=1000)

def visualize_gpcca(n_macrostates, ax, mus, macrostate_assignments, viz_gpcca=True, verbose=False):
    meta_mus = []
    valid_macrostates = []
    for i in range(n_macrostates):
        cur_mus = mus[macrostate_assignments == i]
        if verbose:
            print(f"Macrostate {i+1} has {cur_mus.shape[0]} points")
        if cur_mus.shape[0] == 0:
            if verbose:
                print(f"Macrostate {i+1} has no points, skipping.")
            continue
        if cur_mus.shape[0] in [1,2]:
            # plot small circle around them
            circle = plt.Circle((cur_mus[0, 0], cur_mus[0, 1]), radius=3, color="Blue", linewidth=3, fill=False)
            if viz_gpcca:
                ax.add_artist(circle)
        else:
            vertices = calc_smooth_convex_hull(cur_mus, smoothing_factor=0.1, num_points=100)
            if viz_gpcca:
                ax.plot(vertices[:, 0], vertices[:, 1], color="Blue", linewidth=3, label=f'Macrostate {i+1}')
        valid_macrostates.append(i)
        meta_mu = np.mean(cur_mus, axis=0)
        if viz_gpcca:
            ax.text(meta_mu[0], meta_mu[1], f'{i+1}', fontsize=20, color="Blue", ha='center', va='center', fontweight='bold')
        meta_mus.append(meta_mu)
    meta_mus = np.array(meta_mus)
    
    return meta_mus, valid_macrostates

# def evaluate_gpcca(n_macrostates, P, method="brandts"):    
#     gpcca = pygpcca.GPCCA(P,
#                           eta=None, # input probably can be None cause reversible assumption?
#                           z="LM",
#                           method=method)
#     gpcca.optimize(n_macrostates)
#     return gpcca

def visualize_arrows_between_mesostates(P, fig, ax, good_cluster_indices, mus, show_colorbar_title=True, in_out_flow=None, show_colorbar=True, min_rate=0.01, max_rate=0.5, time_step_us=1):
    Q = infinitesimal_generator(P, dt=time_step_us) # rate at 1 / us
    avgs = np.zeros_like(P)
    # min_transition_probability = 0.01
    # max_transition_probability = 0.05
    # transitions_vmin, transitions_vmax = np.min(P), np.max(P)
    transitions_cmap = LinearSegmentedColormap.from_list(
        'slight_gray_to_black',
        [plt.cm.Greys(0.25), plt.cm.Greys(1.0)]
    )
    transitions_norm = plt.Normalize(vmin=np.log10(min_rate), vmax=np.log10(max_rate))
    
    if in_out_flow not in [None, "in", "out"]:
        raise ValueError("in_out_flow must be None, 'in', or 'out'")
    
    if in_out_flow == "in":
        # draw arrows pointing inwards
        ax.arrow(-25, -20, 0, 15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)
        ax.arrow(-25, 20, 0, -15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)
    if in_out_flow == "out":
        # draw arrows pointing outwards
        ax.arrow(-25, -5, 0, -15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)
        ax.arrow(-25, 5, 0, 15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)

    
     # Draw arrows between mesostates based on transition rates
    
    for i in range(len(good_cluster_indices) - 1):
        for j in range(i + 1, len(good_cluster_indices)):
            cluster_i = good_cluster_indices[i]
            cluster_j = good_cluster_indices[j]
            if in_out_flow is None:
                rate = (Q[cluster_i, cluster_j] + Q[cluster_j, cluster_i]) / 2
            else:
                center_z = 0
                i_closer = False
                if np.abs(mus[i][1] - center_z) <= np.abs(mus[j][1] - center_z):
                    i_closer = True
                if in_out_flow == "in":
                    rate = Q[cluster_j, cluster_i] if i_closer else Q[cluster_i, cluster_j]
                if in_out_flow == "out":
                    rate = Q[cluster_i, cluster_j] if i_closer else Q[cluster_j, cluster_i]
            if rate > min_rate:  # Only draw lines for significant transitions
                mu_i = mus[i]
                mu_j = mus[j]
                color = transitions_cmap(transitions_norm(np.log10(rate)))
                ax.plot([mu_i[0], mu_j[0]], [mu_i[1], mu_j[1]], color=color, alpha=0.8, linewidth=2, zorder=-999)
    # Add transition colorbar
# Add transition colorbar
    if show_colorbar:
        # Create a ScalarMappable to link the cmap and norm to the colorbar
        sm = plt.cm.ScalarMappable(cmap=transitions_cmap, norm=transitions_norm)
        sm.set_array([]) # Required in some versions of matplotlib to avoid errors
        
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cbar = fig.colorbar(sm, cax=cax)
        
        # cbar.ax.set_title(r'$\frac{P[i,j] + P[j,i]}{2}$', fontsize=14)
        if show_colorbar_title:
            cbar.ax.set_title(r'Transition Rate $\log_{10} (\frac{1}{\mu s})$', fontsize=14)

def adjust_nucleus_cytoplasm_mus(clusters, P, good_cluster_indices, mus, pie_scaling=15, z_cyto_center=35.0, z_nuc_center=-35.0):
    stationary_dist = np.power(stationary_distribution(P), 1/3)
    radii = stationary_dist[good_cluster_indices] * pie_scaling
    mus_adjusted = np.copy(mus)
    for i, cluster_i in enumerate(good_cluster_indices):
        cluster = clusters[cluster_i]
        rads = calc_non_spoke_cluster_makeup(cluster)
        if rads[0] > 0.99 or rads[-1] > 0.99:
            radius = radii[i]
            area = np.pi * radius**2
            rect_width = 45
            rect_height = area / rect_width
            if rads[-1] > 0.99:  # Cytoplasm: center at +z_cyto_center, bottom edge at z_cyto_center - rect_height/2
                mus_adjusted[i][1] = z_cyto_center - rect_height / 2
            else:  # Nucleus: center at z_nuc_center, top edge at z_nuc_center + rect_height/2
                mus_adjusted[i][1] = z_nuc_center + rect_height / 2
    return mus_adjusted

def visualize_pie_mesostates(clusters, P, ax2, good_cluster_indices, mus, pie_colors, alpha = 0.8, add_nucleus_cytoplasm_text=True, pie_scaling = 15, dots_only=False, z_cyto_center=35.0, z_nuc_center=-35.0, nucleus_cytoplasm_fontsize=36):
    # stationary_dist = stationary_distribution(P)
    stationary_dist = np.power(stationary_distribution(P), 1/3)  # Adjusted for better visualization
    radii = stationary_dist[good_cluster_indices] * pie_scaling # scale for visibility
    for i, cluster_i in enumerate(good_cluster_indices):
        radius = radii[i]
        cluster = clusters[cluster_i]
        rads = calc_non_spoke_cluster_makeup(cluster)
        if rads[0] > 0.99 or rads[-1] > 0.99:
            # visualize as 1:n rectangle
            # n=5
            area = np.pi * radius**2
            rect_width = 45
            rect_height = area / rect_width
            if rads[-1] > 0.99:  # Cytoplasm
                rect_xy = (mus[i][0] - rect_width / 2, z_cyto_center - rect_height / 2)
                text_y = z_cyto_center
            else:  # Nucleus
                rect_xy = (mus[i][0] - rect_width / 2, z_nuc_center - rect_height / 2)
                text_y = z_nuc_center
            rectangle = FancyBboxPatch(rect_xy,
                                  rect_width, rect_height, facecolor=pie_colors[0] if rads[0] > 0.99 else pie_colors[-1],
                                  zorder = -radius, alpha=alpha, edgecolor='black', linewidth=1.2, boxstyle="Round,pad=0.2,rounding_size=3")
            ax2.add_patch(rectangle)
            if add_nucleus_cytoplasm_text:
                ax2.text(mus[i][0], text_y, 'Nucleus' if rads[0] > 0.99 else 'Cytoplasm', fontsize=nucleus_cytoplasm_fontsize, color="#515151", ha='center', va='center', fontweight='bold', zorder = 9999, fontfamily='Roboto Condensed')
            continue
        
        if dots_only:
            ax2.scatter(mus[i][0], mus[i][1], color='black', s=50, zorder=1000)
            continue
            
        theta_start = 0.0
        paired = sorted(zip(rads, pie_colors, list(range(len(rads)))), reverse=True)
        rads_sorted, pie_colors_sorted, indexes_sorted = map(list, zip(*paired))
        for frac, col, j in zip(rads_sorted, pie_colors_sorted, indexes_sorted):
            if frac == 0:          # skip empty slices
                continue
            if j in[0, len(rads)-1]:
                continue
            wedge = Wedge(center=(mus[i][0], mus[i][1]),
                          r=radius,
                          theta1=theta_start * 360,
                          theta2=(theta_start + frac) * 360,
                          facecolor=col,
                          edgecolor='white',
                          linewidth=0.6,
                          alpha=alpha,
                          zorder = -radius)
            ax2.add_patch(wedge)
            theta_start += frac
        
        circle_border = plt.Circle((mus[i][0], mus[i][1]), radius=radius, facecolor='none', edgecolor='black', linewidth=1.2, zorder=-radius + 0.1)
        ax2.add_patch(circle_border)

def vizualize_spoke_boundries(ax):
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    draw_curved_vertical_line(ax, (-10, -35), (-10, 40), rad=-0.27, color='black', linewidth=2, linestyle='--')
    draw_curved_vertical_line(ax, (10, -35), (10, 40), rad=0.27, color='black', linewidth=2, linestyle='--')
    draw_curved_vertical_line(ax, (-5, -35), (-5, 40), rad=-0.17, color='black', linewidth=2, linestyle='--')
    draw_curved_vertical_line(ax, (5, -35), (5, 40), rad=0.17, color='black', linewidth=2, linestyle='--')
    # ax2.text(-7, -33, 'Spoke 1', fontsize=12, color="black", ha='center', va='center', fontweight='bold')
    # ax2.text(7, -33, 'Spoke 2', fontsize=12, color="black", ha='center', va='center', fontweight='bold')

def estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_cluster_indices):
    mus = []
    covs = []
    for cluster_i in good_cluster_indices:
        cluster = clusters[cluster_i]
        spatial_coordinates = sample_from_cluster(cluster, n_samples, coordinate_edges)[:, [1, 2]] # keep only the y and z coordinates
        
        # Estimate gaussian that generates the cluster
        if cluster[0] > 0.9 or cluster[-1] > 0.9:
            mu_y = np.mean(spatial_coordinates[:, 0])
        else:
            cluster_no_nuc_cyt = cluster.copy()
            cluster_no_nuc_cyt[0] = 0
            cluster_no_nuc_cyt[-1] = 0
            spatial_coordinates_no_nuc_cyt = sample_from_cluster(cluster_no_nuc_cyt, n_samples, coordinate_edges)[:, [1, 2]] # keep only the y and z coordinates
            mu_y = np.mean(spatial_coordinates_no_nuc_cyt[:, 0])
        mu_z = np.mean(spatial_coordinates[:, 1])
        mu = np.array([mu_y, mu_z])
        mus.append(mu)
        cov = np.cov(spatial_coordinates, rowvar=False)
        covs.append(cov)
    return np.array(mus), np.array(covs)

def estimate_clusters_mu_2(clusters_3d_locations, good_cluster_indices):
    """
    Estimates the mus via the actual locations of the relevant kaps. 
    clusters_3d_locations: 3d locations of the clusters 
    good_cluster_indices: indices of the clusters we want to visualize, i.e. those that are mostly in the nucleus, cytoplasm, or channel
    """
    mus = []
    for cluster_i in good_cluster_indices:
        mus.append(clusters_3d_locations[cluster_i][[1, 2]])
    return np.array(mus)
    

def estimate_unprojected_mus(n_samples, clusters, coordinate_edges, good_cluster_indices):
    mus = []
    for cluster_i in good_cluster_indices:
        cluster = clusters[cluster_i]
        spatial_coordinates = sample_from_cluster(cluster, n_samples, coordinate_edges) # keep all coordinates
        
        mu_x = np.mean(spatial_coordinates[:, 0])
        mu_z = np.mean(spatial_coordinates[:, 2])
        if cluster[0] > 0.9 or cluster[-1] > 0.9:
            mu_y = np.mean(spatial_coordinates[:, 1])
        else:
            cluster_no_nuc_cyt = cluster.copy()
            cluster_no_nuc_cyt[0] = 0
            cluster_no_nuc_cyt[-1] = 0
            spatial_coordinates_no_nuc_cyt = sample_from_cluster(cluster_no_nuc_cyt, n_samples, coordinate_edges)
            mu_y = np.mean(spatial_coordinates_no_nuc_cyt[:, 1])
        mu = np.array([mu_x, mu_y, mu_z])
        mus.append(mu)
    return np.array(mus)

def visualize_mesostates_guassians(n_samples, clusters, coordinate_edges, ax1, good_cluster_indices, colors, mus, covs, confidence_level=0.1, viz_contours=True):
    for color_i, cluster_i in enumerate(good_cluster_indices):
        cluster = clusters[cluster_i]
        
        # Plot the points
        # spatial_coordinates = sample_from_cluster(cluster, n_samples, coordinate_edges)[:, [1, 2]] # keep only the y and z coordinates
        # ax1.scatter(spatial_coordinates[:, 0], spatial_coordinates[:, 1], 
        #         alpha=0.3, color=colors[color_i], s=1, label=f'Cluster {cluster_i+1}')
        
        # Estimate gaussian that generates the cluster
        mu = mus[color_i]
        cov = covs[color_i]
        rv = multivariate_normal(mu, cov)
        # plot contour of the gaussian at 95% confidence level
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        chi2_val = chi2.ppf(confidence_level, df=2)
        # colors=colors[color_i]
        if viz_contours:
            ax1.contour(X, Y, rv.pdf(pos), levels=[np.exp(-chi2_val/2) * rv.pdf(mu)], colors="#d98c0b", linewidths=1.5, alpha=0.5)
    return

def pick_good_clusters(n_samples, clusters, coordinate_edges, verbose=False):
    good_cluster_indices = []
    for i, cluster in enumerate(clusters):
        # Include clusters with >= 75% in the nuc and cyt microstates
        if ((cluster[0]) >= 0.75 or (cluster[-1] >= 0.75)):
            good_cluster_indices.append(i)
            continue 
        
        # Include clusters with > 80% of sampled points in [-(1/2)*pi, (1/2)*pi] z in [-20 25]
        spatial_coordinates = sample_from_cluster(cluster, n_samples, coordinate_edges)
        min_phi = -np.pi / 2
        max_phi = np.pi / 2
        phi = np.arctan2(spatial_coordinates[:, 1], spatial_coordinates[:, 0])
        z = spatial_coordinates[:, 2]
        spatial_coordinates = spatial_coordinates[(phi >= min_phi) & (phi <= max_phi) | (z < -20) | (z > 25)]
        if spatial_coordinates.shape[0] < 0.8 * n_samples:
            if verbose:
                print(f"Cluster {i+1} has too few points ({spatial_coordinates.shape[0]}), skipping.")
            continue  # Skip empty clusters
        good_cluster_indices.append(i)
    return good_cluster_indices

def set_up_matplotlib(n_clusters, n_macrostates):
    plt.rcParams['text.usetex'] = False # Use LaTeX for text rendering
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 2, 1]},
                                sharey=True)

    fig.set_figheight(10)
    fig.set_figwidth(25)
    fig.suptitle(f'2D projection of transition network of NPC 4 spoke slice, n_clusters={n_clusters}, n_macrostates={n_macrostates}')
    # fig = plt.figure(figsize=(20, 10))
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(221)

    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Z (nm)')
    ax1.set_title('Mesostate Boundaries (10% probability mass)')
    ax1.set_xlim(-40, 40)
    ax1.set_ylim(-40, 40)
    ax1.set_aspect('equal', adjustable='box')

    ax2.set_xlabel('X (nm)')
    ax2.set_title('Mesostate Composition')
    ax2.set_xlim(-40, 40)
    ax2.set_ylim(-40, 40)
    ax2.set_aspect('equal', adjustable='box')
 
    ax3.set_xlabel('X (nm)')
    ax3.set_title(r'Coarse Grained Markov Transition Network $\log_{10} (\frac{1}{\mu s})$')
    ax3.set_xlim(-20, 20)   
    ax3.set_ylim(-40, 40)
    ax3.set_aspect('equal', adjustable='box')
    return fig,ax1,ax2,ax3

def load_cluster_data(tm_path, cluster_path):
    with open(cluster_path, "rb") as f:
            pca_cluster = pickle.load(f)
    clusters = pca_cluster.get_inverse_centers()
    clusters[clusters < 0.001] = 0  # Set negative probabilities to zero
    anchor_coordinates = get_sorted_anchor_coordinates_np()[1]
    coordinate_edges = calc_coordinate_edges_dict(anchor_coordinates)
    with open(tm_path, "rb") as f:
        P = pickle.load(f)
    return clusters,coordinate_edges,P,pca_cluster

def calc_cluster_avgs(pca_cluster, embedded_path):
    with open(embedded_path, "rb") as f:
        embedded_sections = pickle.load(f)
    reshaped_embedded = np.zeros((embedded_sections.shape[0] * embedded_sections.shape[2], embedded_sections.shape[1])) # (n_diffusers * 8 * n_sections, 218)
    for i in range(embedded_sections.shape[0]):
        for j in range(embedded_sections.shape[2]):
            reshaped_embedded[i * embedded_sections.shape[2] + j] = embedded_sections[i, :, j]
    cluster_avgs = np.zeros_like(pca_cluster.get_inverse_centers())
    n_clusters = cluster_avgs.shape[0]
    y = pca_cluster.y
    for i in range(n_clusters):
        cluster_avgs[i] = np.mean(reshaped_embedded[y == i], axis=0)
    pass

    return cluster_avgs

def add_npc_scaffold_picture(ax):
    img = mpimg.imread('data/volume_flat_90x90_0364level.png')
    ax.imshow(img, extent=[-45, 45, -45, 45], zorder=-1000, aspect='auto', alpha=0.1)
    
def hide_axii_and_show_scale_bar(ax, show_scale_bar=True, show_scale_text=True):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_scale_bar:
        scale_bar_start = (-35, -30)
        scale_bar_length = 10  # nm
        ax.plot(
            [scale_bar_start[0], scale_bar_start[0] + scale_bar_length],
            [scale_bar_start[1], scale_bar_start[1]],
            color='black',
            linewidth=2,
            zorder=10,
        )
        if show_scale_text:
            ax.text(
                scale_bar_start[0] + scale_bar_length / 2,
                scale_bar_start[1] - 3,
                r'$100\,\mathrm{\AA}$',
                ha='center',
                va='top',
                fontsize=14,
                zorder=10,
            )

PIE_COLORS_DICT = {'nuc': "#dddddd", # old = "#aeccdb"
                   'nuc_channel': "#20c9df",
                   'mid_channel': "#AC07A9",
                   'Nup2': "#3274a1",
                   'Nup60': "#b3d495",
                   'Nup1': "#40923a",
                   'Nup145': "#a5a56e",
                   'Nup49': "#ca3335",
                   'Nup57': "#4320df",
                   'Nsp1_cyt': "#df7f20",
                   'Nsp1_inner': "#e84393",
                   'Nsp1': "#df7f20",
                   'Nup100': "#c8b6d2",
                   'Nup159': "#6a498e",
                   'Nup116': "#efa6a5",
                   'cyt_channel' : "#dfcc20",
                   'cyt': "#dddddd"} # old = "#a05e39"

PIE_COLORS = list(PIE_COLORS_DICT.values())
PIE_COLORS_LABELS = ['Nucleus'] + \
                    ['Unbound Channel (Nuc)'] + \
                    ['Unbound Channel (Mid)'] + \
                    ['Nup2', 'Nup60', 'Nup1', 'Nup145', 'Nup49', 'Nup57', 'Nsp1_cyt', 'Nsp1_inner', 'Nsp1', 'Nup100', 'Nup159', 'Nup116'] + \
                    ['Unbound Channel (Cyt)'] + \
                    ['Cytoplasm']

def master_plot(tm_path, cluster_path, embedded_path, n_samples=2000, n_macrostates=10, verbose=False):
    # set seed for reproducibility #
    np.random.seed(123)
    
    # Load the cluster data #
    clusters, coordinate_edges, P, pca_cluster = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
    n_clusters = clusters.shape[0]
    
    # calculate points average for each cluster
    # cluster_avgs = calc_cluster_avgs(pca_cluster, embedded_path=embedded_path)

    # Set up matplotlib figure #
    fig, ax1, ax2, ax3 = set_up_matplotlib(n_clusters, n_macrostates)

    # Pick clusters to include in the plot (since it is a "projection") #
    good_mesostate_indices = pick_good_clusters(n_samples, clusters, coordinate_edges, verbose=verbose)

    # Set colors for the clusters #
    colors = plt.cm.turbo(np.linspace(0, 1, len(good_mesostate_indices)))
    
    # Calculate cluster centers and covariances in the "projection" and visualize them #
    mus, covs = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_mesostate_indices)
    visualize_mesostates_guassians(n_samples, clusters, coordinate_edges, ax1, good_mesostate_indices, colors, mus, covs, viz_contours=False)

    # Add centers as pie charts of the makeup of each point #
    # ax2.scatter(mu[0], mu[1], color=colors[color_i], s=50, label=f'Cluster {cluster_i+1} Mean', edgecolor='black')
    

    # visualize_pie_mesostates(clusters, P, ax2, good_mesostate_indices, mus, PIE_COLORS)

    # draw transtion rate lines between pairs of clusters (ax1) #
    visualize_arrows_between_mesostates(P, fig, ax1, good_mesostate_indices, mus)
    visualize_arrows_between_mesostates(P, fig, ax2, good_mesostate_indices, mus)
    
    # do GPCCA on the transition matrix P #
    # gpcca = evaluate_gpcca(n_macrostates, P)
    # macrostate_assignments = gpcca.macrostate_assignment[good_mesostate_indices]
    # macrostate_mus, valid_macrostates = visualize_gpcca(n_macrostates, ax2, mus, macrostate_assignments, viz_gpcca=False, verbose=verbose)
    # n_valid_macrostates = len(valid_macrostates)
    # coarse_grained_P = gpcca.coarse_grained_transition_matrix[valid_macrostates][:, valid_macrostates]
    # coarse_grained_stationary_distribution = gpcca.coarse_grained_stationary_probability[valid_macrostates]
    
    
    # add nup annotations #
    visualize_nup_annotations(ax1, PIE_COLORS, PIE_COLORS_LABELS)
    visualize_nup_annotations(ax2, PIE_COLORS, PIE_COLORS_LABELS, add_scatters=True)

    # Add coarse grained MC visualization #
    # visualize_coarse_grained(n_valid_macrostates, ax3, coarse_grained_P, coarse_grained_stationary_distribution, macrostate_mus)
    
    # Add spoke boundries
    # vizualize_spoke_boundries(ax1)
    # vizualize_spoke_boundries(ax2)
    
    
    # something is removing ticks so i have to add them here
    ax3.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax3.set_xticks(np.arange(-20, 20 + 1, 10))
    # ax3.set_yticks(np.arange(-30, 40 + 1, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    # fig.tight_layout()
    
    # Show the plot #
    plt.show()

# todo:
# REMEMBER I CAN SHOW THE PROBABLITY MASSES (panel a)
# REMEMBER I CAN SHOW THE MACROSTATE BOUNDARIES (panel b)


def pick_good_clusters_by_mu_angle(mus, clusters, angle_threshold_degrees=90, angle_shift_degrees=0):
    good_cluster_indices = [] 
    angle_threshold_radians = np.radians(angle_threshold_degrees)
    angle_shift_radians = np.radians(angle_shift_degrees)

    
    # Include clusters with >= 75% in the nuc and cyt microstates
    for cluster_i, cluster in enumerate(clusters):
        if ((cluster[0]) >= 0.75 or (cluster[-1] >= 0.75)):
            good_cluster_indices.append(cluster_i)
            continue 
        
        mu = mus[cluster_i]
        angle = np.arctan2(mu[1], mu[0]) + angle_shift_radians  # angle in radians
        if -angle_threshold_radians <= angle <= angle_threshold_radians:
            good_cluster_indices.append(cluster_i)
    return good_cluster_indices

def visualize_vector_field_mesostates(P, fig, ax, good_cluster_indices, mus, show_colorbar_title=True, in_out_flow=None, show_colorbar=True, min_rate=0.01, max_rate=0.5, arrow_scale=5):
    Q = infinitesimal_generator(P, dt=1) # rate at 1 / us
    
    transitions_cmap = LinearSegmentedColormap.from_list(
        'slight_gray_to_black',
        [plt.cm.Greys(0.25), plt.cm.Greys(1.0)]
    )
    transitions_norm = plt.Normalize(vmin=np.log10(min_rate), vmax=np.log10(max_rate))
    
    if in_out_flow not in [None, "in", "out"]:
        raise ValueError("in_out_flow must be None, 'in', or 'out'")
    
    # 1. Draw the macroscopic layout arrows (Kept exactly as you had them)
    if in_out_flow == "in":
        ax.arrow(-25, -20, 0, 15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)
        ax.arrow(-25, 20, 0, -15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)
    if in_out_flow == "out":
        ax.arrow(-25, -5, 0, -15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)
        ax.arrow(-25, 5, 0, 15, head_width=1, head_length=2, fc='black', ec='black', linewidth=2, zorder=1000)

    # 2. Prepare lists for the vector field (Quiver Plot)
    X, Y, U, V, magnitudes = [], [], [], [], []
    
    for i_idx in range(len(good_cluster_indices)):
        cluster_i = good_cluster_indices[i_idx]
        mu_i = mus[i_idx]
        
        vx, vy = 0.0, 0.0
        
        for j_idx in range(len(good_cluster_indices)):
            if i_idx == j_idx:
                continue
                
            cluster_j = good_cluster_indices[j_idx]
            mu_j = mus[j_idx]
            
            # Calculate the direction (unit vector) from i to j
            dx = mu_j[0] - mu_i[0]
            dy = mu_j[1] - mu_i[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist == 0:
                continue
                
            ux, uy = dx / dist, dy / dist
            rate = 0.0
            
            # Determine the appropriate rate based on flow condition
            if in_out_flow is None:
                # Use NET flow from i to j to find the dominant direction
                net_rate = Q[cluster_i, cluster_j] - Q[cluster_j, cluster_i]
                if net_rate > 0: 
                    rate = net_rate
            else:
                center_z = 0
                i_dist = np.abs(mu_i[1] - center_z)
                j_dist = np.abs(mu_j[1] - center_z)
                
                # Filter for inward or outward flow relative to center_z
                if in_out_flow == "in" and j_dist < i_dist: 
                    rate = Q[cluster_i, cluster_j]
                elif in_out_flow == "out" and j_dist > i_dist: 
                    rate = Q[cluster_i, cluster_j]
            
            # Accumulate the vector components
            vx += rate * ux
            vy += rate * uy
            
        # Calculate final net magnitude for this mesostate
        mag = np.sqrt(vx**2 + vy**2)
        
        # Only plot arrows for states with significant net flow
        if mag > min_rate:
            X.append(mu_i[0])
            Y.append(mu_i[1])
            U.append(vx)
            V.append(vy)
            magnitudes.append(mag)
            
    # 3. Draw the Vector Field
    if len(X) > 0:
        # Convert lists to numpy arrays for element-wise division
        U_arr = np.array(U)
        V_arr = np.array(V)
        mags_arr = np.array(magnitudes)
        
        # Normalize the vectors so all arrows have a length of 1
        U_normalized = U_arr / mags_arr
        V_normalized = V_arr / mags_arr
        
        # Clip magnitudes so they map cleanly to your colormap norm without throwing log warnings
        clipped_mags = np.clip(mags_arr, min_rate, max_rate)
        colors = np.log10(clipped_mags)
        
        U_final = U_normalized * arrow_scale
        V_final = V_normalized * arrow_scale
        ax.quiver(X, Y, U_normalized * arrow_scale, V_normalized * arrow_scale, colors, cmap=transitions_cmap, norm=transitions_norm, 
                  pivot='tail', scale=25, width=0.005, alpha=0.9, zorder=-999)

    # 4. Add transition colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=transitions_cmap, norm=transitions_norm)
        sm.set_array([]) 
        
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cbar = fig.colorbar(sm, cax=cax)
        
        if show_colorbar_title:
            cbar.ax.set_title(r'Net Rate $\log_{10} (\frac{1}{\mu s})$', fontsize=14)

def comparison_plot(base_tm_path, base_cluster_path, radii, n_sites, title, in_out_flow=None, ignored_nup_types=None, add_mini_titles=False, pie_scaling=15, min_rate=0.01, max_rate=5, time_step_us=5, show_scale_bars=True, swap_axes=False, use_actual_mus_path=None, dots_only=False, add_nucleus_cytoplasm_text=True, nucleus_cytoplasm_fontsize=36):
    n_samples = 2000
    
    # 1. Determine figure grid dimensions based on the swap_axes argument
    if swap_axes:
        fig_x = len(radii)
        fig_y = len(n_sites)
    else:
        fig_x = len(n_sites)
        fig_y = len(radii)
    
    # --- FIX 1: Adjust Figure Dimensions to Match Data Aspect Ratio ---
    plot_width = 10
    plot_height = plot_width * (92.0 / 60.0)  # 15.333
    
    fig, axes = plt.subplots(fig_y, fig_x, 
                             figsize=(plot_width * fig_x, plot_height * fig_y), 
                             squeeze=False, 
                             gridspec_kw={'wspace': 0.0, 'hspace': 0.02})
    
    # Adjusted title y-position for the taller figure
    fig.suptitle(title, fontsize=20, y=0.95)
    
    pie_colors = PIE_COLORS
    if ignored_nup_types is not None:
        pie_colors = PIE_COLORS.copy()
        for nup_type in ignored_nup_types:
            if nup_type in PIE_COLORS_DICT:
                color_to_remove = PIE_COLORS_DICT[nup_type]
                if color_to_remove in pie_colors:
                    pie_colors.remove(color_to_remove)
 
    for j_r, r in enumerate(radii):
        print("kda: ", radius_a_to_kda(r))
        for i_n, n in enumerate(n_sites):
            
            # 2. Determine the correct row and column index for the current plot
            if swap_axes:
                row_idx = i_n
                col_idx = j_r
            else:
                row_idx = j_r
                col_idx = i_n
                
            ax = axes[row_idx, col_idx]
            
            if add_mini_titles:
                ax.set_title(f'Radius: {r} nm, n_sites: {n}', fontsize=14)
            # set range of axes
            ax.set_xlim(-30, 30)
            ax.set_ylim(-46, 46)
            
            # This ensures 1 unit on x = 1 unit on y (isometric)
            ax.set_aspect('equal', adjustable='box')
            
            # hide axii and show scale bar
            hide_axii_and_show_scale_bar(ax, show_scale_bar=show_scale_bars, show_scale_text=False)
            
            tm_path = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n))
            cluster_path = base_cluster_path.replace("#r#", str(r)).replace("#n#", str(n))
            
            try:
                clusters, coordinate_edges, P, pca_cluster = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
            except Exception as e:
                print(f"Could not load data for radius {r} nm, n_sites {n}: {e}")
                continue
            
            good_mesostate_indices = list(range(len(clusters)))
 
            unprojected_mus = estimate_unprojected_mus(n_samples, clusters, coordinate_edges, good_mesostate_indices)
            good_mesostate_indices = pick_good_clusters_by_mu_angle(unprojected_mus, clusters, angle_threshold_degrees=92, angle_shift_degrees=0)
            if use_actual_mus_path is not None:
                with open(use_actual_mus_path.replace("#r#", str(r)).replace("#n#", str(n)), "rb") as f:
                    actual_mus = pickle.load(f)
                mus = estimate_clusters_mu_2(actual_mus, good_mesostate_indices)
            else:
                mus, covs = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_mesostate_indices)

            mus = adjust_nucleus_cytoplasm_mus(clusters, P, good_mesostate_indices, mus, pie_scaling=pie_scaling)
            
            # 3. Update colorbar logic to check the dynamic row/col index (top-right plot)
            show_colorbar = (row_idx == 0 and col_idx == fig_x - 1)  
            
            visualize_arrows_between_mesostates(P, fig, ax, good_mesostate_indices, mus, show_colorbar_title=False, in_out_flow=in_out_flow, show_colorbar=show_colorbar, min_rate=min_rate, max_rate=max_rate, time_step_us=time_step_us)
            visualize_pie_mesostates(clusters, P, ax, good_mesostate_indices, mus, PIE_COLORS, add_nucleus_cytoplasm_text=add_nucleus_cytoplasm_text, pie_scaling=pie_scaling, dots_only=dots_only, nucleus_cytoplasm_fontsize=nucleus_cytoplasm_fontsize)
            add_npc_scaffold_picture(ax)
            ax.set_aspect('equal', adjustable='box')
    return fig


def comparison_plot_custom(
    tm_paths: List[str],
    cluster_paths: List[str],
    titles: List[str],
    n_rows: int,
    n_cols: int,
    title: str,
    in_out_flow: Optional[str],
    ignored_nup_types: Optional[List[str]],
    add_mini_titles: bool,
    pie_scaling: float,
    min_rate: float,
    max_rate: float,
    time_step_us: float,
    show_scale_bars: bool,
    use_actual_mus_paths: Optional[List[str]],
    dots_only: bool,
    add_nucleus_cytoplasm_text: bool = True,
    nucleus_cytoplasm_fontsize: int = 36,
) -> plt.Figure:
    n_samples: int = 2000
    
    plot_width: float = 10.0
    plot_height: float = plot_width * (92.0 / 60.0)
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(plot_width * n_cols, plot_height * n_rows),
        squeeze=False,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.02}
    )
    
    fig.suptitle(title, fontsize=20, y=0.95)
    
    pie_colors: List[str] = PIE_COLORS
    if ignored_nup_types is not None:
        pie_colors = PIE_COLORS.copy()
        for nup_type in ignored_nup_types:
            if nup_type in PIE_COLORS_DICT:
                color_to_remove: str = PIE_COLORS_DICT[nup_type]
                if color_to_remove in pie_colors:
                    pie_colors.remove(color_to_remove)
                    
    num_plots: int = len(tm_paths)
    for idx in range(n_rows * n_cols):
        row_idx: int = idx // n_cols
        col_idx: int = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        if idx >= num_plots:
            ax.axis('off')
            continue
            
        if add_mini_titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=14)
            
        ax.set_xlim(-30, 30)
        ax.set_ylim(-46, 46)
        ax.set_aspect('equal', adjustable='box')
        
        hide_axii_and_show_scale_bar(ax, show_scale_bar=show_scale_bars, show_scale_text=False)
        
        tm_path: str = tm_paths[idx]
        cluster_path: str = cluster_paths[idx]
        
        try:
            clusters, coordinate_edges, P, pca_cluster = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
        except Exception as e:
            print(f"Could not load data for path {tm_path}: {e}")
            continue
            
        good_mesostate_indices: List[int] = list(range(len(clusters)))
        
        unprojected_mus: np.ndarray = estimate_unprojected_mus(n_samples, clusters, coordinate_edges, good_mesostate_indices)
        good_mesostate_indices = pick_good_clusters_by_mu_angle(unprojected_mus, clusters, angle_threshold_degrees=92, angle_shift_degrees=0)
        
        mus: np.ndarray
        if use_actual_mus_paths is not None and idx < len(use_actual_mus_paths):
            use_actual_mus_path: str = use_actual_mus_paths[idx]
            with open(use_actual_mus_path, "rb") as f:
                actual_mus: np.ndarray = pickle.load(f)
            mus = estimate_clusters_mu_2(actual_mus, good_mesostate_indices)
        else:
            covs: np.ndarray
            mus, covs = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_mesostate_indices)
            
        mus = adjust_nucleus_cytoplasm_mus(clusters, P, good_mesostate_indices, mus, pie_scaling=pie_scaling)

        show_colorbar: bool = (row_idx == 0 and col_idx == n_cols - 1)
        
        visualize_arrows_between_mesostates(P, fig, ax, good_mesostate_indices, mus, show_colorbar_title=False, in_out_flow=in_out_flow, show_colorbar=show_colorbar, min_rate=min_rate, max_rate=max_rate, time_step_us=time_step_us)
        visualize_pie_mesostates(clusters, P, ax, good_mesostate_indices, mus, pie_colors, add_nucleus_cytoplasm_text=add_nucleus_cytoplasm_text, pie_scaling=pie_scaling, dots_only=dots_only, nucleus_cytoplasm_fontsize=nucleus_cytoplasm_fontsize)
        add_npc_scaffold_picture(ax)
        ax.set_aspect('equal', adjustable='box')
        
    return fig





def comparison_vertical_plot(
    tm_paths: List[str],
    cluster_paths: List[str],
    titles: List[str],
    n_rows: int = 1,
    n_cols: int = 4,
    spoke_index: int = 0,
    time_step_us: float = 5.0,
    pie_scaling: float = 12.0,
    min_rate: float = 0.001,
    connect_threshold: float = 0.05,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    neighbor_only: bool = False,
    rate_based_thickness: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    color_by: Union[str, List[str]] = "nup",
    show_arrows: Union[bool, List[bool]] = True,
    show_stationary_dist: Union[bool, List[bool]] = True,
    arrow_opacity: Union[float, List[float]] = 1.0,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    show_colorbar: bool = True,
    legend_nup_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_nup_ncol: int = 4,
    colorbar_fraction: float = 0.046,
    colorbar_pad: Optional[float] = None,
    show_pore_residency: Union[bool, List[bool]] = False,
    show_y_axis: bool = True,
    legend_ncol: Optional[int] = None,
    legend_loc: str = "upper center",
    legend_frameon: bool = False
) -> plt.Figure:
    num_plots = len(tm_paths)
    plot_data = []
    
    def to_list(val, default_val):
        if isinstance(val, list):
            if len(val) < num_plots:
                val = val + [default_val] * (num_plots - len(val))
            return val
        return [val] * num_plots

    color_by_list = to_list(color_by, "nup")
    show_arrows_list = to_list(show_arrows, True)
    show_stationary_dist_list = to_list(show_stationary_dist, True)
    arrow_opacity_list = to_list(arrow_opacity, 1.0)
    show_pore_residency_list = to_list(show_pore_residency, False)
    
    def get_standard_rate(i_idx, j_idx, states_list, pi_array, Q_matrix):
        state_i = states_list[i_idx]
        state_j = states_list[j_idx]
        numerator = sum(pi_array[a] * sum(Q_matrix[a, b] for b in state_j['indices']) for a in state_i['indices'])
        denominator = sum(pi_array[a] for a in state_i['indices'])
        return numerator / denominator if denominator > 0 else 0.0

    def get_rate(i_idx, j_idx, states_list, pi_array, Q_matrix):
        if neighbor_only:
            if abs(j_idx - i_idx) != 1:
                return 0.0
            if j_idx == i_idx - 1:  # going down
                return sum(get_standard_rate(i_idx, m, states_list, pi_array, Q_matrix) for m in range(i_idx))
            if j_idx == i_idx + 1:  # going up
                return sum(get_standard_rate(i_idx, m, states_list, pi_array, Q_matrix) for m in range(i_idx + 1, len(states_list)))
        else:
            return get_standard_rate(i_idx, j_idx, states_list, pi_array, Q_matrix)
    
    # 1. Process data for each path
    for idx in range(num_plots):
        tm_path = tm_paths[idx]
        cluster_path = cluster_paths[idx]
        
        clusters, coordinate_edges, P, pca_cluster = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
        pi = stationary_distribution(P)
        Q = infinitesimal_generator(P, dt=time_step_us)
        
        # Compute committors on clusters
        n_clusters = P.shape[0]
        idx_nuc = [i for i in range(n_clusters) if clusters[i][0] >= 0.75]
        idx_cyt = [i for i in range(n_clusters) if clusters[i][-1] >= 0.75]
        idx_trans = [i for i in range(n_clusters) if i not in idx_nuc and i not in idx_cyt]

        I_C = np.eye(len(idx_trans)) if len(idx_trans) > 0 else np.array([])
        P_C = P[np.ix_(idx_trans, idx_trans)] if len(idx_trans) > 0 else np.array([])
        
        # Nucleus committor
        q_nuc = np.zeros(n_clusters)
        for i in idx_nuc:
            q_nuc[i] = 1.0
        if len(idx_trans) > 0 and len(idx_nuc) > 0:
            b_nuc = np.sum(P[np.ix_(idx_trans, idx_nuc)], axis=1)
            try:
                q_nuc_trans = np.linalg.solve(I_C - P_C, b_nuc)
                q_nuc[idx_trans] = q_nuc_trans
            except np.linalg.LinAlgError:
                pass

        # Cytoplasm committor
        q_cyt = np.zeros(n_clusters)
        for i in idx_cyt:
            q_cyt[i] = 1.0
        if len(idx_trans) > 0 and len(idx_cyt) > 0:
            b_cyt = np.sum(P[np.ix_(idx_trans, idx_cyt)], axis=1)
            try:
                q_cyt_trans = np.linalg.solve(I_C - P_C, b_cyt)
                q_cyt[idx_trans] = q_cyt_trans
            except np.linalg.LinAlgError:
                pass
        
        # Map microstates to spokes
        def get_microstate_spoke(m: int):
            if m == 0 or m == 457:
                return None
            if 1 <= m <= 8:
                return m - 1
            if 225 <= m <= 232:
                return m - 225
            if 449 <= m <= 456:
                return m - 449
            adj = m - 9 if m < 225 else m - 17
            return (adj // 2) % 8

        # Pick clusters of interest dynamically
        spoke_0_indices = []
        for i, cluster in enumerate(clusters):
            if not (cluster[0] >= 0.75 or cluster[-1] >= 0.75):
                spoke_masses = np.zeros(8)
                for m in range(1, 457):
                    s = get_microstate_spoke(m)
                    if s is not None:
                        spoke_masses[s] += cluster[m]
                if np.argmax(spoke_masses) == spoke_index:
                    spoke_0_indices.append(i)
                    
        connected_bulk_indices = []
        for i, cluster in enumerate(clusters):
            if cluster[0] >= 0.75 or cluster[-1] >= 0.75:
                connected_bulk_indices.append(i)
                    
        good_cluster_indices = spoke_0_indices + connected_bulk_indices
        
        # Calculate Z centroids (X is 0)
        n_samples = 2000
        mus = []
        for idx_c in good_cluster_indices:
            cluster = clusters[idx_c]
            spatial_coordinates = sample_from_cluster(cluster, n_samples, coordinate_edges)
            z_coords = spatial_coordinates[:, 2]
            mus.append(np.array([0.0, np.mean(z_coords)]))
        mus = np.array(mus)
        
        # Initialize list of states to be merged
        states = []
        safe_pi_for_indices = np.array([pi[c] if c < len(pi) else 0.0 for c in good_cluster_indices])
        scaled_pi = np.power(safe_pi_for_indices, 1/3)
        radii = scaled_pi * pie_scaling
        radii = np.clip(radii, 1.2, 3.2)
        
        for i, idx_c in enumerate(good_cluster_indices):
            cluster = clusters[idx_c]
            rads = calc_non_spoke_cluster_makeup(cluster)
            is_nuc = rads[0] > 0.75
            is_cyt = rads[-1] > 0.75
            
            center_pos = mus[i].copy()
            if is_nuc:
                center_pos[1] = -35.0
            elif is_cyt:
                center_pos[1] = 35.0
                
            states.append({
                'indices': [idx_c],
                'pi': pi[idx_c] if idx_c < len(pi) else 0.0,
                'center': center_pos,
                'radius': radii[i],
                'makeups': [cluster],
                'makeup': cluster,
                'is_nuc': is_nuc,
                'is_cyt': is_cyt
            })
            
        while True:
            n_states = len(states)
            max_overlap = -1.0
            pair_to_merge = None
            
            for i in range(n_states):
                for j in range(i + 1, n_states):
                    dist = np.linalg.norm(states[i]['center'] - states[j]['center'])
                    overlap = (states[i]['radius'] + states[j]['radius']) - dist
                    if overlap > 0 and overlap > max_overlap:
                        max_overlap = overlap
                        pair_to_merge = (i, j)
                        
            if pair_to_merge is None:
                break
                
            i, j = pair_to_merge
            s_i = states[i]
            s_j = states[j]
            
            total_pi = s_i['pi'] + s_j['pi']
            new_center = (s_i['center'] * s_i['pi'] + s_j['center'] * s_j['pi']) / total_pi
            
            new_is_nuc = s_i['is_nuc'] or s_j['is_nuc']
            new_is_cyt = s_i['is_cyt'] or s_j['is_cyt']
            
            new_makeups = s_i['makeups'] + s_j['makeups']
            if new_is_nuc:
                new_makeup = np.zeros_like(s_i['makeup'])
                new_makeup[0] = 1.0
            elif new_is_cyt:
                new_makeup = np.zeros_like(s_i['makeup'])
                new_makeup[-1] = 1.0
            else:
                new_makeup = np.mean(new_makeups, axis=0)
                new_makeup = new_makeup / np.sum(new_makeup)
            
            new_indices = s_i['indices'] + s_j['indices']
            new_radius = np.clip(np.power(total_pi, 1/3) * pie_scaling, 1.2, 3.2)
            
            new_state = {
                'indices': new_indices,
                'pi': total_pi,
                'center': new_center,
                'radius': new_radius,
                'makeups': new_makeups,
                'makeup': new_makeup,
                'is_nuc': new_is_nuc,
                'is_cyt': new_is_cyt
            }
            
            states.pop(j)
            states.pop(i)
            states.append(new_state)
            
        states.sort(key=lambda s: s['center'][1])
        
        plot_data.append({
            'states': states,
            'pi': pi,
            'Q': Q,
            'q_nuc': q_nuc,
            'q_cyt': q_cyt
        })
        
    # 2. Find unified label_col_x and max_rate_all across all subplots
    global_max_x_mid = 0.0
    max_rate_all = -np.inf
    for idx, data in enumerate(plot_data):
        states = data['states']
        pi = data['pi']
        Q = data['Q']
        if not show_arrows_list[idx]:
            continue
        for i in range(len(states)):
            for j in range(len(states)):
                if i == j:
                    continue
                rate = get_rate(i, j, states, pi, Q)
                
                if (abs(j - i) == 1) if neighbor_only else (rate > min_rate):
                    if rate > max_rate_all:
                        max_rate_all = rate
                
                if (abs(j - i) == 1) if neighbor_only else (rate > 0.05):
                    p1 = states[i]['center']
                    p2 = states[j]['center']
                    dy = p2[1] - p1[1]
                    abs_diff = abs(j - i)
                    rad_val = 0.08 + 0.075 * abs_diff
                    x_mid = 0.5 * rad_val * abs(dy)
                    if x_mid > global_max_x_mid:
                        global_max_x_mid = x_mid

    if global_max_x_mid == 0.0:
        global_max_x_mid = 4.0
    label_col_x = global_max_x_mid + 3.5

    effective_min_rate = min_rate if min_rate > 0.0 else 1e-4
    if max_rate_all <= effective_min_rate:
        max_rate_all = effective_min_rate * 10.0

    custom_gray_cmap = LinearSegmentedColormap.from_list(
        'custom_gray', 
        [plt.cm.Greys(0.25), plt.cm.Greys(1.0)]
    )
    transitions_norm = Normalize(vmin=np.log10(effective_min_rate), vmax=np.log10(max_rate_all))
    
    # 3. Figure Setup
    y_min, y_max = -40.0, 40.0
    if rate_based_thickness:
        half_width = global_max_x_mid + 1.5
        xlim_left = -half_width
        xlim_right = half_width
    else:
        half_width = label_col_x + 1.8
        xlim_left = -half_width - 1.5
        xlim_right = half_width + 3.0
        
    if xlim is not None:
        xlim_left, xlim_right = xlim
        
    data_width = xlim_right - xlim_left
    data_height = y_max - y_min
    
    fig_height = 20.0
    single_plot_width = fig_height * (data_width / data_height)
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(single_plot_width * n_cols, fig_height * n_rows),
        squeeze=False
    )
    
    if title is not None:
        fig.suptitle(title, fontsize=49, y=0.98)
        
    for idx in range(n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        if idx >= num_plots:
            ax.axis('off')
            continue
            
        ax.set_xlim(xlim_left, xlim_right)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('', fontsize=63)
        
        # Only show Y-axis label and ticks on the first column to avoid redundancy, and only if show_y_axis is True
        if col_idx == 0 and show_y_axis:
            ax.set_ylabel('Z (nm)', fontsize=63)
            ax.tick_params(axis='y', which='major', labelsize=54)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xticks([])
        ax.xaxis.set_visible(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if col_idx > 0 or not show_y_axis:
            ax.spines['left'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        data = plot_data[idx]
        states = data['states']
        pi = data['pi']
        Q = data['Q']
        q_nuc = data['q_nuc']
        q_cyt = data['q_cyt']
        
        if idx < len(titles):
            title_text = titles[idx]
            if show_pore_residency_list[idx]:
                pore_res_val = sum(s['pi'] for s in states if not (s['is_nuc'] or s['is_cyt'])) * 8
                title_text += f"\nResidency: {pore_res_val * 100:.1f}%"
            ax.set_title(title_text, fontsize=49)
        
        # Draw states
        circles = []
        for i, state in enumerate(states):
            center_x, center_y = state['center']
            radius = state['radius']
            cluster = state['makeup']
            rads = calc_non_spoke_cluster_makeup(cluster)
            
            if rads[0] > 0.75 or rads[-1] > 0.75:
                if color_by_list[idx] == "committor":
                    color = '#66c2a5' if rads[0] > 0.75 else '#fc8d62'
                    edgecolor = 'white'
                else:
                    color = 'white'
                    edgecolor = 'white'
                center_y = -35 if rads[0] > 0.75 else 35
                state['center'][1] = center_y
                circle = plt.Circle((center_x, center_y), radius=radius, facecolor=color, edgecolor=edgecolor, linewidth=1, zorder=5)
                ax.add_patch(circle)
                circles.append(circle)
                text_str = 'Nucleus' if rads[0] > 0.75 else 'Cytoplasm'
                ax.text(center_x, center_y, text_str[0], fontsize=54, color='black', ha='center', va='center', zorder=6, fontfamily='Roboto Condensed')
            else:
                c_patch = plt.Circle((center_x, center_y), radius=radius, facecolor='none', edgecolor='none', zorder=0)
                ax.add_patch(c_patch)
                circles.append(c_patch)
                
                if color_by_list[idx] == "committor":
                    valid_indices = [c for c in state['indices'] if c < len(q_nuc) and c < len(pi)]
                    if len(valid_indices) > 0 and sum(pi[c] for c in valid_indices) > 0:
                        state_q_nuc = sum(pi[c] * q_nuc[c] for c in valid_indices) / sum(pi[c] for c in valid_indices)
                        state_q_cyt = sum(pi[c] * q_cyt[c] for c in valid_indices) / sum(pi[c] for c in valid_indices)
                    else:
                        state_q_nuc = 0.0
                        state_q_cyt = 0.0
                    
                    q_sum = state_q_nuc + state_q_cyt
                    if q_sum > 0:
                        state_q_nuc /= q_sum
                        state_q_cyt /= q_sum
                    else:
                        state_q_nuc = 0.5
                        state_q_cyt = 0.5
                    
                    theta_start = 0.0
                    for frac, col in [(state_q_cyt, '#fc8d62'), (state_q_nuc, '#66c2a5')]:
                        if frac == 0:
                            continue
                        wedge = Wedge(center=(center_x, center_y),
                                      r=radius,
                                      theta1=theta_start * 360,
                                      theta2=(theta_start + frac) * 360,
                                      facecolor=col,
                                      edgecolor='white',
                                      linewidth=0.5,
                                      alpha=1.0,
                                      zorder=5)
                        ax.add_patch(wedge)
                        theta_start += frac
                else:
                    if color_by_list[idx] == "type":
                        glfg_color = '#ff4d4d'
                        fsfg_color = '#1e90ff'
                        
                        rads_transformed = np.zeros(7)
                        rads_transformed[0] = rads[0]
                        rads_transformed[1] = rads[1]
                        rads_transformed[2] = rads[2]
                        
                        nup100 = rads[12]
                        nup116 = rads[14]
                        nup49 = rads[7]
                        nup57 = rads[8]
                        nup145 = rads[6]
                        nsp1_parts = rads[9] + rads[10] + rads[11]
                        nup1_part = rads[5]
                        
                        glfg_sum = nup100 + nup116 + nup49 + nup57 + nup145 + nsp1_parts * 0.3273 + nup1_part * 0.3475
                        
                        nup159 = rads[13]
                        nup60 = rads[4]
                        nup2 = rads[3]
                        
                        fsfg_sum = nup159 + nup60 + nup2 + nsp1_parts * 0.6727 + nup1_part * 0.6525
                        
                        rads_transformed[3] = glfg_sum
                        rads_transformed[4] = fsfg_sum
                        rads_transformed[5] = rads[15]
                        rads_transformed[6] = rads[16]
                        
                        colors_to_use = [PIE_COLORS[0], PIE_COLORS[1], PIE_COLORS[2], glfg_color, fsfg_color, PIE_COLORS[15], PIE_COLORS[16]]
                        
                        rads_for_sort = rads_transformed
                        colors_for_sort = colors_to_use
                    elif color_by_list[idx] == "z":
                        z_makeup = np.zeros(3)
                        anchor_coordinates = get_sorted_anchor_coordinates()
                        nups = ['Nup2_08', 'Nup2_09', 'Nup2_10', 'Nup2_11', 'Nup2_12', 'Nup2_13', 'Nup2_14', 'Nup2_15', 'Nup60_08', 'Nup60_09', 'Nup60_10', 'Nup60_11', 'Nup60_12', 'Nup60_13', 'Nup60_14', 'Nup60_15', 'Nup2_00', 'Nup2_01', 'Nup2_02', 'Nup2_03', 'Nup2_04', 'Nup2_05', 'Nup2_06', 'Nup2_07', 'Nup60_00', 'Nup60_01', 'Nup60_02', 'Nup60_03', 'Nup60_04', 'Nup60_05', 'Nup60_06', 'Nup60_07', 'Nup1_00', 'Nup1_01', 'Nup1_02', 'Nup1_03', 'Nup1_04', 'Nup1_05', 'Nup1_06', 'Nup1_07', 'Nup145_00', 'Nup145_01', 'Nup145_02', 'Nup145_03', 'Nup145_04', 'Nup145_05', 'Nup145_06', 'Nup145_07', 'Nup49_24', 'Nup49_25', 'Nup49_26', 'Nup49_27', 'Nup49_28', 'Nup49_29', 'Nup49_30', 'Nup49_31', 'Nsp1_40', 'Nsp1_41', 'Nsp1_42', 'Nsp1_43', 'Nsp1_44', 'Nsp1_45', 'Nsp1_46', 'Nsp1_47', 'Nup49_16', 'Nup49_17', 'Nup49_18', 'Nup49_19', 'Nup49_20', 'Nup49_21', 'Nup49_22', 'Nup49_23', 'Nsp1_32', 'Nsp1_33', 'Nsp1_34', 'Nsp1_35', 'Nsp1_36', 'Nsp1_37', 'Nsp1_38', 'Nsp1_39', 'Nup57_24', 'Nup57_25', 'Nup57_26', 'Nup57_27', 'Nup57_28', 'Nup57_29', 'Nup57_30', 'Nup57_31', 'Nup145_08', 'Nup145_09', 'Nup145_10', 'Nup145_11', 'Nup145_12', 'Nup145_13', 'Nup145_14', 'Nup145_15', 'Nup57_16', 'Nup57_17', 'Nup57_18', 'Nup57_19', 'Nup57_20', 'Nup57_21', 'Nup57_22', 'Nup57_23', 'Nup57_00', 'Nup57_01', 'Nup57_02', 'Nup57_03', 'Nup57_04', 'Nup57_05', 'Nup57_06', 'Nup57_07', 'Nup57_08', 'Nup57_09', 'Nup57_10', 'Nup57_11', 'Nup57_12', 'Nup57_13', 'Nup57_14', 'Nup57_15', 'Nsp1_16', 'Nsp1_17', 'Nsp1_18', 'Nsp1_19', 'Nsp1_20', 'Nsp1_21', 'Nsp1_22', 'Nsp1_23', 'Nup49_00', 'Nup49_01', 'Nup49_02', 'Nup49_03', 'Nup49_04', 'Nup49_05', 'Nup49_06', 'Nup49_07', 'Nsp1_24', 'Nsp1_25', 'Nsp1_26', 'Nsp1_27', 'Nsp1_28', 'Nsp1_29', 'Nsp1_30', 'Nsp1_31', 'Nup49_08', 'Nup49_09', 'Nup49_10', 'Nup49_11', 'Nup49_12', 'Nup49_13', 'Nup49_14', 'Nup49_15', 'Nup100_00', 'Nup100_01', 'Nup100_02', 'Nup100_03', 'Nup100_04', 'Nup100_05', 'Nup100_06', 'Nup100_07', 'Nup159_00', 'Nup159_01', 'Nup159_02', 'Nup159_03', 'Nup159_04', 'Nup159_05', 'Nup159_06', 'Nup159_07', 'Nup100_08', 'Nup100_09', 'Nup100_10', 'Nup100_11', 'Nup100_12', 'Nup100_13', 'Nup100_14', 'Nup100_15', 'Nup159_08', 'Nup159_09', 'Nup159_10', 'Nup159_11', 'Nup159_12', 'Nup159_13', 'Nup159_14', 'Nup159_15', 'Nsp1_00', 'Nsp1_01', 'Nsp1_02', 'Nsp1_03', 'Nsp1_04', 'Nsp1_05', 'Nsp1_06', 'Nsp1_07', 'Nsp1_08', 'Nsp1_09', 'Nsp1_10', 'Nsp1_11', 'Nsp1_12', 'Nsp1_13', 'Nsp1_14', 'Nsp1_15', 'Nup116_00', 'Nup116_01', 'Nup116_02', 'Nup116_03', 'Nup116_04', 'Nup116_05', 'Nup116_06', 'Nup116_07', 'Nup116_08', 'Nup116_09', 'Nup116_10', 'Nup116_11', 'Nup116_12', 'Nup116_13', 'Nup116_14', 'Nup116_15']
                        for i in range(458):
                            if i in [0, 457] or (1 <= i <= 8) or (225 <= i <= 232) or (449 <= i <= 456):
                                continue
                            new_i = i
                            if i < 225:
                                new_i -= 9
                            else:
                                new_i -= 17
                            new_i //= 2
                            nup_anchor = nups[new_i]
                            z = anchor_coordinates[nup_anchor][2]
                            if z < -5.0:
                                z_makeup[0] += cluster[i]
                            elif z > 5.0:
                                z_makeup[2] += cluster[i]
                            else:
                                z_makeup[1] += cluster[i]
                        rads_for_sort = list(z_makeup)
                        colors_for_sort = ['#C040B0', '#40BF89', '#FFD255']
                    else:
                        rads_for_sort = rads
                        colors_for_sort = PIE_COLORS
                    
                    if color_by_list[idx] == "type":
                        exclude_indices = {0, 1, 2, 5, 6}
                    elif color_by_list[idx] == "z":
                        exclude_indices = set()
                    else:
                        exclude_indices = {0, 1, 2, 15, 16}

                    
                    theta_start = 0.0
                    paired = sorted(zip(rads_for_sort, colors_for_sort, list(range(len(rads_for_sort)))), reverse=True)
                    rads_sorted, colors_sorted, indexes_sorted = map(list, zip(*paired))
                    for frac, col, j in zip(rads_sorted, colors_sorted, indexes_sorted):
                        if frac == 0:
                            continue
                        if j in exclude_indices:
                            continue
                        wedge = Wedge(center=(center_x, center_y),
                                      r=radius,
                                      theta1=theta_start * 360,
                                      theta2=(theta_start + frac) * 360,
                                      facecolor=col,
                                      edgecolor='white',
                                      linewidth=0.5,
                                      alpha=1.0,
                                      zorder=5)
                        ax.add_patch(wedge)
                        theta_start += frac
                    
            circle_border = plt.Circle((center_x, center_y), radius=radius, facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)
            ax.add_patch(circle_border)
            
            # Show stationary distribution number if specified
            if show_stationary_dist_list[idx]:
                pi_str = f"{state['pi'] * 100:.1f}%"
                ax.text(center_x + radius + 0.6, center_y, pi_str, fontsize=53, ha='left', va='center', color='#1d4ed8',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        
        # Collect active transitions for overlap resolution
        active_transitions = []
        if show_arrows_list[idx]:
            for i in range(len(states)):
                for j in range(len(states)):
                    if i == j:
                        continue
                    rate = get_rate(i, j, states, pi, Q)
                    
                    if (abs(j - i) == 1) if neighbor_only else (rate > min_rate):
                        p1 = states[i]['center']
                        p2 = states[j]['center']
                        d = p2 - p1
                        dist = np.linalg.norm(d)
                        if dist == 0:
                            continue
                        active_transitions.append({
                            'i': i,
                            'j': j,
                            'p1': p1,
                            'p2': p2,
                            'rate': rate,
                            'diff': j - i,
                            'y_mid': (p1[1] + p2[1]) / 2.0
                        })
                    
        # Helper to resolve overlaps along Y axis
        def resolve_overlaps(y_coords, min_dist=4.0, max_iter=1000):
            y = np.array(y_coords, dtype=float)
            n = len(y)
            if n <= 1:
                return y
            idx = np.argsort(y)
            y_sorted = y[idx]
            for _ in range(max_iter):
                moved = False
                for i in range(n - 1):
                    d = y_sorted[i+1] - y_sorted[i]
                    if d < min_dist:
                        overlap = min_dist - d
                        y_sorted[i] -= overlap * 0.5
                        y_sorted[i+1] += overlap * 0.5
                        moved = True
                if not moved:
                    break
            y_out = np.zeros_like(y)
            y_out[idx] = y_sorted
            return y_out

        left_trans = [t for t in active_transitions if t['diff'] < 0]
        right_trans = [t for t in active_transitions if t['diff'] > 0]
        
        if left_trans:
            y_left_adj = resolve_overlaps([t['y_mid'] for t in left_trans])
            for t, y in zip(left_trans, y_left_adj):
                t['y_label'] = y
                
        if right_trans:
            y_right_adj = resolve_overlaps([t['y_mid'] for t in right_trans])
            for t, y in zip(right_trans, y_right_adj):
                t['y_label'] = y

        for t in left_trans + right_trans:
            i, j = t['i'], t['j']
            p1, p2 = t['p1'], t['p2']
            diff = t['diff']
            rate = t['rate']
            y_mid = t['y_mid']
            y_label = t['y_label']
            
            abs_diff = abs(diff)
            rad_val = 0.08 + 0.05 * abs_diff
            rad = rad_val
            
            if rate_based_thickness:
                clipped_rate = np.clip(rate, effective_min_rate, max_rate_all)
                log_min = np.log10(effective_min_rate)
                log_max = np.log10(max_rate_all)
                lw = 1.0 + 7.0 * (np.log10(clipped_rate) - log_min) / (log_max - log_min)
                mutation_scale = 4.0 + 14.0 * (np.log10(clipped_rate) - log_min) / (log_max - log_min)
                alpha = (0.25 + 0.65 * (np.log10(clipped_rate) - log_min) / (log_max - log_min)) * arrow_opacity_list[idx]
                color = custom_gray_cmap(transitions_norm(np.log10(clipped_rate)))
            else:
                lw = 4.0
                mutation_scale = 12.0
                alpha = 0.8 * arrow_opacity_list[idx]
                color = 'black'
                
            arrow = patches.FancyArrowPatch(
                posA=(p1[0], p1[1]),
                posB=(p2[0], p2[1]),
                patchA=circles[i],
                patchB=circles[j],
                arrowstyle=f"-|>,head_width=1.2,head_length=2.4",
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=mutation_scale,
                color=color,
                alpha=alpha,
                lw=lw,
                shrinkA=1.0,
                shrinkB=1.0,
                zorder=4
            )
            ax.add_patch(arrow)
            
            x_mid = 0.5 * rad * (p2[1] - p1[1])
            
            if not rate_based_thickness:
                if diff > 0:
                    text_x = label_col_x + 2.5
                    text_color = 'white'
                    box_fc = "#cc7043"
                    box_ec = "#b35a2b"
                else:
                    text_x = -label_col_x
                    text_color = 'white'
                    box_fc = "#4a6fa5"
                    box_ec = "#3b5984"
                
                ax.plot([x_mid, text_x], [y_mid, y_label], color='gray', linestyle='--', linewidth=2.0, zorder=3)
                
                if rate < 0.001:
                    rate_str = f"{rate:.1e}"
                else:
                    rate_str = f"{rate:.3f}"
                    if rate_str.startswith("0."):
                        rate_str = rate_str[1:]
                ax.text(text_x, y_label, rate_str, color=text_color, fontsize=41, ha='center', va='center',
                        zorder=8, bbox=dict(boxstyle="round,pad=0.15", fc=box_fc, ec=box_ec, lw=0.8, alpha=0.9))
                        
    if show_colorbar and rate_based_thickness and any(show_arrows_list):
        active_axes = [axes[i // n_cols, i % n_cols] for i in range(num_plots) if show_arrows_list[i]]
        if active_axes:
            sm = plt.cm.ScalarMappable(cmap=custom_gray_cmap, norm=transitions_norm)
            sm.set_array([])
            pos = active_axes[-1].get_position()
            cbar_pad = colorbar_pad if colorbar_pad is not None else (0.08 if n_rows == 1 else 0.04)
            cbar_width = colorbar_fraction * (pos.x1 - pos.x0)
            cbar_ax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.y1 - pos.y0])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.ax.set_ylabel(r'transition rate ($\mu\mathrm{s}^{-1}$)', fontsize=54)
            cbar.ax.tick_params(labelsize=45)
            
            def log_tick_formatter(val, pos):
                if abs(val - round(val)) < 1e-9:
                    return f"$10^{{{int(round(val))}}}$"
                return f"$10^{{{val:.1f}}}$"
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        
    legend_handles = []
    if any(cb == "type" for cb in color_by_list):
        glfg_patch = patches.Patch(color='#ff4d4d', label='GLFG')
        fsfg_patch = patches.Patch(color='#1e90ff', label='FSFG')
        legend_handles.extend([glfg_patch, fsfg_patch])
    if any(cb == "z" for cb in color_by_list):
        z_nuc_patch = patches.Patch(color='#C040B0', label='Nup anchor Z < -5 nm')
        z_mid_patch = patches.Patch(color='#40BF89', label='-5 to 5 nm')
        z_cyt_patch = patches.Patch(color='#FFD255', label='Nup anchor Z > 5 nm')
        legend_handles.extend([z_nuc_patch, z_mid_patch, z_cyt_patch])
    if any(cb == "committor" for cb in color_by_list):
        cyt_patch = patches.Patch(color='#fc8d62', label='Committor to Cytoplasm')
        nuc_patch = patches.Patch(color='#66c2a5', label='Committor to Nucleus')
        legend_handles.extend([cyt_patch, nuc_patch])

    if legend_handles:
        bbox = legend_bbox_to_anchor if legend_bbox_to_anchor is not None else ((0.5, 0.96) if n_rows == 1 else (0.5, 0.95))
        ncol = legend_ncol if legend_ncol is not None else len(legend_handles)
        fig.legend(handles=legend_handles, loc=legend_loc, bbox_to_anchor=bbox, fontsize=36, ncol=ncol, frameon=legend_frameon)
        
    if any(cb == "nup" for cb in color_by_list):
        fig.subplots_adjust(bottom=0.15)
        nup_handles = []
        nup_colors = []
        for k in range(3, 15):
            if PIE_COLORS_LABELS[k] == 'Nsp1':
                continue
            patch = patches.Patch(color=PIE_COLORS[k], label=PIE_COLORS_LABELS[k])
            nup_handles.append(patch)
            nup_colors.append(PIE_COLORS[k])
        nup_bbox = legend_nup_bbox_to_anchor if legend_nup_bbox_to_anchor is not None else ((0.5, 0.12) if n_rows == 1 else (0.5, 0.05))
        nup_legend = fig.legend(
            handles=nup_handles,
            loc='upper center',
            bbox_to_anchor=nup_bbox,
            fontsize=36,
            ncol=legend_nup_ncol,
            frameon=False
        )
        for i, text in enumerate(nup_legend.get_texts()):
            text.set_color(nup_colors[i])
            text.set_weight("bold")
                    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
    return fig