import numpy as np
import pickle
from scipy.spatial import ConvexHull
import pygpcca
from scipy.stats import multivariate_normal, chi2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from adjustText import adjust_text
from matplotlib.patches import Wedge, FancyBboxPatch
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import networkx as nx
import itertools as it
from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates, get_sorted_anchor_coordinates_np, infinitesimal_generator, stationary_distribution, radius_a_to_kda

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
    makeup = {'nuc': 0.0, 'nuc_channel': 0.0, 'mid_channel': 0.0, 'Nup2': 0.0, 'Nup60': 0.0, 'Nup1': 0.0, 'Nup145': 0.0, 'Nup49': 0.0, 'Nup57': 0.0, 'Nsp1': 0.0, 'Nup100': 0.0, 'Nup159': 0.0, 'Nup116': 0.0, 'cyt_channel' : 0.0, 'cyt': 0.0}

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
        makeup[nup.split("_")[0]] += cluster[i]
        
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
        cyt_text = ax.text(-30, 30, "Cytoplasm", fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Cytoplasm')])
        cyt_channel_text = ax.text(-30, 28, 'Unbound Channel (Cyt)', fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Unbound Channel (Cyt)')])
        nuc_text = ax.text(-30, -30, "Nucleus", fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Nucleus')])
        nuc_channel_text = ax.text(-30, -28, 'Unbound Channel (Nuc)', fontsize=11, fontweight='bold', color=pie_colors[pie_colors_labels.index('Unbound Channel (Nuc)')])
    else:
        cyt_text = ax.text(-30, 30, "Cytoplasm", fontsize=11, fontweight='bold')
        nuc_text = ax.text(-30, -30, "Nucleus", fontsize=11, fontweight='bold')
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

def evaluate_gpcca(n_macrostates, P, method="brandts"):    
    gpcca = pygpcca.GPCCA(P,
                          eta=None, # input probably can be None cause reversible assumption?
                          z="LM",
                          method=method)
    gpcca.optimize(n_macrostates)
    return gpcca

def visualize_arrows_between_mesostates(P, fig, ax, good_cluster_indices, mus, show_colorbar_title=True, in_out_flow=None, show_colorbar=True):
    Q = infinitesimal_generator(P, dt=1) # rate at 1 / us
    avgs = np.zeros_like(P)
    min_rate = 0.01
    max_rate = 0.5    
    # min_transition_probability = 0.01
    # max_transition_probability = 0.05
    # transitions_vmin, transitions_vmax = np.min(P), np.max(P)
    transitions_cmap = plt.cm.Greys 
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
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=transitions_cmap, norm=transitions_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax.inset_axes([0.8, 0.1, 0.02, 0.8]))
        # cbar.ax.set_title(r'$\frac{P[i,j] + P[j,i]}{2}$', fontsize=14)
        if show_colorbar_title:
            cbar.ax.set_title(r'Transition Rate $\log_{10} (\frac{1}{\mu s})$', fontsize=14)

def visualize_pie_mesostates(clusters, P, ax2, good_cluster_indices, mus, pie_colors, alpha = 0.8, add_nucleus_cytoplasm_text=True, pie_scaling = 15):
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
            rect_xy = (mus[i][0] - rect_width / 2, mus[i][1] if rads[-1] > 0.99 else mus[i][1] - rect_height)
            rectangle = FancyBboxPatch(rect_xy,
                                  rect_width, rect_height, facecolor=pie_colors[0] if rads[0] > 0.99 else pie_colors[-1],
                                  zorder = -radius, alpha=alpha, edgecolor='white', linewidth=0.6, boxstyle="Round,pad=0.2,rounding_size=3")
            ax2.add_patch(rectangle)
            if add_nucleus_cytoplasm_text:
                ax2.text(mus[i][0], mus[i][1] + rect_height / 2 if rads[-1] > 0.99 else mus[i][1] - rect_height / 2, 'Nucleus' if rads[0] > 0.99 else 'Cytoplasm', fontsize=24, color="#515151", ha='center', va='center', fontweight='bold', zorder = 9999)
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
    
def hide_axii_and_show_scale_bar(ax, show_scale_bar=True):
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
                    ['Nup2', 'Nup60', 'Nup1', 'Nup145', 'Nup49', 'Nup57', 'Nsp1', 'Nup100', 'Nup159', 'Nup116'] + \
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

def comparison_plot(base_tm_path, base_cluster_path, radii, n_sites, title, in_out_flow=None, ignored_nup_types=None, add_mini_titles=False, pie_scaling=15):
    n_samples = 2000
    fig_x = len(n_sites)
    fig_y = len(radii)
    
    # --- FIX 1: Adjust Figure Dimensions to Match Data Aspect Ratio ---
    # X range is 60 (-30 to 30), Y range is 80 (-40 to 40).
    # To get a 1:1 unit ratio without whitespace, the plot height must be 
    # ~1.33x the width (80/60).
    plot_width = 10
    plot_height = plot_width * (80 / 60)  # approx 13.33
    
    fig, axes = plt.subplots(fig_y, fig_x, 
                             figsize=(plot_width * fig_x, plot_height * fig_y), 
                             squeeze=False, 
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
    
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


    for j, r in enumerate(reversed(radii)):
        print("kda: ", radius_a_to_kda(r))
        for i, n in enumerate(n_sites):
            if add_mini_titles:
                axes[j, i].set_title(f'Radius: {r} nm, n_sites: {n}', fontsize=14)
            # set range of axes
            axes[j, i].set_xlim(-30, 30)
            axes[j, i].set_ylim(-40, 40)
            
            # This ensures 1 unit on x = 1 unit on y (isometric)
            axes[j, i].set_aspect('equal')
            
            # hide axii and show scale bar
            hide_axii_and_show_scale_bar(axes[j, i], show_scale_bar=False)
            
            tm_path = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n))
            cluster_path = base_cluster_path.replace("#r#", str(r)).replace("#n#", str(n))
            
            try:
                clusters, coordinate_edges, P, pca_cluster = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
            except Exception as e:
                print(f"Could not load data for radius {r} nm, n_sites {n}: {e}")
                continue
            
            good_mesostate_indices = list(range(len(clusters)))

            unprojected_mus = estimate_unprojected_mus(n_samples, clusters, coordinate_edges, good_mesostate_indices)
            good_mesostate_indices = pick_good_clusters_by_mu_angle(unprojected_mus, clusters, angle_threshold_degrees=90, angle_shift_degrees=0)
            mus, covs = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_mesostate_indices)
            
            visualize_arrows_between_mesostates(P, fig, axes[j, i], good_mesostate_indices, mus, show_colorbar_title=False, in_out_flow=in_out_flow, show_colorbar=False)
            visualize_pie_mesostates(clusters, P, axes[j, i], good_mesostate_indices, mus, PIE_COLORS, add_nucleus_cytoplasm_text=True, pie_scaling=pie_scaling)
            add_npc_scaffold_picture(axes[j, i])
            
    # Removed the 'big_ax' block (plot level axis arrows) entirely.