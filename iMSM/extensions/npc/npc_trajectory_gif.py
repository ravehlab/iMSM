import sys
import os
# Ensure the root of NPC-markov is in the Python path so iMSM can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from iMSM.extensions.npc.npc_graph_figure import (
    load_cluster_data, 
    estimate_unprojected_mus, 
    pick_good_clusters_by_mu_angle, 
    estimate_cluters_mu_cov,
    visualize_arrows_between_mesostates,
    visualize_pie_mesostates,
    add_npc_scaffold_picture,
    hide_axii_and_show_scale_bar,
    PIE_COLORS,
    PIE_COLORS_DICT
)

def generate_trajectory_gif(
    base_tm_path: str, 
    base_cluster_path: str, 
    radius: float, 
    n_sites: int, 
    output_gif_path: str,
    trajectory_length: int = 500,
    frames: int = 500,
    interval: int = 100,
    title: str = None,
    show_scale_bars: bool = True,
    pie_scaling: float = 15, 
    min_rate: float = 0.01, 
    max_rate: float = 5, 
    time_step_us: float = 5,
    ignored_nup_types: list = None
):
    """
    Simulates a long trajectory on the transition network and outputs it as a GIF.
    The graph figure looks exactly like one panel in comparison_plot.
    """
    # 1. Load Data
    tm_path = base_tm_path.replace("#r#", str(radius)).replace("#n#", str(n_sites))
    cluster_path = base_cluster_path.replace("#r#", str(radius)).replace("#n#", str(n_sites))
    
    n_samples = 2000
    try:
        clusters, coordinate_edges, P, pca_cluster = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
    except Exception as e:
        print(f"Could not load data for radius {radius} nm, n_sites {n_sites}: {e}")
        return

    # Find valid states for drawing the static background (like comparison_plot)
    all_cluster_indices = list(range(len(clusters)))
    unprojected_mus = estimate_unprojected_mus(n_samples, clusters, coordinate_edges, all_cluster_indices)
    good_mesostate_indices = pick_good_clusters_by_mu_angle(unprojected_mus, clusters, angle_threshold_degrees=92, angle_shift_degrees=0)
    
    # We estimate MUs for ALL states so we can traverse the full markov chain
    all_mus, covs = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, all_cluster_indices)
    mus = all_mus[good_mesostate_indices] # For the background drawing, which expects mus indexed relative to good_mesostate_indices

    # 2. Setup Figure background
    fig, ax = plt.subplots(figsize=(10, 10 * (80 / 60)))
    if title:
        ax.set_title(title, fontsize=20, y=0.95)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    hide_axii_and_show_scale_bar(ax, show_scale_bar=show_scale_bars, show_scale_text=False)

    pie_colors = PIE_COLORS
    if ignored_nup_types is not None:
        pie_colors = PIE_COLORS.copy()
        for nup_type in ignored_nup_types:
            if nup_type in PIE_COLORS_DICT:
                color_to_remove = PIE_COLORS_DICT[nup_type]
                if color_to_remove in pie_colors:
                    pie_colors.remove(color_to_remove)

    # Draw the background exactly like the latest comparison_plot
    visualize_arrows_between_mesostates(P, fig, ax, good_mesostate_indices, mus, show_colorbar_title=True, in_out_flow=None, show_colorbar=True, min_rate=min_rate, max_rate=max_rate, time_step_us=time_step_us)
    visualize_pie_mesostates(clusters, P, ax, good_mesostate_indices, mus, pie_colors, add_nucleus_cytoplasm_text=True, pie_scaling=pie_scaling)
    add_npc_scaffold_picture(ax)

    # 3. Simulate Markov Chain Trajectory
    n_states = P.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stat_dist = np.real(eigenvectors[:, idx])
    stat_dist = np.abs(stat_dist)
    stat_dist /= stat_dist.sum()

    cum_tm = np.cumsum(P, axis=1)
    cum_tm[:, -1] = 1.0

    state = int(np.random.choice(n_states, p=stat_dist))
    traj_states = [state]
    for _ in range(trajectory_length):
        state = int(np.searchsorted(cum_tm[state], np.random.random()))
        traj_states.append(state)

    # 4. Set up animation elements
    # Marker for the current position
    point_marker, = ax.plot([], [], 'o', color='red', markersize=15, markeredgecolor='white', markeredgewidth=2, zorder=9999)
    # Line for the tail (history)
    tail_line, = ax.plot([], [], '-', color='darkred', linewidth=3, alpha=0.6, zorder=9998)
    
    # We keep a history tail to make it look nicer
    tail_length = 20

    # 5. Create and Save animation
    print("Pre-computing and generating frames rapidly...")
    traj_mus = np.array([all_mus[s] for s in traj_states])
    
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    width, height = fig.canvas.get_width_height()
    
    gif_frames = []
    
    for frame in range(frames):
        fig.canvas.restore_region(bg)
        
        progress = (float(frame) / frames) * float(trajectory_length)
        idx_floor = int(np.floor(progress))
        idx_ceil = min(idx_floor + 1, len(traj_states) - 1)
        sub_progress = progress - idx_floor

        current_x = traj_mus[idx_floor, 0] + (traj_mus[idx_ceil, 0] - traj_mus[idx_floor, 0]) * sub_progress
        current_y = traj_mus[idx_floor, 1] + (traj_mus[idx_ceil, 1] - traj_mus[idx_floor, 1]) * sub_progress

        point_marker.set_data([current_x], [current_y])

        tx = []
        ty = []
        for i in range(1, tail_length + 1):
            history_progress = max(0, progress - (i * (float(trajectory_length) / frames) * 2))
            h_floor = int(np.floor(history_progress))
            h_ceil = min(h_floor + 1, len(traj_states) - 1)
            h_sub = history_progress - h_floor
            
            x = traj_mus[h_floor, 0] + (traj_mus[h_ceil, 0] - traj_mus[h_floor, 0]) * h_sub
            y = traj_mus[h_floor, 1] + (traj_mus[h_ceil, 1] - traj_mus[h_floor, 1]) * h_sub
            tx.append(x)
            ty.append(y)
            
        tx.reverse()
        ty.reverse()
        tx.append(current_x)
        ty.append(current_y)
        
        tail_line.set_data(tx, ty)
        
        ax.draw_artist(tail_line)
        ax.draw_artist(point_marker)
        fig.canvas.blit(fig.bbox)
        
        try:
            buf = fig.canvas.buffer_rgba()
            img = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'RGBA', 0, 1)
        except AttributeError:
            buf = fig.canvas.tostring_argb()
            img = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'ARGB', 0, 1)
            
        gif_frames.append(img.copy())

    print(f"Saving animation to {output_gif_path}...")
    if gif_frames:
        gif_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=interval,
            loop=0
        )
    print("Done!")
    plt.close(fig)

if __name__ == "__main__":
    # Example usage:
    generate_trajectory_gif(
        base_tm_path="/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/ntr_variants_46R/#n#_#r#/6_transition_matrices_subsets/1.00fraction/0index/160clusters.pickle",
        base_cluster_path="/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/ntr_variants_46R/#n#_#r#/5_clustering_subsets/1.00fraction/0index/160clusters.pickle",
        radius=26,
        n_sites=6,
        output_gif_path="trajectory.gif"
    )
    pass
