import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import importlib
import scipy
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from pygam import LinearGAM, s
import scipy.stats as st
from scipy.interpolate import pchip_interpolate


def plot_states_found_over_subset():
    # Double the size of all fonts and elements
    sns.set_theme(font_scale=1.5)

    sites_list = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    subsets = [2/60, 4/60, 8/60, 1/30, 2/30, 5/30, 10/30, 15/30, 20/30, 25/30, 1.0]

    data = []

    for radius in radii:
        for sites in sites_list:
            for subset in subsets:
                if subset in [2/60, 4/60, 8/60]:
                    path = f"data/ntr_variants_low_time_subsets/{sites}_{radius}_more/6_transition_matrices_subsets/{subset:.2f}fraction/0index/320clusters.pickle"
                else:
                    path = f"data/ntr_variants/{sites}_{radius}_more/6_transition_matrices_subsets/{subset:.2f}fraction/0index/320clusters.pickle"
                
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        tm = pickle.load(f)
                        data.append({
                            "radius": radius,
                            "sites": sites,
                            "subset": subset,
                            "states": tm.shape[0]
                        })

    df_states = pd.DataFrame(data)

    # Plotting
    fig = plt.figure(figsize=(10, 6))

    # Extract the colors we need from seaborn's tab10 palette
    colors = sns.color_palette("tab10", n_colors=len(sites_list))

    for i, sites in enumerate(sites_list):
        df_sub = df_states[df_states["sites"] == sites]
        
        # Calculate means and group the data
        grouped = df_sub.groupby("subset")["states"]
        x: np.ndarray = np.sort(list(grouped.groups.keys()))
        means: np.ndarray = np.array([grouped.get_group(k).mean() for k in x])
        
        ci_lower: list[float] = []
        ci_upper: list[float] = []
        
        # Calculate 95% Confidence Intervals manually
        for k in x:
            data_points: np.ndarray = grouped.get_group(k).values
            if len(data_points) > 1:
                interval = st.t.interval(
                    0.95, 
                    df=len(data_points)-1, 
                    loc=np.mean(data_points), 
                    scale=st.sem(data_points)
                )
                ci_lower.append(interval[0])
                ci_upper.append(interval[1])
            else:
                ci_lower.append(data_points[0])
                ci_upper.append(data_points[0])
                
        ci_lower_arr: np.ndarray = np.array(ci_lower)
        ci_upper_arr: np.ndarray = np.array(ci_upper)

        # Generate a denser set of x values for smooth interpolation
        x_smooth: np.ndarray = np.linspace(x.min(), x.max(), 300)

        # Use PCHIP interpolation to avoid unnatural wiggles/overshooting
        y_lower_smooth: np.ndarray = pchip_interpolate(x, ci_lower_arr, x_smooth)
        y_upper_smooth: np.ndarray = pchip_interpolate(x, ci_upper_arr, x_smooth)

        # Plot smoothed CI bounds
        plt.fill_between(x_smooth, y_lower_smooth, y_upper_smooth, alpha=0.2, color=colors[i])
        
        # Plot markers for the means (This matches the marker="o", lw=0 from seaborn)
        plt.plot(x, means, 'o', color=colors[i], label=f"{sites}")

    plt.title("")
    plt.xlabel("Subset size (fraction)")
    plt.ylabel("Number of unique states found")

    # Ensures the legend title is appropriately named
    plt.legend(title="Sites")
    plt.grid(True, alpha=0.3)
    plt.show()
    return fig


def plot_implied_timescales():
    n_top_eigenvalues = 5

    timescales = []
    for i in range(n_top_eigenvalues):
        timescales.append([])
        
    for window_size_steps in [5, 10, 25, 50, 100, 200]: 
        with open(f"data/ntr_variants_4_26_timescales/window_{window_size_steps}_steps/6_transition_matrices_subsets/1.00fraction_simulations/0index/160clusters.pickle", "rb") as f:
            tm = pickle.load(f)
            
        # 1. Compute the eigenvalues of the transition matrix
        eigenvalues = np.linalg.eigvals(tm)
        
        # 2. Sort the eigenvalues by absolute magnitude in descending order
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        # 3. Compute the implied timescales
        for i in range(n_top_eigenvalues):
            # We use i + 1 to skip the 0th eigenvalue (which is ~1.0 for the stationary distribution)
            lambda_i = sorted_eigenvalues[i + 1]
            
            # Catch numerical instability edge-cases (lambda should be strictly between 0 and 1)
            if lambda_i >= 1.0 or lambda_i <= 0.0:
                t_i = np.nan
            else:
                # Apply the formula: t_i = -tau / ln|lambda_i|
                t_i = -window_size_steps / np.log(lambda_i)
                
            timescales[i].append(t_i)
            
    # Plotting
    # The lag times used in your loop
    lag_times = [5, 10, 25, 50, 100, 200]

    fig = plt.figure(figsize=(8, 6))

    # Plot each of the computed timescales
    for i in range(n_top_eigenvalues):
        plt.plot(lag_times, timescales[i], marker='o', linewidth=2, label=f'Process {i+1} ($\lambda_{i+2}$)')

    # Plot the t = tau boundary
    plt.plot(lag_times, lag_times, color='black', linestyle='--', label='t = $\\tau$ boundary')
    plt.fill_between(lag_times, 0, lag_times, color='grey', alpha=0.2)

    # Standard ITS plots use a logarithmic y-axis
    plt.yscale('log')

    # Formatting
    plt.xlabel('Lag time $\\tau$ (steps)')
    plt.ylabel('Implied timescale $t_i$ (steps)')
    plt.title('Implied Timescales vs. Lag Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Display the plot
    plt.show()
    return fig


def plot_committor_vs_z(
    z_nuc: float,
    z_cyt: float
) -> plt.Figure:
    """
    Computes the committor probability to reach Cytoplasm before Nucleus
    dependent on the Z value of the state, and plots the results across
    3 subplots (one for each number of sites: 2, 4, 6) with custom diameter/radius colors.
    """
    def get_variant_color(
        diameter: int,
        radius: float,
        min_r: float,
        max_r: float,
    ) -> tuple[float, float, float]:
        diam_colors: dict[int, str] = {
            46: "#1f77b4",  # blue
            54: "#ff7f0e",  # orange
            62: "#2ca02c",  # green
            70: "#d62728"   # red
        }
        base_hex: str = diam_colors.get(diameter, "#7f7f7f")
        base_rgb: np.ndarray = np.array(mcolors.to_rgb(base_hex))
        frac: float = (radius - min_r) / (max_r - min_r)
        light_rgb: np.ndarray = 0.4 * base_rgb + 0.6 * np.array([1.0, 1.0, 1.0])
        dark_rgb: np.ndarray = 0.7 * base_rgb + 0.3 * np.array([0.0, 0.0, 0.0])
        interpolated_rgb: np.ndarray = light_rgb + frac * (dark_rgb - light_rgb)
        return (float(interpolated_rgb[0]), float(interpolated_rgb[1]), float(interpolated_rgb[2]))

    def compute_committor(tm: np.ndarray, z_vals: np.ndarray) -> np.ndarray:
        n_states: int = tm.shape[0]
        idx_cyt: np.ndarray = np.where(z_vals >= z_cyt)[0]
        idx_trans: np.ndarray = np.where((z_vals > z_nuc) & (z_vals < z_cyt))[0]
        
        q: np.ndarray = np.zeros(n_states)
        q[idx_cyt] = 1.0
        
        if len(idx_trans) > 0:
            I_C: np.ndarray = np.eye(len(idx_trans))
            P_C: np.ndarray = tm[np.ix_(idx_trans, idx_trans)]
            b: np.ndarray = np.sum(tm[np.ix_(idx_trans, idx_cyt)], axis=1)
            try:
                q_trans: np.ndarray = np.linalg.solve(I_C - P_C, b)
                q[idx_trans] = q_trans
            except np.linalg.LinAlgError:
                pass
        return q

    # Apply style ticks & no background color
    sns.set_theme(style="ticks", font_scale=1.5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.patch.set_facecolor('none')

    sites_list: list[int] = [2, 4, 6]
    radii: list[int] = [10, 14, 18, 22, 26]
    configs: list[tuple[str, int, str]] = [
        ("ntr_variants_46R", 46, ""),
        ("ntr_variants", 54, "_more"),
        ("ntr_variants_62R", 62, ""),
        ("ntr_variants_70R", 70, ""),
    ]

    plotted_diams_per_ax: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}

    for ax_idx, sites in enumerate(sites_list):
        ax: plt.Axes = axes[ax_idx]
        ax.set_title(f"{sites} Sites", fontsize=16, fontweight="bold")
        ax.set_xlabel("Z Coordinate (Å)", fontsize=14)
        if ax_idx == 0:
            ax.set_ylabel("Committor to Cytoplasm", fontsize=14)
        
        # Style clean background
        sns.despine(ax=ax)
        ax.set_facecolor('none')
        
        for dir_name, diameter, suffix in configs:
            for radius in radii:
                fraction_dir: str = "1.00fraction_simulations" if diameter == 54 else "1.00fraction"
                tm_path: str = f"data/{dir_name}/{sites}_{radius}{suffix}/6_transition_matrices_subsets/{fraction_dir}/0index/320clusters.pickle"
                clustering_path: str = f"data/{dir_name}/{sites}_{radius}{suffix}/5_clustering_subsets/{fraction_dir}/0index/320clusters.pickle"
                
                if not os.path.exists(tm_path) or not os.path.exists(clustering_path):
                    continue
                    
                with open(tm_path, "rb") as f:
                    tm: np.ndarray = pickle.load(f)
                with open(clustering_path, "rb") as f:
                    clustering: Any = pickle.load(f)
                    
                z_vals: np.ndarray = clustering.centroids_z_actual.flatten()
                q: np.ndarray = compute_committor(tm, z_vals)
                
                color: tuple[float, float, float] = get_variant_color(
                    diameter=diameter,
                    radius=float(radius),
                    min_r=10.0,
                    max_r=26.0,
                )
                
                sort_idx: np.ndarray = np.argsort(z_vals)
                
                label: str | None = f"{diameter} nm" if diameter not in plotted_diams_per_ax[ax_idx] else None
                plotted_diams_per_ax[ax_idx].add(diameter)
                
                ax.plot(z_vals[sort_idx], q[sort_idx], color=color, linewidth=2.5, linestyle="-", label=label)

        # Vertical boundary lines
        ax.axvline(x=z_nuc, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
        ax.axvline(x=z_cyt, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
        
        # Sort legend handles to keep them consistent (46, 54, 62, 70)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            sorted_pairs = sorted(zip(handles, labels), key=lambda x: int(x[1].split()[0]))
            shandles, slabels = zip(*sorted_pairs)
            ax.legend(shandles, slabels, title="NPC Diameter", loc="best", frameon=False)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig