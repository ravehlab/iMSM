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