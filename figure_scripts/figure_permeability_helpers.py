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

import importlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import deeptime
import scipy
import pandas as pd
import os
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from pygam import LinearGAM, s
import scipy.stats as st
from scipy.interpolate import pchip_interpolate
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker 


import iMSM.extensions.npc.npc
importlib.reload(iMSM.extensions.npc.npc)
from iMSM.extensions.npc.npc import run
# --
import raveh_2025.show_transport_stats_v3
importlib.reload(raveh_2025.show_transport_stats_v3)
from raveh_2025.show_transport_stats_v3 import read_df, calc_filtered_transport_stats
# --
import iMSM.extensions.npc.npc_utils
importlib.reload(iMSM.extensions.npc.npc_utils)
from iMSM.extensions.npc.npc_utils import radius_a_to_kda, amount_to_concentration


# convergence stuff params

radii = [10, 14, 18, 22, 26]
kdas = [radius_a_to_kda(r) for r in radii]
sites_list = [2, 4, 6]
n_clusters = 320
# subsets = [0.04, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1]
subsets = [2/60, 4/60, 8/60, 10/60, 15/60, 20/60, 25/60, 30/60, 40/60, 50/60, 60/60]
total_simulation_times = [int(60 * subset) * 30 for subset in subsets]

sims_subsets = [1/30, 2 / 30, 5 / 30, 10 / 30, 15 / 30, 20 / 30, 25 / 30, 1.0]
sims_total_simulation_times = [subset * 30  * 60 for subset in sims_subsets]


STYLE = {
    'md_time':  {'color': '#c0702f', 'marker': 'o', 'ms': 6, 'lw': 4.0, 'alpha': 0.6, 'zorder': 2},
    'msm_time': {'color': '#2f9acf', 'marker': 'o', 'ms': 6, 'lw': 4.0, 'alpha': 0.6, 'zorder': 4},
    'md_sim':   {'color': '#c0702f', 'marker': 'o', 'ms': 6, 'lw': 4.0, 'alpha': 0.6, 'zorder': 2},
    'msm_sim':  {'color': '#2f9acf', 'marker': 'o', 'ms': 6, 'lw': 4.0, 'alpha': 0.6, 'zorder': 4},
}

STAR_Y_PERM = 10**-0.8

def get_gam_curve(x, y, n_points=100):
    """Fits a GAM in log10 space and returns a smooth curve."""
    valid = (x > 0) & (y > 0) & ~np.isnan(x) & ~np.isnan(y)
    x_val, y_val = x[valid], y[valid]
    
    if len(x_val) < 4: 
        return x_val, y_val
        
    x_log = np.log10(x_val).reshape(-1, 1)
    y_log = np.log10(y_val)
    
    n_splines = min(10, len(x_val))
    gam = LinearGAM(s(0, n_splines=n_splines)).gridsearch(x_log, y_log, progress=False)
    
    x_grid = np.linspace(x_log.min(), x_log.max(), n_points).reshape(-1, 1)
    return 10**x_grid.flatten(), 10**gam.predict(x_grid)

def plot_target_line(ax, x, target_val, lo_val=None, hi_val=None):
    """Plots the horizontal target regression line and optional confidence bands."""
    line, = ax.plot(x, [target_val] * len(x), color='red', linestyle='--', alpha=0.8, zorder=1, label='Fitted Target', lw=3.0)
    if lo_val is not None and hi_val is not None:
        ax.fill_between(x, lo_val, hi_val, color='red', alpha=0.15, zorder=1)
    return line

def plot_series(ax, x, y, style, star_y, y_lower=None, y_upper=None, gam_ls='-'):
    """Plots the scatter data, GAM curve, zero-value stars, and optional error bars."""
    valid = y > 0
    y_plot = np.where(valid, y, np.nan)
    
    # Scatter
    ax.plot(x, y_plot, marker=style['marker'], markersize=style['ms'], color=style['color'], 
            linestyle='None', alpha=style['alpha'], zorder=style['zorder'])
    
    # GAM Line
    x_gam, y_gam = get_gam_curve(x, y_plot)
    ax.plot(x_gam, y_gam, color=style['color'], linestyle=gam_ls, linewidth=style['lw'], 
            alpha=style['alpha'], zorder=style['zorder'], label=style.get('label', None))
    
    # Stars for zeros
    zeros = ~valid
    if np.any(zeros):
        ax.plot(x[zeros], [star_y] * np.sum(zeros), marker='*', markersize=12, 
                color=style['color'], linestyle='None', zorder=style['zorder'] + 1)
        
    # Error bars (only plotted for valid > 0 points)
    if y_lower is not None and y_upper is not None:
        y_err = [(y - y_lower)[valid], (y_upper - y)[valid]]
        ax.errorbar(x[valid], y[valid], yerr=y_err, fmt='none', ecolor=style['color'], 
                    alpha=0.9, elinewidth=2, capsize=3, zorder=style['zorder'])

def get_ratio(y_num, y_den):
    """Calculates ratio where both arrays are strictly positive, returning NaNs otherwise."""
    ratio = np.full_like(y_num, np.nan, dtype=float)
    valid = (y_num > 0) & (y_den > 0)
    ratio[valid] = y_num[valid] / y_den[valid]
    return ratio

def save_convergence_data():
    # save convergence data
    sites_to_fitted_perms = {}
    sites_to_fitted_perms_lo = {}
    sites_to_fitted_perms_hi = {}

    for sites in sites_list:
        sites_perms = []
        sites_perms_lower = []
        sites_perms_upper = []
        sites_errs = []
        for radius in radii:
            kap_coords_folder = f"data/ntr_variants/{sites}_{radius}_more/1_single_sim_kap_coords/"
            results = collect_perms_md_single_type(kap_coords_folder, n_sims=30, start_time=10, end_time=70, shorten_time_to_us=None)
            perms = results[2]
            perms_lower = results[3]
            perms_upper = results[4]
            y_err_lower = perms - perms_lower
            y_err_upper = perms_upper - perms
            y_err = (y_err_lower + y_err_upper) / 2
            sites_perms.append(perms)
            sites_perms_lower.append(perms_lower)
            sites_perms_upper.append(perms_upper)
            sites_errs.append(y_err)
        sites_perms = np.array(sites_perms)
        sites_perms_lower = np.array(sites_perms_lower)
        sites_perms_upper = np.array(sites_perms_upper)
        sites_errs = np.array(sites_errs)
        _, _, slope, intercept, y_fit = weighted_power_law_fit(kdas, sites_perms, y_err=None) #sites_errs)
        _, _, _, _, y_fit_lo = weighted_power_law_fit(kdas, sites_perms_lower, y_err=None) #sites_errs)
        _, _, _, _, y_fit_hi = weighted_power_law_fit(kdas, sites_perms_upper, y_err=None) #sites_errs)
        sites_to_fitted_perms[sites] = y_fit
        sites_to_fitted_perms_lo[sites] = y_fit_lo
        sites_to_fitted_perms_hi[sites] = y_fit_hi

    perms_all_md = np.zeros((len(radii), len(sites_list), len(subsets)))
    lower_perms_all_md = np.zeros((len(radii), len(sites_list), len(subsets)))
    upper_perms_all_md = np.zeros((len(radii), len(sites_list), len(subsets)))

    perms_all_md_sim_subset = np.zeros((len(radii), len(sites_list), len(sims_subsets)))
    lower_perms_all_md_sim_subset = np.zeros((len(radii), len(sites_list), len(sims_subsets)))
    upper_perms_all_md_sim_subset = np.zeros((len(radii), len(sites_list), len(sims_subsets)))

    perms_all_msm = np.zeros((len(radii), len(sites_list), len(subsets)))
    perms_all_msm_with_init_time = np.ones((len(radii), len(sites_list), len(subsets))) * -1  # Set default to -1 so we can easily filter them out if needed
    perms_all_msm_sim_subset = np.zeros((len(radii), len(sites_list), len(sims_subsets)))



    for i, radius in enumerate(radii):
        for j, sites in enumerate(sites_list):

            # Collect MSM permeabilities (time subset)
            for conf_name, target_arr in [
                ("ntr_variants", perms_all_msm), 
                ("ntr_variants_with_init_time", perms_all_msm_with_init_time)
            ]:
                if conf_name == "ntr_variants_with_init_time" and sites == 6:
                    continue # Skip 6 sites variant for init time
                
                low_time_folder = "ntr_variants_low_time_subsets" if conf_name == "ntr_variants" else "ntr_variants_with_init_time_low_time_subsets"
                for k, subset in enumerate(subsets):
                    path = f"data/{conf_name}/{sites}_{radius}_more/7_permeabilities_subsets/{subset:.2f}fraction/0index/7_permeabilities.pickle"
                    if subset in [2/60, 4/60, 8/60]:
                        path = f"data/{low_time_folder}/{sites}_{radius}_more/7_permeabilities_subsets/{subset:.2f}fraction/0index/7_permeabilities.pickle"
                    with open(path, "rb") as f:
                        perms = pickle.load(f)
                        target_arr[i, j, k] = perms[n_clusters]
                    
            # Collect MSM permeabilities (sim subset)
            for k, subset in enumerate(sims_subsets):
                with open(f"data/ntr_variants/{sites}_{radius}_more/7_permeabilities_subsets/{subset:.2f}fraction_simulations/{0}index/7_permeabilities.pickle", "rb") as f:
                    perms = pickle.load(f)
                    perms_all_msm_sim_subset[i, j, k] = perms[n_clusters]
            
            # Collect MD permeabilities (time subset)
            for k, subset in enumerate(subsets):
                kap_coords_folder = f"data/ntr_variants/{sites}_{radius}_more/1_single_sim_kap_coords/"
                results = collect_perms_md_single_type(kap_coords_folder, n_sims=30, start_time=10, end_time=70, shorten_time_to_us=int(subset * 60))
                perms_all_md[i, j, k] = results[2]
                lower_perms_all_md[i, j, k] = results[3]
                upper_perms_all_md[i, j, k] = results[4]
                
            # Collect MD permeabilities (sim subset)
            for k, subset in enumerate(sims_subsets):
                kap_coords_folder = f"data/ntr_variants/{sites}_{radius}_more/1_single_sim_kap_coords/"
                results = collect_perms_md_single_type(kap_coords_folder, n_sims=int(30*subset), start_time=10, end_time=70, shorten_time_to_us=None)
                perms_all_md_sim_subset[i, j, k] = results[2]
                lower_perms_all_md_sim_subset[i, j, k] = results[3]
                upper_perms_all_md_sim_subset[i, j, k] = results[4]
        
    # save convergence data
    # md time subset
    with open("data/convergence/perms_all_md.pickle", "wb") as f:
        pickle.dump(perms_all_md, f)
    with open("data/convergence/lower_perms_all_md.pickle", "wb") as f:
        pickle.dump(lower_perms_all_md, f)
    with open("data/convergence/upper_perms_all_md.pickle", "wb") as f:
        pickle.dump(upper_perms_all_md, f)
        
    # md sim subset
    with open("data/convergence/perms_all_md_sim_subset.pickle", "wb") as f:
        pickle.dump(perms_all_md_sim_subset, f)
    with open("data/convergence/lower_perms_all_md_sim_subset.pickle", "wb") as f:
        pickle.dump(lower_perms_all_md_sim_subset, f)
    with open("data/convergence/upper_perms_all_md_sim_subset.pickle", "wb") as f:
        pickle.dump(upper_perms_all_md_sim_subset, f)

    # msm time subset
    with open("data/convergence/perms_all_msm.pickle", "wb") as f:
        pickle.dump(perms_all_msm, f)

    with open("data/convergence/perms_all_msm_with_init_time.pickle", "wb") as f:
        pickle.dump(perms_all_msm_with_init_time, f)

    # msm sim subset
    with open("data/convergence/perms_all_msm_sim_subset.pickle", "wb") as f:
        pickle.dump(perms_all_msm_sim_subset, f)

    # ideal fits
    with open("data/convergence/sites_to_fitted_perms.pickle", "wb") as f:
        pickle.dump(sites_to_fitted_perms, f)
    with open("data/convergence/sites_to_fitted_perms_lo.pickle", "wb") as f:
        pickle.dump(sites_to_fitted_perms_lo, f)
    with open("data/convergence/sites_to_fitted_perms_hi.pickle", "wb") as f:
        pickle.dump(sites_to_fitted_perms_hi, f)

def load_convergence_data():
    # load convergence data
    # md time subset
    with open("data/convergence/perms_all_md.pickle", "rb") as f:
        perms_all_md = pickle.load(f)
    with open("data/convergence/lower_perms_all_md.pickle", "rb") as f:
        lower_perms_all_md = pickle.load(f)
    with open("data/convergence/upper_perms_all_md.pickle", "rb") as f:
        upper_perms_all_md = pickle.load(f)
        
    # md sim subset
    with open("data/convergence/perms_all_md_sim_subset.pickle", "rb") as f:
        perms_all_md_sim_subset = pickle.load(f)
    with open("data/convergence/lower_perms_all_md_sim_subset.pickle", "rb") as f:
        lower_perms_all_md_sim_subset = pickle.load(f)
    with open("data/convergence/upper_perms_all_md_sim_subset.pickle", "rb") as f:
        upper_perms_all_md_sim_subset = pickle.load(f)
        
    # msm time subset
    with open("data/convergence/perms_all_msm.pickle", "rb") as f:
        perms_all_msm = pickle.load(f)
        
    with open("data/convergence/perms_all_msm_with_init_time.pickle", "rb") as f:
        perms_all_msm_with_init_time = pickle.load(f)
        
    # msm sim subset
    with open("data/convergence/perms_all_msm_sim_subset.pickle", "rb") as f:
        perms_all_msm_sim_subset = pickle.load(f)

    # ideal fits
    with open("data/convergence/sites_to_fitted_perms.pickle", "rb") as f:
        sites_to_fitted_perms = pickle.load(f)
    with open("data/convergence/sites_to_fitted_perms_lo.pickle", "rb") as f:
        sites_to_fitted_perms_lo = pickle.load(f)
    with open("data/convergence/sites_to_fitted_perms_hi.pickle", "rb") as f:
        sites_to_fitted_perms_hi = pickle.load(f)
        
    return perms_all_md, lower_perms_all_md, upper_perms_all_md, perms_all_md_sim_subset, lower_perms_all_md_sim_subset, upper_perms_all_md_sim_subset, perms_all_msm, perms_all_msm_with_init_time, perms_all_msm_sim_subset, sites_to_fitted_perms, sites_to_fitted_perms_lo, sites_to_fitted_perms_hi


def plot_4_6_cmp():

    perms_all_md, lower_perms_all_md, upper_perms_all_md, perms_all_md_sim_subset, lower_perms_all_md_sim_subset, upper_perms_all_md_sim_subset, perms_all_msm, perms_all_msm_with_init_time, perms_all_msm_sim_subset, sites_to_fitted_perms, sites_to_fitted_perms_lo, sites_to_fitted_perms_hi = load_convergence_data()

    # ==========================================
    # CONFIGURATION & MAGIC NUMBERS
    # ==========================================
    RADIUS_TARGET = 26
    SITE_A = 4
    SITE_B = 6

    STAR_Y_RATIO = 0.0  # Changed for linear scale

    AXIS_LIMITS = {
        'perm': (10**-0.5, 10**4),
        'ratio': (-0.4, 0.4)  
    }

    # ==========================================
    # SETUP & DATA EXTRACTION
    # ==========================================

    r_idx = list(radii).index(RADIUS_TARGET)
    s_a_idx = list(sites_list).index(SITE_A)
    s_b_idx = list(sites_list).index(SITE_B)

    kda_val = kdas[r_idx]
    target_ratio = sites_to_fitted_perms[SITE_A][r_idx] / sites_to_fitted_perms[SITE_B][r_idx]

    # Define data subsets for the two rows to avoid massive blocks of duplicated code
    subsets = [
        {
            'title': 'Time Subset',
            'x': np.array(total_simulation_times),
            'md': perms_all_md,
            'md_lo': lower_perms_all_md,
            'md_hi': upper_perms_all_md,
            'msm': perms_all_msm,
            'style_md': STYLE['md_time'],
            'style_msm': STYLE['msm_time'],
            'ratio_ls': '-'
        },
        {
            'title': 'Sim Subset',
            'x': np.array(sims_total_simulation_times),
            'md': perms_all_md_sim_subset,
            'md_lo': lower_perms_all_md_sim_subset,
            'md_hi': upper_perms_all_md_sim_subset,
            'msm': perms_all_msm_sim_subset,
            'style_md': STYLE['md_sim'],
            'style_msm': STYLE['msm_sim'],
            'ratio_ls': '-'
        }
    ]

    # ==========================================
    # MAIN PLOTTING LOOP
    # ==========================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    targ_line = None  # Reference for the legend

    for i, sub in enumerate(subsets):
        x = sub['x']
        
        # Row title on the left (20% larger: 22 -> 26, not bold)
        axes[i, 0].annotate(sub['title'], xy=(-0.25, 0.5), xycoords='axes fraction', 
                           fontsize=26, va='center', ha='right', rotation=90)
        
        # Data extraction for this specific row
        md_a, msm_a = sub['md'][r_idx, s_a_idx, :], sub['msm'][r_idx, s_a_idx, :]
        md_b, msm_b = sub['md'][r_idx, s_b_idx, :], sub['msm'][r_idx, s_b_idx, :]
        
        # --- Panel 1: Site A ---
        ax = axes[i, 0]
        targ_line = plot_target_line(ax, x, sites_to_fitted_perms[SITE_A][r_idx], 
                                    sites_to_fitted_perms_lo[SITE_A][r_idx], sites_to_fitted_perms_hi[SITE_A][r_idx])
        plot_series(ax, x, md_a, sub['style_md'], 1, sub['md_lo'][r_idx, s_a_idx, :], sub['md_hi'][r_idx, s_a_idx, :])
        plot_series(ax, x, msm_a, sub['style_msm'], 1)
        
        # --- Panel 2: Site B ---
        ax = axes[i, 1]
        plot_target_line(ax, x, sites_to_fitted_perms[SITE_B][r_idx], 
                        sites_to_fitted_perms_lo[SITE_B][r_idx], sites_to_fitted_perms_hi[SITE_B][r_idx])
        plot_series(ax, x, md_b, sub['style_md'], 1, sub['md_lo'][r_idx, s_b_idx, :], sub['md_hi'][r_idx, s_b_idx, :])
        plot_series(ax, x, msm_b, sub['style_msm'], 1)
        
        # --- Panel 3: Ratio (A / B) ---
        ax = axes[i, 2]
        plot_target_line(ax, x, target_ratio)
        plot_series(ax, x, get_ratio(md_a, md_b), sub['style_md'], STAR_Y_RATIO, gam_ls=sub['ratio_ls'])
        plot_series(ax, x, get_ratio(msm_a, msm_b), sub['style_msm'], STAR_Y_RATIO, gam_ls=sub['ratio_ls'])
        
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel('Ratio', fontsize=20)  # 17 -> 20

    # Big column titles on top (22 -> 26, not bold)
    axes[0, 0].set_title(f'{SITE_A} Sites', fontsize=26, pad=12)
    axes[0, 1].set_title(f'{SITE_B} Sites', fontsize=26, pad=12)
    axes[0, 2].set_title(f'Ratio {SITE_A}/{SITE_B}', fontsize=26, pad=12)

    # ==========================================
    # AESTHETICS & LAYOUT
    # ==========================================

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=19)  # 16 -> 19
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            
            # Apply conditional formatting depending on if it's a Ratio panel (j==2)
            if j == 2:
                ax.set_yscale('linear')
                ax.set_ylim(AXIS_LIMITS['ratio'])
                # Format the y-axis to show percentages
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            else:
                ax.set_yscale('log')
                ax.set_ylim(AXIS_LIMITS['perm'])
                
            ax.set_xlim(subsets[i]['x'][0]*0.9, subsets[i]['x'][-1]*1.1)
                
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Common text and Unified Legend (20 -> 24, 16 -> 19)
    fig.text(0.5, 0.02, 'Accumulated Simulation Time (μs)', ha='center', fontsize=24)
    fig.text(0.02, 0.5, 'Permeability (1/s/µM/NPC)', va='center', rotation='vertical', fontsize=24)
    fig.suptitle(f'Convergence & Ratio Analysis for {RADIUS_TARGET}Å Radius (Sites {SITE_A} vs {SITE_B})\n(Red band indicates fitted power law regressor leeway)', 
                fontsize=19, y=0.98)

    targ_line.set_label('Fitted')
    custom_lines = [
        Line2D([0], [0], color=STYLE['msm_time']['color'], marker=STYLE['msm_time']['marker'], linestyle='-', linewidth=2, label='iMSM'),
        Line2D([0], [0], color=STYLE['md_time']['color'], marker=STYLE['md_time']['marker'], linestyle='-', linewidth=1.5, alpha=0.8, label='MD'),
        targ_line
    ]

    fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False, fontsize=22)  # 18 -> 22

    plt.tight_layout(rect=[0.06, 0.04, 1, 0.90])
    plt.show()
    return fig



# Defines helper functions to collect empirical and modeled permeabilities, 
# confidence intervals, and plot power-law fits




def poisson_confidence_interval(count, alpha=0.05):
    lower = scipy.stats.chi2.ppf(alpha/2, 2*count) / 2
    upper = scipy.stats.chi2.ppf(1-alpha/2, 2*(count+1)) / 2
    
    return lower, upper

# Collect data
def collect_perms_md():
    n_sites = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    kdas = [radius_a_to_kda(r) for r in radii]
    results = []
    for sites in n_sites:
        for radius in radii:
            kap_coords_folder = f"data/ntr_variants/{sites}_{radius}_more/1_single_sim_kap_coords/"
            n_trajs, total_full_transports, permeability, lower_permeability, upper_permeability = collect_perms_md_single_type(kap_coords_folder, n_sims=30, start_time=10, end_time=70, shorten_time_to_us=None)
            
            results.append({
                'sites': sites,
                'radius': radius,
                'kda': radius_a_to_kda(radius),
                'total_transports': total_full_transports,
                'n_trajs': n_trajs,
                'permeability': permeability,
                'lower_permeability': lower_permeability if lower_permeability >=0 else 0,
                'upper_permeability': upper_permeability if upper_permeability >=0 else 0
            })
    return results

def collect_perms_md_single_type(kap_coords_folder, n_sims, start_time, end_time, shorten_time_to_us, step_ns=100):
    """
    Count transport events from raw 3D kap coordinates.
    
    kap_coords_folder: path containing subfolders 1, 2, ..., n_sims,
                       each with a file 'start_time-end_time.pickle' of shape (kap_amount, 3, n_frames).
    step_ns: time per frame in nanoseconds (default 100 ns).
    """
    # Load and concatenate raw 3D coordinates across all simulations
    all_coords = []
    for sim_idx in range(1, n_sims + 1):
        path = os.path.join(kap_coords_folder, str(sim_idx), f"{start_time}-{end_time}.pickle")
        with open(path, "rb") as f:
            coords = pickle.load(f)  # shape: (kap_amount, 3, n_frames)
        all_coords.append(coords)

    all_coords = np.concatenate(all_coords, axis=0)  # shape: (kap_amount * n_sims, 3, n_frames)

    _, _, n_frames = all_coords.shape
    time_us = n_frames * step_ns / 1000  # convert ns → µs

    if shorten_time_to_us is not None:
        if shorten_time_to_us > time_us:
            raise ValueError(f"shorten_time_to {shorten_time_to_us} µs is greater than trajectory length {time_us} µs")
        n_frames_to_use = int(shorten_time_to_us * 1000 / step_ns)
        time_us = shorten_time_to_us
        all_coords = all_coords[:, :, :n_frames_to_use]

    # Count transport events across all kap trajectories
    # z-axis is index 2 in the (kap_amount, 3, n_frames) array
    total_full_transports = 0
    n_trajs = all_coords.shape[0]
    for i in range(n_trajs):
        z = all_coords[i, 2, :]  # shape: (n_frames,)
        last_side = None

        for z_val in z:
            if z_val >= 10:
                current_side = 1   # cytoplasmic side (top)
            elif z_val <= -10:
                current_side = 0   # nuclear side (bottom)
            else:
                continue           # inside the channel, not yet committed

            if last_side is None:
                last_side = current_side   # initialise, don't count
            elif last_side != current_side:
                total_full_transports += 1
                last_side = current_side

    concentration_M = amount_to_concentration(100, box_side_a=800)
    concentration_uM = concentration_M * 1e6
            
    time_s = time_us * 1e-6

    lower, upper = poisson_confidence_interval(total_full_transports)
    rate = total_full_transports / (time_s * n_sims)
    lower_rate = lower / (time_s * n_sims)
    upper_rate = upper / (time_s * n_sims)
    permeability = rate / concentration_uM
    lower_permeability = lower_rate / concentration_uM
    upper_permeability = upper_rate / concentration_uM
    return n_trajs, total_full_transports, permeability, lower_permeability, upper_permeability

def collect_perms_msm_bootstrap(n_clusters=640):
    n_sites = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    kdas = [radius_a_to_kda(r) for r in radii]
    results = []
    for sites in n_sites:
        for radius in radii:
            with open(f"data/ntr_variants/{sites}_{radius}/7_permeabilities_bootstrap.pickle", "rb") as f:
                bootstrapped_perms = pickle.load(f)
            n_bootstraps = len(bootstrapped_perms)
            perms = []
            for b in range(n_bootstraps):
                perms.append(bootstrapped_perms[b][n_clusters])
            perms = np.array(perms)
            meadian_perm = np.median(perms)
            lower = np.percentile(perms, 5)
            upper = np.percentile(perms, 95)
            results.append({
                'sites': sites,
                'radius': radius,
                'kda': radius_a_to_kda(radius),
                'median_permeability': meadian_perm,
                'lower_permeability': lower,
                'upper_permeability': upper
            })
    return results

def collect_perms_msm(n_clusters=160):
    n_sites = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    kdas = [radius_a_to_kda(r) for r in radii]
    results = []
    for sites in n_sites:
        for radius in radii:
            with open(f"data/ntr_variants/{sites}_{radius}_more/7_permeabilities_subsets/1.00fraction_simulations/0index/7_permeabilities.pickle", "rb") as f:
                perms_dict = pickle.load(f)
            permeability = perms_dict[n_clusters]
            results.append({
                'sites': sites,
                'radius': radius,
                'kda': radius_a_to_kda(radius),
                'permeability': permeability
            })
    return results

def weighted_power_law_fit(x, y, y_err=None):
    weights = None
    if y_err is not None:
        log_y_err = y_err / y
        weights = 1 / (log_y_err ** 2)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = weights / np.mean(weights)
        weights = np.clip(weights, 0.1, 10)

    coeffs = np.polyfit(np.log10(x), np.log10(y), deg=1, w=weights)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = 10 ** poly(np.log10(x_fit))
    y_fit_per_original_x = 10 ** poly(np.log10(x))
    slope, intercept = coeffs
    return x_fit, y_fit, slope, intercept, y_fit_per_original_x

def plot_permeability_dataset(ax, data, x_col, y_col, color, label, marker, low_col, high_col, linewidth=2, markersize=8):
    
    if data.empty:
        return

    data = data.sort_values(x_col)

    
    if low_col is not None and high_col is not None:
        y_err_lower = data[y_col] - data[low_col]
        y_err_upper = data[high_col] - data[y_col]
        y_err = (y_err_lower + y_err_upper) / 2
        
        x_fit, y_fit, slope, intercept, _ = weighted_power_law_fit(data[x_col], data[y_col], y_err=None)
        ax.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.7, linewidth=linewidth)
        
        x_fit_lo, y_fit_lo, _, _, _ = weighted_power_law_fit(data[x_col], data[low_col], y_err=None)
        x_fit_hi, y_fit_hi, _, _, _ = weighted_power_law_fit(data[x_col], data[high_col], y_err=None)
        
        ax.fill_between(x_fit, y_fit_lo, y_fit_hi, color=color, alpha=0.1)
        ax.errorbar(
            data[x_col],
            data[y_col],
            yerr=[y_err_lower, y_err_upper],
            marker=marker,
            markersize=markersize,
            linestyle='None',
            # label=fr'{label}: $y={10**intercept:.2f}x^{{{slope:.2f}}}$',
            label = fr'{label}$\propto x^{{{slope:.2f}}}$',
            color=color,
            linewidth=linewidth,
        )
    else:
        x_fit, y_fit, slope, intercept, _ = weighted_power_law_fit(data[x_col], data[y_col], None)
        ax.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.7, linewidth=linewidth)
        ax.plot(
            data[x_col],
            data[y_col],
            marker=marker,
            markersize=markersize,
            linestyle='None',
            # label=fr'{label}: $y={10**intercept:.2f}x^{{{slope:.2f}}}$',
            label = fr'{label}$\propto x^{{{slope:.2f}}}$',
            color=color,
            linewidth=linewidth
        )


def plot_full_permeability_comparison():
    n_sites = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    kdas = [radius_a_to_kda(r) for r in radii]
    results_emp = collect_perms_md()
    df_results_emp = pd.DataFrame(results_emp)
    results_tm = collect_perms_msm(n_clusters=320)
    df_results_tm = pd.DataFrame(results_tm)


    # Keep your original method colors
    colors = {"MSM": "#2f9acf", "Empirical": "#c0702f"}
    site_values = [2, 4, 6]

    # Differentiate the site amounts by marker shape
    site_markers = {2: 'o', 4: 's', 6: '^'}

    # Create 1 row, 2 columns (one for MSM, one for MD)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Permeability Comparison: iMSM vs MD', fontsize=30)

    axes[0].set_title('iMSM', fontsize=40)
    axes[1].set_title('MD', fontsize=40)

    for site in site_values:
        # Use .copy() to prevent Pandas SettingWithCopyWarning
        msm_site = df_results_tm[df_results_tm['sites'] == site].copy()
        emp_site = df_results_emp[df_results_emp['sites'] == site].copy()
        
        free_coeff = 1 # single free parameter
        msm_site['permeability'] = msm_site['permeability'] / free_coeff

        # 1. Plot MSM data
        plot_permeability_dataset(
            ax=axes[0],
            data=msm_site,
            x_col='kda',
            y_col='permeability',
            color=colors['MSM'],         # Lock to MSM color
            label=f'{site} sites',
            marker=site_markers[site],   # Differentiate by marker shape
            low_col=None,
            high_col=None,
            linewidth=3,
            markersize=11,
        )
        
        # 2. Plot MD data
        plot_permeability_dataset(
            ax=axes[1],
            data=emp_site,
            x_col='kda',
            y_col='permeability',
            color=colors['Empirical'],   # Lock to MD color
            label=f'{site} sites',
            marker=site_markers[site],   # Differentiate by marker shape
            low_col='lower_permeability',
            high_col='upper_permeability',
            linewidth=3,
            markersize=11,
        )

    # Format both axes
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([10**1, 10**2])
        ax.set_xticklabels([r'$10^1$', r'$10^2$'])
        ax.set_ylim(0.3, 10**4.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Molecular mass (kDa)', fontsize=30)
        
        # Grab the current legend handles and labels, then reverse them
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=24)
        
        ax.tick_params(axis='x', labelsize=27)
        ax.tick_params(axis='y', labelsize=27)

    # Only the leftmost subplot needs the Y-axis label
    axes[0].set_ylabel('Permeability (1/s/µM/NPC)', fontsize=32)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig

def plot_full_permeability_comparison_alt():
    n_sites = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    kdas = [radius_a_to_kda(r) for r in radii]
    results_emp = collect_perms_md()
    df_results_emp = pd.DataFrame(results_emp)
    results_tm = collect_perms_msm(n_clusters=320)
    df_results_tm = pd.DataFrame(results_tm)


    colors = {"MSM": "#2f9acf", "Empirical": "#c0702f"}
    site_values = [2, 4, 6]

    fig, axes = plt.subplots(1, len(site_values), figsize=(18, 6), sharey=True)
    fig.suptitle('Permeability Comparison: MSM vs Empirical MD (Single Free Parameter)', fontsize=24)

    for ax, site in zip(axes, site_values):
        msm_site = df_results_tm[df_results_tm['sites'] == site]
        emp_site = df_results_emp[df_results_emp['sites'] == site]
        # if site == 2 and 'total_transports' in emp_site:
        #     emp_site = emp_site[emp_site['total_transports'] > 0]
        free_coeff = 1 # single free parameter
        msm_site.loc[:, 'permeability'] = msm_site['permeability'] / free_coeff
        # emp_site.loc[:, 'lower_permeability'] = emp_site['lower_permeability'] * free_coeff
        # emp_site.loc[:, 'upper_permeability'] = emp_site['upper_permeability'] * free_coeff

        ax.set_title(f'{site} interaction sites', fontsize=24)
        plot_permeability_dataset(
            ax,
            msm_site,
            'kda',
            'permeability',
            colors['MSM'],
            'MSM',
            'o',
            None,
            None
        )
        plot_permeability_dataset(
            ax,
            emp_site,
            'kda',
            'permeability',
            colors['Empirical'],
            'MD',
            's',
            'lower_permeability',
            'upper_permeability'
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([10**1, 10**2])
        ax.set_xticklabels([r'$10^1$', r'$10^2$'])
        ax.set_ylim(1, 10**4.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Molecular Weight (kDa)', fontsize=22)
        ax.legend(fontsize=16)
        
        # set tick font size
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    axes[0].set_ylabel('Permeability (1/s/µM/NPC)', fontsize=22)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig




def plot_speedup_factor_heatmap(use_sim_subset=False):

    perms_all_md, lower_perms_all_md, upper_perms_all_md, perms_all_md_sim_subset, lower_perms_all_md_sim_subset, upper_perms_all_md_sim_subset, perms_all_msm, perms_all_msm_with_init_time, perms_all_msm_sim_subset, sites_to_fitted_perms, sites_to_fitted_perms_lo, sites_to_fitted_perms_hi = load_convergence_data()


    # Ensure clean white background style for premium plots
    sns.set_theme(style='white')

    # ----------------- Data Processing (Preserved) -----------------
    min_convergence_factor = 0.5
    max_convergence_factor = 2

    md_convergences = np.zeros((len(radii), len(sites_list)))
    msm_convergences = np.zeros((len(radii), len(sites_list)))

    current_subsets = sims_subsets if use_sim_subset else subsets
    current_times = sims_total_simulation_times if use_sim_subset else total_simulation_times

    for i, radius in enumerate(radii):
        for j, sites in enumerate(sites_list):
            if use_sim_subset:
                md_perms = perms_all_md_sim_subset[i, j, :]
                msm_perms = perms_all_msm_sim_subset[i, j, :]
            else:
                md_perms = perms_all_md[i, j, :]
                msm_perms = perms_all_msm[i, j, :]
            
            md_perms_smooth = md_perms
            msm_perms_smooth = np.zeros_like(msm_perms)
            
            for w in range(len(md_perms)):
                msm_perms_smooth[w] = np.mean(msm_perms[: w + 1])
            
            final_md_perm = md_perms_smooth[-1]
            final_msm_perm = msm_perms_smooth[-1]
            
            for k in range(len(current_subsets)):
                if np.all(md_perms_smooth[k:] >= min_convergence_factor * final_md_perm) and \
                np.all(md_perms_smooth[k:] <= max_convergence_factor * final_md_perm):
                    md_convergences[i, j] = current_times[k]
                    break
                    
            for k in range(len(current_subsets)):
                if np.all(msm_perms_smooth[k:] >= min_convergence_factor * final_msm_perm) and \
                np.all(msm_perms_smooth[k:] <= max_convergence_factor * final_msm_perm):
                    msm_convergences[i, j] = current_times[k]
                    break
                
            if (sites == 6 and radius == 22):
                print(md_perms_smooth)
                print(f"MD convergence time: {md_convergences[i, j]} μs")
                print(msm_perms_smooth)
                print(f"iMSM convergence time: {msm_convergences[i, j]} μs")

    min_convergence_time = 60
    max_convergence_time = 1800

    def format_val(v):
        if np.isnan(v) or v == 0:
            return ""
        if v >= max_convergence_time:
            return f"\u2265 {max_convergence_time}"
        if v <= min_convergence_time:
            return f"\u2264 {min_convergence_time}"
        return f"{v:.0f}"

    md_convergences_t = md_convergences.T
    msm_convergences_t = msm_convergences.T
    speedup_factor_t = (md_convergences / msm_convergences).T

    md_labels = np.vectorize(format_val)(md_convergences_t)
    msm_labels = np.vectorize(format_val)(msm_convergences_t)

    # Define custom row labels based on the interaction sites
    row_labels = [f"{s} sites" for s in sites_list]

    # ----------------- Visualizations & Layout Improvements -----------------
    # 1. We define 4 subplots (3 heatmaps and 1 dedicated colorbar axes).
    #    By using a custom width ratio for the colorbar, we prevent matplotlib 
    #    from stealing space from the second heatmap, making all three heatmaps 
    #    the exact same size.
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=4, 
        figsize=(25, 7), 
        gridspec_kw={'width_ratios': [1.0, 1.0, 0.05, 1.0]}, 
        dpi=300
    )

    # 2. Main suptitle with premium size and bold styling
    fig.suptitle(
        'Time to reach between 50% and 200% of Final Value.', 
        fontsize=24, 
        y=0.995, 
        weight='bold'
    )

    # ----------------- MD heatmap (Axis 0) -----------------
    im0 = sns.heatmap(
        md_convergences_t, 
        annot=md_labels, 
        fmt='', 
        cmap='Purples', 
        ax=axes[0], 
        cbar=False, 
        square=True,
        vmin=min_convergence_time, 
        vmax=max_convergence_time, 
        annot_kws={"size": 24},
        linewidth=3
    )
    axes[0].set_xticklabels([f"{kda:.1f}" for kda in kdas], fontsize=21.6)
    axes[0].set_yticklabels(row_labels, rotation=90, va='center', ha='right', fontsize=21.6)
    axes[0].set_xlabel('Molecular mass (kDa)', fontsize=24, labelpad=10)
    axes[0].set_ylabel('')
    axes[0].set_title('MD Convergence Time (μs)', fontsize=28, pad=12)

    # ----------------- MSM heatmap (Axis 1) -----------------
    # We pass the third axes (axes[2]) as cbar_ax to keep axes[1] square and correctly sized
    im1 = sns.heatmap(
        msm_convergences_t, 
        annot=msm_labels, 
        fmt='', 
        cmap='Purples', 
        ax=axes[1], 
        cbar=True, 
        cbar_ax=axes[2],
        square=True,
        vmin=min_convergence_time, 
        vmax=max_convergence_time, 
        annot_kws={"size": 24},
        linewidth=3
    )
    axes[1].set_xticklabels([f"{kda:.1f}" for kda in kdas], fontsize=21.6)
    axes[1].set_yticklabels(row_labels, rotation=90, va='center', ha='right', fontsize=21.6)
    axes[1].set_xlabel('Molecular mass (kDa)', fontsize=24, labelpad=10)
    axes[1].set_ylabel('')
    axes[1].set_title('iMSM Convergence Time (μs)', fontsize=28, pad=12)
    # Style the shared purple colorbar in axes[2]
    cbar1 = im1.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=21.6)
    cbar1.set_label('Convergence Time (μs)', size=24, labelpad=10)
    # <-- 2. Reduce tick density (change nbins to get more or fewer ticks)
    cbar1.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 

    # ------------ Speedup factor heatmap (Axis 3) ------------
    v_min = 0.1
    v_max = 10.0
    norm = mcolors.TwoSlopeNorm(vmin=v_min, vcenter=1.0, vmax=v_max)

    im2 = sns.heatmap(
        speedup_factor_t, 
        annot=True, 
        fmt='.1f', 
        cmap='PiYG', 
        norm=norm,  
        ax=axes[3], 
        square=True, 
        annot_kws={"size": 24},
        cbar=False,
        linewidth=3
    )

    axes[3].set_xticklabels([f"{kda:.1f}" for kda in kdas], fontsize=24)
    axes[3].set_yticklabels(row_labels, rotation=90, va='center', ha='right', fontsize=24)
    axes[3].set_xlabel('Molecular mass (kDa)', fontsize=24, labelpad=10)
    axes[3].set_ylabel('')
    axes[3].set_title(
        f'Speedup Factor. Mean: {np.mean(md_convergences / msm_convergences):.2f}x', 
        fontsize=28, 
        pad=12
    )

    # Render layout nicely with adjusted horizontal pad
    plt.tight_layout(w_pad=0.0)
    # <-- 3. Align colorbar height and adjust distance to the heatmap
    pos1 = axes[1].get_position()  # Position of the square MSM heatmap
    pos2 = axes[2].get_position()  # Position of the colorbar
    # Gap controls the horizontal distance from the MSM heatmap to the colorbar.
    # - Decrease it (e.g. 0.01) to bring the colorbar closer to the left.
    # - Increase it (e.g. 0.03) to move it further to the right.
    gap = 0.015 
    axes[2].set_position([pos1.x1 + gap, pos1.y0, pos2.width, pos1.height])

    plt.show()
    return fig


def plot_all_convergences(use_sim_subset=False):
    perms_all_md, lower_perms_all_md, upper_perms_all_md, perms_all_md_sim_subset, lower_perms_all_md_sim_subset, upper_perms_all_md_sim_subset, perms_all_msm, perms_all_msm_with_init_time, perms_all_msm_sim_subset, sites_to_fitted_perms, sites_to_fitted_perms_lo, sites_to_fitted_perms_hi = load_convergence_data()

    # --- New Boolean Flags ---
    smooth_msm = True           # Set to True to perform cumulative smoothing on MSM data
    include_init_time = False

    # Helper function for MSM smoothing
    def apply_smoothing(data):
        smoothed = np.zeros_like(data)
        for w in range(len(data)):
            smoothed[w] = np.mean(data[: w + 1])
        return smoothed
    # -------------------------

    fig, axes = plt.subplots(len(sites_list), len(radii), figsize=(20, 15), sharex=True, sharey=True, dpi=300)

    labels_for_legend = []
    targ_line = None

    for i, radius in enumerate(radii):
        for j, sites in enumerate(sites_list):
            ax = axes[j, i]
            
            # MD Data
            if use_sim_subset:
                md_perms = perms_all_md_sim_subset[i, j, :]
                md_lower_perms = lower_perms_all_md_sim_subset[i, j, :]
                md_upper_perms = upper_perms_all_md_sim_subset[i, j, :]
                msm_perms = perms_all_msm_sim_subset[i, j, :]
                x_times = np.array(sims_total_simulation_times)
                md_style = STYLE['md_sim']
                msm_style = STYLE['msm_sim']
            else:
                md_perms = perms_all_md[i, j, :]
                md_lower_perms = lower_perms_all_md[i, j, :]
                md_upper_perms = upper_perms_all_md[i, j, :]
                msm_perms = perms_all_msm[i, j, :]
                msm_perms_init_time = perms_all_msm_with_init_time[i, j, :]
                x_times = np.array(total_simulation_times)
                md_style = STYLE['md_time']
                msm_style = STYLE['msm_time']
            
            # Apply smoothing to MSM data if flag is set
            if smooth_msm:
                msm_perms = apply_smoothing(msm_perms)
                if not use_sim_subset:
                    msm_perms_init_time = apply_smoothing(msm_perms_init_time)
                
            div_factor = 1  
            
            # 1. Target "Leeway" Zone
            t_line = plot_target_line(ax, x_times, sites_to_fitted_perms[sites][i], 
                                    sites_to_fitted_perms_lo[sites][i], sites_to_fitted_perms_hi[sites][i])
            if targ_line is None:
                targ_line = t_line

            # 2. MD Plots
            plot_series(ax, x_times, md_perms, md_style, STAR_Y_PERM, md_lower_perms, md_upper_perms)


            # 3. iMSM Plots
            plot_series(ax, x_times, msm_perms / div_factor, msm_style, STAR_Y_PERM)
            if not use_sim_subset and include_init_time and sites != 6:
                # We can use a different marker/color for distinction, here we modify the msm_time style slightly
                style_msm_init = STYLE['msm_time'].copy()
                style_msm_init['color'] = '#4daf4a'  # green
                plot_series(ax, np.array(total_simulation_times), msm_perms_init_time / div_factor, style_msm_init, STAR_Y_PERM)
            
            # Subplot aesthetics
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            ax.set_xlim(x_times[0]*0.9, x_times[-1]*1.1)
            ax.set_ylim(10**-1, 10**4)
            
            ax.tick_params(axis='both', which='major', labelsize=24)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # Add labels to the left of the leftmost subplots
            if i == 0:
                ax.set_ylabel(f'{sites} Sites', fontsize=28, labelpad=15)
                
            # Add labels to the top of the top-row subplots
            if j == 0:
                ax.set_title(f'{kdas[i]:.1f} kDa', fontsize=28, pad=15)


    # 4. Common text and Unified Legend
    fig.text(0.5, 0.02, 'Accumulated Simulation Time (μs)', ha='center', fontsize=24)
    fig.text(0.02, 0.5, 'Permeability (1/s/µM/NPC)', va='center', rotation='vertical', fontsize=24)

    fig.suptitle('Permeability Convergence Across Different Radii and Interaction Sites\n(Red band indicates fitted power law regressor leeway)', 
                fontsize=20, y=0.98)

    # Dynamically build the legend based on the boolean flags
    custom_lines = []

    if use_sim_subset:
        custom_lines.append(Line2D([0], [0], color=STYLE['msm_sim']['color'], marker=STYLE['msm_sim']['marker'], linestyle='-', linewidth=2, label='iMSM (sim subset)'))
        custom_lines.append(Line2D([0], [0], color=STYLE['md_sim']['color'], marker=STYLE['md_sim']['marker'], linestyle='-', linewidth=1.5, alpha=0.8, label='MD (sim subset)'))
    else:
        custom_lines.append(Line2D([0], [0], color=STYLE['msm_time']['color'], marker=STYLE['msm_time']['marker'], linestyle='-', linewidth=2, label='iMSM (time subset)'))
        if include_init_time:
            custom_lines.append(Line2D([0], [0], color='#4daf4a', marker=STYLE['msm_time']['marker'], linestyle='-', linewidth=2, label='iMSM (init time)'))
        custom_lines.append(Line2D([0], [0], color=STYLE['md_time']['color'], marker=STYLE['md_time']['marker'], linestyle='-', linewidth=1.5, alpha=0.8, label='MD (time subset)'))

    custom_lines.append(targ_line)

    # Place one main legend at the top of the figure
    ncols = len(custom_lines)
    fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 0.93), 
            ncol=ncols, frameon=False, fontsize=24)

    # Adjust layout to make room for the new suptitle and top legend
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.90])
    plt.show()
    return fig


def plot_md_vs_msm_convergence():
    """Plots direct-MD convergence time versus iMSM convergence time for all 15 NTR variants."""
    (perms_all_md, _, _, _, _, _, perms_all_msm, _, _, _, _, _) = load_convergence_data()

    min_convergence_factor = 0.5
    max_convergence_factor = 2.0

    md_convergences = np.zeros((len(radii), len(sites_list)))
    msm_convergences = np.zeros((len(radii), len(sites_list)))

    for i, radius in enumerate(radii):
        for j, sites in enumerate(sites_list):
            md_perms = perms_all_md[i, j, :]
            msm_perms = perms_all_msm[i, j, :]
            
            md_perms_smooth = md_perms
            msm_perms_smooth = np.zeros_like(msm_perms)
            
            for w in range(len(md_perms)):
                msm_perms_smooth[w] = np.mean(msm_perms[: w + 1])
            
            final_md_perm = md_perms_smooth[-1]
            final_msm_perm = msm_perms_smooth[-1]
            
            for k in range(len(subsets)):
                if np.all(md_perms_smooth[k:] >= min_convergence_factor * final_md_perm) and \
                   np.all(md_perms_smooth[k:] <= max_convergence_factor * final_md_perm):
                    md_convergences[i, j] = total_simulation_times[k]
                    break
                    
            for k in range(len(subsets)):
                if np.all(msm_perms_smooth[k:] >= min_convergence_factor * final_msm_perm) and \
                   np.all(msm_perms_smooth[k:] <= max_convergence_factor * final_msm_perm):
                    msm_convergences[i, j] = total_simulation_times[k]
                    break

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Square plot with equal logarithmic axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(30, 3000)
    ax.set_ylim(30, 3000)
    ax.set_aspect('equal')
    
    # Grid lines for both major and minor ticks
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Plot identity line (y=x)
    ax.plot([30, 3000], [30, 3000], color='gray', linestyle='--', linewidth=1.5, zorder=1)

    # Plot speedup lines (y = S * x)
    x_vals = np.logspace(np.log10(30), np.log10(3000), 100)
    for s_val in [0.5, 2, 4, 8]:
        y_vals = s_val * x_vals
        valid = (y_vals <= 3000) & (y_vals >= 30)
        if np.any(valid):
            ax.plot(x_vals[valid], y_vals[valid], color='gray', linestyle=':', alpha=0.4, linewidth=1.5, zorder=1)
            # Label speedup line
            if s_val >= 1:
                x_text = 2000 / s_val
                y_text = 2000
            else:
                x_text = 2000
                y_text = 2000 * s_val
            
            y_text_plot = y_text * 1.25
            label_text = f'{s_val}x speedup'
            if x_text >= 30 and y_text_plot >= 30:
                ax.text(x_text, y_text_plot, label_text, rotation=45, color='gray', alpha=0.6, fontsize=18, ha='center', va='center')

    # Color palette
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'] # Viridis
    const_size = 300
    site_markers = {2: 'o', 4: 's', 6: '^'}

    # Plot data points with jitter in log space
    rng = np.random.default_rng(42)
    for i, radius in enumerate(radii):
        kda = kdas[i]
        color = colors[i]
        for j, sites in enumerate(sites_list):
            x_val = msm_convergences[i, j]
            y_val = md_convergences[i, j]
            
            # Add small log10-space jitter
            x_jittered = x_val * (10 ** rng.uniform(-0.03, 0.03))
            y_jittered = y_val * (10 ** rng.uniform(-0.03, 0.03))
            
            marker = site_markers[sites]
            ax.scatter(x_jittered, y_jittered, s=const_size, color=color, marker=marker, edgecolors='black', alpha=0.85, zorder=3)

    # Style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    
    ax.set_xlabel('iMSM Convergence Time (μs)', fontsize=20, labelpad=10)
    ax.set_ylabel('MD Convergence Time (μs)', fontsize=20, labelpad=10)

    # Legends: Both in the same legend box, side-by-side (2 columns) below the plot
    hdr_sites = Line2D([0], [0], color='none', label=r'$\bf{Binding\ Sites}$')
    hdr_kda = Line2D([0], [0], color='none', label=r'$\bf{Molecular\ Mass}$')
    empty_handle = Line2D([0], [0], color='none', label='')

    ms_legend = 10
    col1 = [
        hdr_sites,
        Line2D([0], [0], marker='o', color='none', markerfacecolor='gray', markeredgecolor='black', markersize=ms_legend, label='2 sites'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='gray', markeredgecolor='black', markersize=ms_legend, label='4 sites'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='gray', markeredgecolor='black', markersize=ms_legend, label='6 sites'),
        empty_handle,
        empty_handle
    ]

    col2 = [hdr_kda]
    for i, kda in enumerate(kdas):
        color = colors[i]
        col2.append(
            Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markeredgecolor='black', markersize=ms_legend, label=f'{kda:.1f} kDa')
        )

    # Concatenate columns directly since matplotlib legend is column-major by default
    combined_handles = col1 + col2

    ax.legend(
        handles=combined_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=True,
        fontsize=18,
        columnspacing=2.0,
        handletextpad=0.5
    )

    plt.tight_layout()
    return fig


def collect_perms_md_single_type_diameters(kap_coords_folder, n_sims, start_time, end_time, shorten_time_to, step_ns=100):
    """
    Count transport events from raw 3D kap coordinates.
    
    kap_coords_folder: path containing subfolders 1, 2, ..., n_sims,
                       each with a file 'start_time-end_time.pickle' of shape (kap_amount, 3, n_frames).
    step_ns: time per frame in nanoseconds (default 100 ns).
    """
    # Load and concatenate raw 3D coordinates across all simulations
    all_coords = []
    for sim_idx in range(1, n_sims + 1):
        path = os.path.join(kap_coords_folder, str(sim_idx), f"{start_time}-{end_time}.pickle")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        with open(path, "rb") as f:
            coords = pickle.load(f)  # shape: (kap_amount, 3, n_frames)
        all_coords.append(coords)

    if not all_coords:
        return 0, 0, 0, 0, 0

    all_coords = np.concatenate(all_coords, axis=0)  # shape: (kap_amount * n_sims, 3, n_frames)

    _, _, n_frames = all_coords.shape
    time_us = n_frames * step_ns / 1000  # convert ns → µs

    if shorten_time_to is not None:
        if shorten_time_to > time_us:
            # raise ValueError(f"shorten_time_to {shorten_time_to} µs is greater than trajectory length {time_us} µs")
            n_frames_to_use = n_frames
            time_us = time_us
        else:
            n_frames_to_use = int(shorten_time_to * 1000 / step_ns)
            time_us = shorten_time_to
        all_coords = all_coords[:, :, :n_frames_to_use]

    # Count transport events across all kap trajectories
    # z-axis is index 2 in the (kap_amount, 3, n_frames) array
    total_full_transports = 0
    n_trajs = all_coords.shape[0]
    for i in range(n_trajs):
        z = all_coords[i, 2, :]  # shape: (n_frames,)
        last_side = None

        for z_val in z:
            if z_val >= 10:
                current_side = 1   # cytoplasmic side (top)
            elif z_val <= -10:
                current_side = 0   # nuclear side (bottom)
            else:
                continue           # inside the channel, not yet committed

            if last_side is None:
                last_side = current_side   # initialise, don't count
            elif last_side != current_side:
                total_full_transports += 1
                last_side = current_side

    concentration_M = amount_to_concentration(100, box_side_a=800)
    concentration_uM = concentration_M * 1e6
            
    time_s = time_us * 1e-6

    lower, upper = poisson_confidence_interval(total_full_transports)
    rate = total_full_transports / (time_s * n_sims)
    lower_rate = lower / (time_s * n_sims)
    upper_rate = upper / (time_s * n_sims)
    permeability = rate / concentration_uM
    lower_permeability = lower_rate / concentration_uM
    upper_permeability = upper_rate / concentration_uM
    return n_trajs, total_full_transports, permeability, lower_permeability, upper_permeability


# Collect data
def collect_perms_md_diameters(n_sites=None, radii=None, shorten_time_to=None):
    results = []
    for sites in n_sites:
        for radius in radii:
            for tunnel_diameter_nm in [46, 54, 62, 70]:
                if tunnel_diameter_nm == 54:
                    kap_coords_folder = f"data/ntr_variants/{sites}_{radius}_more/1_single_sim_kap_coords/"
                    n_trajs, total_full_transports, permeability, lower_permeability, upper_permeability = collect_perms_md_single_type_diameters(kap_coords_folder, n_sims=30, start_time=10, end_time=70, shorten_time_to=shorten_time_to)
                else:
                    kap_coords_folder = f"data/ntr_variants_{tunnel_diameter_nm}R/{sites}_{radius}/1_single_sim_kap_coords/"
                    n_trajs, total_full_transports, permeability, lower_permeability, upper_permeability = collect_perms_md_single_type_diameters(kap_coords_folder, n_sims=30, start_time=10, end_time=30, shorten_time_to=shorten_time_to)
                
                results.append({
                    'sites': sites,
                    'radius': radius,
                    'kda': radius_a_to_kda(radius),
                    'tunnel_diameter_nm': tunnel_diameter_nm,
                    'total_transports': total_full_transports,
                    'n_trajs': n_trajs,
                    'permeability': permeability,
                    'lower_permeability': lower_permeability if lower_permeability >=0 else 0,
                    'upper_permeability': upper_permeability if upper_permeability >=0 else 0
                })
    return results


def collect_perms_msm_diameters(n_sites=None, radii=None, n_clusters=320):
    results = []
    for sites in n_sites:
        for radius in radii:
            for tunnel_diameter_nm in [46, 54, 62, 70]:
                path = None
                if tunnel_diameter_nm == 54:
                    path = f"data/ntr_variants/{sites}_{radius}_more/7_permeabilities_subsets/1.00fraction_simulations/0index/7_permeabilities.pickle"
                else:
                    path = f"data/ntr_variants_{tunnel_diameter_nm}R/{sites}_{radius}/7_permeabilities_subsets/1.00fraction/0index/7_permeabilities.pickle"
                
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        perms_dict = pickle.load(f)
                    permeability = perms_dict[n_clusters]
                    results.append({
                        'sites': sites,
                        'radius': radius,
                        'kda': radius_a_to_kda(radius),
                        'tunnel_diameter_nm': tunnel_diameter_nm,
                        'permeability': permeability
                    })
    return results


def plot_permeability_pore_diameter_comparison():
    n_sites = [2, 4, 6]
    radii = [10, 14, 18, 22, 26]
    md_results = collect_perms_md_diameters(n_sites=n_sites, radii=radii, shorten_time_to=20)
    msm_results = collect_perms_msm_diameters(n_sites=n_sites, radii=radii, n_clusters=320)

    # Convert the results list of dicts to Pandas DataFrames
    df_md = pd.DataFrame(md_results)
    df_msm = pd.DataFrame(msm_results)

    # Get the unique site values and diameters
    n_sites_list = sorted(df_md['sites'].unique())
    diameters = sorted(df_md['tunnel_diameter_nm'].unique()) # [46, 54, 62, 70]

    # Create a figure with 3 rows (Sites) and 4 columns (Diameters)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=True, sharex=True)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    diameter_colors = {d: color_cycle[i] for i, d in enumerate(diameters)}

    for row_idx, sites in enumerate(n_sites_list):
        df_md_sites = df_md[df_md['sites'] == sites]
        df_msm_sites = df_msm[df_msm['sites'] == sites]
        
        for col_idx, diameter in enumerate(diameters):
            ax = axes[row_idx, col_idx]
            color = diameter_colors[diameter]
            
            # --- MD Data ---
            df_plot = df_md_sites[df_md_sites['tunnel_diameter_nm'] == diameter].sort_values('kda')
            if not df_plot.empty:
                x = df_plot['kda'].values
                y = df_plot['permeability'].values
                valid = y > 0
                invalid = y == 0
                
                # Plot valid points (greater than 0 transport events)
                if np.any(valid):
                    x_v = x[valid]
                    y_v = y[valid]
                    yerr_l = np.maximum(0, y_v - df_plot['lower_permeability'].values[valid])
                    yerr_u = np.maximum(0, df_plot['upper_permeability'].values[valid] - y_v)
                    
                    ax.errorbar(
                        x_v, y_v,
                        yerr=[yerr_l, yerr_u],
                        marker='o',
                        capsize=5,
                        capthick=1.5,
                        elinewidth=1.5,
                        linestyle='none',
                        color=color
                    )
                    
                    if len(x_v) >= 2:
                        x_fit, y_fit, _, _, _ = weighted_power_law_fit(x_v, y_v, y_err=None)
                        ax.plot(x_fit, y_fit, linestyle='-', color=color, alpha=0.7)

                # Plot invalid points (0 transport events) as upper limits
                if np.any(invalid):
                    x_inv = x[invalid]
                    y_inv = df_plot['upper_permeability'].values[invalid]
                    ax.errorbar(
                        x_inv, y_inv,
                        yerr=[y_inv, np.zeros_like(y_inv)],  # Lower error goes to 0, upper is 0
                        marker='none',
                        capsize=5,
                        capthick=1.5,
                        elinewidth=1.5,
                        linestyle='none',
                        color=color
                    )

            # --- MSM Data ---
            df_msm_plot = df_msm_sites[df_msm_sites['tunnel_diameter_nm'] == diameter].sort_values('kda')
            if not df_msm_plot.empty:
                x_msm = df_msm_plot['kda'].values
                y_msm = df_msm_plot['permeability'].values
                valid_msm = y_msm > 0
                if np.any(valid_msm):
                    x_mv = x_msm[valid_msm]
                    y_mv = y_msm[valid_msm]

                    ax.plot(
                        x_mv, y_mv,
                        marker='s',
                        linestyle='none',
                        color=color
                    )
                    
                    if len(x_mv) >= 2:
                        x_fit_msm, y_fit_msm, _, _, _ = weighted_power_law_fit(x_mv, y_mv, y_err=None)
                        ax.plot(x_fit_msm, y_fit_msm, linestyle='--', color=color, alpha=0.7)

            # --- Formatting ---
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Row labels on the y-axis of the first column
            if col_idx == 0:
                ax.set_ylabel(f'{sites} Sites\nPermeability', fontsize=16)
            
            # Column labels on the title of the first row
            if row_idx == 0:
                ax.set_title(f'{diameter} nm', fontsize=18)
                
            # X-axis label on the bottom row only
            if row_idx == len(n_sites_list) - 1:
                ax.set_xlabel('Molecular mass (kDa)', fontsize=16)

    # Adjust x-axis across all columns
    axes[0, 0].set_xlim(left=2.5)

    # Add a global legend for MD and MSM markers/linestyles on the first plot
    md_legend = Line2D([], [], color='gray', marker='o', linestyle='-', label='MD')
    msm_legend = Line2D([], [], color='gray', marker='s', linestyle='--', label='MSM')
    axes[0, 0].legend(handles=[md_legend, msm_legend], loc='lower left', fontsize=14)

    plt.tight_layout()
    return fig