# %% 
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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Any
from iMSM.extensions.npc.npc_utils import radius_a_to_kda

# Set random seed for reproducibility in sampling
np.random.seed(42)

def load_all_coordinates(base_dir: str, num_folders: int, filename: str, max_frames: int, exclude_folders: List[int]) -> np.ndarray:
    """
    Loads coordinate files from folders 1 to num_folders under base_dir, excluding specified folders.
    Extracts and merges all 3D coordinates over all molecules and all times.
    Slices the frame dimension to max_frames to ensure equal data sizes.
    """
    coords_list: List[np.ndarray] = []
    for i in range(1, num_folders + 1):
        if i in exclude_folders:
            continue
        folder_path: str = os.path.join(base_dir, str(i))
        file_path: str = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required coordinate file not found: {file_path}")
            
        with open(file_path, "rb") as f:
            coords: np.ndarray = pickle.load(f)
            # Expect shape (kap_amount, 3, n_frames) or (kap_amount, 1, 3, n_frames)
            if len(coords.shape) == 4:
                coords = coords[:, 0, :, :]
            
            # Slice frame dimension to ensure equal amount of data across variants
            if coords.shape[-1] > max_frames:
                coords = coords[:, :, :max_frames]
            
            # Transpose from (kap_amount, 3, n_frames) to (kap_amount, n_frames, 3)
            # then reshape to (kap_amount * n_frames, 3)
            reshaped_coords: np.ndarray = coords.transpose(0, 2, 1).reshape(-1, 3)
            coords_list.append(reshaped_coords)
            
    return np.concatenate(coords_list, axis=0)

def apply_eightwise_symmetry(coords: np.ndarray, num_rotations: int) -> np.ndarray:
    """
    Applies eightwise symmetry (or any specified num_rotations) around the z-axis.
    Copies and rotates the coordinates around the Z-axis in equal angular steps.
    """
    sym_coords: List[np.ndarray] = []
    for shift in range(num_rotations):
        theta: float = shift * 2.0 * np.pi / num_rotations
        c: float = np.cos(theta)
        s: float = np.sin(theta)
        rotated: np.ndarray = np.copy(coords)
        x: np.ndarray = rotated[:, 0]
        y: np.ndarray = rotated[:, 1]
        
        # Calculate new x and y, leaving z unchanged
        rotated[:, 0] = x * c - y * s
        rotated[:, 1] = x * s + y * c
        sym_coords.append(rotated)
        
    return np.vstack(sym_coords)

def compute_free_energy_landscapes(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 1D and 2D free energy landscapes.
    The distributions of positions are computed over a cylinder (length=50 nm; radius=25 nm)
    centered at Z=0. Voxel size is 1 nm.
    """
    x: np.ndarray = coords[:, 0]
    y: np.ndarray = coords[:, 1]
    z: np.ndarray = coords[:, 2]
    r: np.ndarray = np.sqrt(x**2 + y**2)
    
    # Filter coords in the cylinder
    mask: np.ndarray = (z >= -25.0) & (z <= 25.0) & (r <= 45.0)
    filtered_z: np.ndarray = z[mask]
    filtered_r: np.ndarray = r[mask]
    
    # 1D Free Energy Landscape along z (voxel size 1 nm)
    z_edges: np.ndarray = np.linspace(-25.0, 25.0, 51)
    z_centers: np.ndarray = 0.5 * (z_edges[:-1] + z_edges[1:])
    hist_1d, _ = np.histogram(filtered_z, bins=z_edges)
    
    hist_1d_cleaned: np.ndarray = np.where(hist_1d > 0.0, hist_1d, np.nan)
    p_1d: np.ndarray = hist_1d_cleaned / np.nansum(hist_1d_cleaned)
    free_energy_1d: np.ndarray = -np.log(p_1d)
    
    # Correctly scale by subtracting average of (0,0,25) and (0,0,-25)
    z_neg25_idx = np.argmin(np.abs(z_centers - (-25.0)))
    z_pos25_idx = np.argmin(np.abs(z_centers - 25.0))
    baseline_1d = 0.5 * (free_energy_1d[z_neg25_idx] + free_energy_1d[z_pos25_idx])
    free_energy_1d = free_energy_1d - baseline_1d
    
    # 2D Free Energy Landscape along R and Z (voxel size 1 nm)
    r_edges: np.ndarray = np.linspace(0.0, 25.0, 26)
    r_centers: np.ndarray = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist_2d, _ = np.histogramdd(np.column_stack((filtered_r, filtered_z)), bins=[r_edges, z_edges])
    
    density_2d: np.ndarray = np.zeros_like(hist_2d)
    for i in range(len(r_centers)):
        density_2d[i, :] = hist_2d[i, :] / r_centers[i]
        
    density_2d_cleaned: np.ndarray = np.where(density_2d > 0.0, density_2d, np.nan)
    p_2d: np.ndarray = density_2d_cleaned / np.nansum(density_2d_cleaned)
    free_energy_2d: np.ndarray = -np.log(p_2d)
    
    # Correctly scale by subtracting average of (0,0,25) and (0,0,-25), which is r=0, z=-25 and r=0, z=25
    r0_idx = np.argmin(np.abs(r_centers - 0.0))
    baseline_2d = 0.5 * (free_energy_2d[r0_idx, z_neg25_idx] + free_energy_2d[r0_idx, z_pos25_idx])
    free_energy_2d = free_energy_2d - baseline_2d
    
    return free_energy_1d, z_centers, free_energy_2d, r_centers


def get_variant_color(
    diameter: int,
    radius: float,
    min_r: float,
    max_r: float,
) -> Tuple[float, float, float]:
    """
    Computes a color for a given pore diameter and molecule radius.
    The color gets darker as the molecule radius (size) is bigger.
    """
    diam_colors: Dict[int, str] = {
        46: "#1f77b4",  # blue
        54: "#ff7f0e",  # orange
        62: "#2ca02c",  # green
        70: "#d62728"   # red
    }
    base_hex: str = diam_colors.get(diameter, "#7f7f7f")
    base_rgb: np.ndarray = np.array(mcolors.to_rgb(base_hex))
    
    frac: float = (radius - min_r) / (max_r - min_r)
    
    # Blend with white for the lightest shade and black for the darkest shade
    light_rgb: np.ndarray = 0.4 * base_rgb + 0.6 * np.array([1.0, 1.0, 1.0])
    dark_rgb: np.ndarray = 0.7 * base_rgb + 0.3 * np.array([0.0, 0.0, 0.0])
    
    interpolated_rgb: np.ndarray = light_rgb + frac * (dark_rgb - light_rgb)
    return (float(interpolated_rgb[0]), float(interpolated_rgb[1]), float(interpolated_rgb[2]))


# %%
# Coordinates processing & generating/saving landscapes
def generate_free_energy_data() -> None:
    configs: List[Tuple[str, int, str, str]] = [
        ("ntr_variants_46R", 46, "", "10-30.pickle"),
        ("ntr_variants", 54, "_more", "10-70.pickle"),
        ("ntr_variants_62R", 62, "", "10-30.pickle"),
        ("ntr_variants_70R", 70, "", "10-30.pickle"),
    ]
    
    sites_list: List[int] = [2, 4, 6]
    radius_list: List[int] = [10, 14, 18, 22, 26]
    num_sim_folders: int = 30
    
    DATA_DIR: str = "data/023_EXP_stair"
    os.makedirs(os.path.join(DATA_DIR, "landscapes"), exist_ok=True)
    
    results_list: List[Dict[str, Any]] = []
    
    print("Starting processing of NTR variants and generating landscapes...")
    
    for dir_name, diameter, suffix, pickle_name in configs:
        print(f"\n==================================================")
        print(f"Processing diameter {diameter} nm in {dir_name}")
        print(f"==================================================")
        
        for sites in sites_list:
            for radius in radius_list:
                mol_name: str = f"{sites}_{radius}{suffix}"
                unique_name: str = f"{diameter}nm_{sites}_{radius}{suffix}"
                base_coords_dir: str = f"data/{dir_name}/{mol_name}/1_single_sim_kap_coords"
                
                if not os.path.exists(base_coords_dir):
                    print(f"Skipping {mol_name} (directory does not exist: {base_coords_dir})")
                    continue
                
                print(f"\n--- Loading coordinates for molecule {unique_name} ---")
                exclude: List[int] = []
                if diameter == 70 and sites == 2:
                    exclude = [1, 2, 3, 4, 21, 22]
                try:
                    raw_coords: np.ndarray = load_all_coordinates(
                        base_dir=base_coords_dir,
                        num_folders=num_sim_folders,
                        filename=pickle_name,
                        max_frames=200,
                        exclude_folders=exclude
                    )
                except Exception as e:
                    print(f"Error loading coordinates for {unique_name}: {e}")
                    continue
                
                print(f"Loaded raw coordinates shape: {raw_coords.shape}")
                
                # Apply 8-wise symmetry around Z-axis
                sym_coords: np.ndarray = apply_eightwise_symmetry(coords=raw_coords, num_rotations=8)
                print(f"Symmetrized coordinates shape: {sym_coords.shape}")
                
                # Compute landscapes
                free_energy_1d, z_centers, free_energy_2d, r_centers = compute_free_energy_landscapes(sym_coords)
                z0_idx = np.argmin(np.abs(z_centers - 0.0))
                z0_free_energy: float = float(free_energy_1d[z0_idx])
                
                # Save landscapes
                landscape_file: str = os.path.join(DATA_DIR, "landscapes", f"{unique_name}_landscapes.pickle")
                with open(landscape_file, "wb") as f:
                    pickle.dump({
                        "free_energy_1d": free_energy_1d,
                        "z_centers": z_centers,
                        "free_energy_2d": free_energy_2d,
                        "r_centers": r_centers,
                        "z0_free_energy": z0_free_energy
                    }, f)
                    
                results_list.append({
                    "diameter": diameter,
                    "sites": sites,
                    "radius": radius,
                    "molecule_name": unique_name,
                    "landscape_file": landscape_file,
                    "z0_free_energy": z0_free_energy
                })

    results_path: str = os.path.join(DATA_DIR, "results.pickle")
    with open(results_path, "wb") as f:
        pickle.dump(results_list, f)

    print("\n==================================================")
    print(f"Successfully processed all variants.")
    print(f"Results list saved to: {results_path}")
    print("==================================================")


# %%
# Plotting results
def plot_z0_free_energy_vs_mw() -> None:
    """
    Plots 1D free energy at z=0 vs Molecular Weight.
    """
    results_path: str = "data/023_EXP_stair/results.pickle"
    plot_output_path: str = "plots/023_EXP_stair_z0_plot.png"
    
    if not os.path.exists(results_path):
        print(f"Results file not found for plotting: {results_path}")
        return

    with open(results_path, "rb") as f:
        results_data: List[Dict[str, Any]] = pickle.load(f)

    # Extract the z=0 energy directly from the saved landscape files if not already present
    processed_results: List[Dict[str, Any]] = []
    for item in results_data:
        copied_item: Dict[str, Any] = dict(item)
        if "z0_free_energy" not in copied_item:
            landscape_path = copied_item.get("landscape_file")
            if landscape_path and os.path.exists(landscape_path):
                with open(landscape_path, "rb") as lf:
                    lf_data = pickle.load(lf)
                    fe_1d = lf_data["free_energy_1d"]
                    z_centers = lf_data["z_centers"]
                    z0_idx = np.argmin(np.abs(z_centers - 0.0))
                    copied_item["z0_free_energy"] = float(fe_1d[z0_idx])
        processed_results.append(copied_item)

    # Get sorted unique sites and diameters
    unique_sites: List[int] = sorted(list(set(item["sites"] for item in processed_results if "z0_free_energy" in item)))
    unique_diams: List[int] = sorted(list(set(item["diameter"] for item in processed_results if "z0_free_energy" in item)))
    num_subplots: int = len(unique_sites)
    
    if num_subplots == 0:
        print("No results found for 1D free energy at z=0.")
        return
        
    fig, axes = plt.subplots(1, 4, figsize=(28, 9.0))
    # Share y-axis for the first three panels
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])
    # Hide y-tick labels for shared axes
    plt.setp(axes[1].get_yticklabels(), visible=False)
    plt.setp(axes[2].get_yticklabels(), visible=False)
        
    fits_to_annotate_per_ax: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_subplots)}
    slopes_data: Dict[int, Dict[int, float]] = {site: {} for site in unique_sites}

    for idx, num_sites in enumerate(unique_sites):
        ax = axes[idx]
        site_results = [item for item in processed_results if item["sites"] == num_sites and "z0_free_energy" in item]
        
        for diam in unique_diams:
            diam_site_results = [item for item in site_results if item["diameter"] == diam]
            if not diam_site_results:
                continue
                
            diam_site_results = sorted(diam_site_results, key=lambda x: x["radius"])
            
            mws: List[float] = [float(radius_a_to_kda(item["radius"])) for item in diam_site_results]
            energies: List[float] = [item["z0_free_energy"] for item in diam_site_results]
            
            valid_indices = [i for i, val in enumerate(energies) if np.isfinite(val)]
            if not valid_indices:
                continue
            
            filtered_mws = np.array([mws[i] for i in valid_indices])
            filtered_energies = np.array([energies[i] for i in valid_indices])
            
            if num_sites == 2 and diam == 46:
                if len(filtered_mws) >= 2:
                    mws_to_fit = filtered_mws[:2]
                    energies_to_fit = filtered_energies[:2]
                    
                    log_mws = np.log10(mws_to_fit)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_mws, energies_to_fit)
                    
                    line, = ax.plot(filtered_mws, filtered_energies, marker="o", linestyle="", markersize=9, label=f"{diam} nm")
                    color = line.get_color()
                    
                    # Do not plot the fit line, but compute annotation using full range of filtered_mws
                    fits_to_annotate_per_ax[idx].append({
                        "type": "fit",
                        "diam": diam,
                        "slope": slope,
                        "intercept": intercept,
                        "mws": filtered_mws,
                        "color": color,
                        "label_text": "*"
                    })
                    continue
                else:
                    line, = ax.plot(filtered_mws, filtered_energies, marker="o", linestyle="", markersize=9, label=f"{diam} nm")
                    fits_to_annotate_per_ax[idx].append({
                        "type": "insufficient",
                        "diam": diam,
                        "x": filtered_mws[0],
                        "y": filtered_energies[0],
                        "color": line.get_color()
                    })
                    continue

            log_mws = np.log10(filtered_mws)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_mws, filtered_energies)
            
            line, = ax.plot(filtered_mws, filtered_energies, marker="o", linestyle="", markersize=9, label=f"{diam} nm")
            color = line.get_color()
            
            slopes_data[num_sites][diam] = slope
            
            fit_mws = np.linspace(min(filtered_mws), max(filtered_mws), 100)
            fit_energies = slope * np.log10(fit_mws) + intercept
            ax.plot(fit_mws, fit_energies, linestyle="--", linewidth=3.0, color=color)
            
            fits_to_annotate_per_ax[idx].append({
                "type": "fit",
                "diam": diam,
                "slope": slope,
                "intercept": intercept,
                "mws": filtered_mws,
                "color": color
            })
            
        ax.set_title(f"{num_sites} Sites", fontsize=33)
        ax.set_xlabel("Molecular Weight (kDa)", fontsize=28)
        ax.set_xscale("log")
        ax.tick_params(axis='both', labelsize=28)
        
        if idx == 0:
            ax.set_ylabel(r"$\Delta G$ ($k_B T$)", fontsize=28)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-1.2, 6)
        ax.set_xlim(right=100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if num_sites == 6:
            ax.legend(
                loc="upper center",
                ncol=2,
                fontsize=26,
                frameon=True,
                facecolor="white",
                edgecolor="none",
                markerscale=2.5
            )
        
    # Plot fourth panel: slope vs pore diameter
    ax_slope = axes[3]
    marker_shapes = {2: "o", 4: "s", 6: "^"}
    purple_color = "#6f42c1" # Nice modern purple
    for valency in unique_sites:
        diams = []
        slopes = []
        for diam in unique_diams:
            if diam in slopes_data[valency]:
                diams.append(diam)
                slopes.append(slopes_data[valency][diam])
        if diams:
            # Plot the points
            ax_slope.plot(
                diams, 
                slopes, 
                marker=marker_shapes.get(valency, "o"), 
                linestyle="", 
                markersize=16, 
                color=purple_color, 
                label=f"{valency} Sites"
            )
            # Compute and plot linear regression line (dashed)
            if len(diams) >= 2:
                slope_fit, intercept_fit, _, _, _ = stats.linregress(diams, slopes)
                fit_diams = np.array(diams)
                fit_slopes = slope_fit * fit_diams + intercept_fit
                ax_slope.plot(
                    fit_diams, 
                    fit_slopes, 
                    linestyle="--", 
                    linewidth=2.0, 
                    color=purple_color
                )
            
    ax_slope.set_title("Slope vs Pore Diameter", fontsize=33)
    ax_slope.set_xlabel("Pore Diameter (nm)", fontsize=28)
    ax_slope.set_ylabel(r"Slope ($k_B T$)", fontsize=28)
    ax_slope.tick_params(axis='both', labelsize=28)
    ax_slope.grid(True, linestyle="--", alpha=0.5)
    ax_slope.set_xticks(unique_diams)
    ax_slope.spines['top'].set_visible(False)
    ax_slope.spines['right'].set_visible(False)
    
    # Adjust y-limit top to be +0.5 from current max limit
    y_min, y_max = ax_slope.get_ylim()
    ax_slope.set_ylim(y_min, y_max + 0.5)
    
    # Legend in 1 line at the top with minimal padding and no white box frame
    ax_slope.legend(
        fontsize=22, 
        loc="upper center", 
        ncol=3, 
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
        borderpad=0.3
    )
    
    plt.suptitle("1D Free Energy at z=0 vs Molecular Weight", fontsize=38, y=1.02)
    plt.tight_layout()
    fig.canvas.draw()

    # Add annotations after layout is resolved so that screen/display angles are accurate
    for idx, num_sites in enumerate(unique_sites):
        ax = axes[idx]
        
        # Separate insufficient from fit annotations
        all_fits = fits_to_annotate_per_ax[idx]
        insufficient_fits = [f for f in all_fits if f["type"] == "insufficient"]
        fit_lines = sorted([f for f in all_fits if f["type"] == "fit"], key=lambda f: f["diam"])
        
        # Annotate insufficient data
        for fit in insufficient_fits:
            ax.text(
                0.05,
                0.90,
                f"{fit['diam']} nm *",
                color=fit["color"],
                weight="bold",
                fontsize=26,
                ha="left",
                va="top",
                transform=ax.transAxes
            )
            
        for fit in fit_lines:
            slope = fit["slope"]
            intercept = fit["intercept"]
            mws = fit["mws"]
            color = fit["color"]
            diam = fit["diam"]
            
            log_mws_min = np.log10(min(mws))
            log_mws_max = np.log10(max(mws))
            
            # Midpoint placement (fraction = 0.5)
            log_x_text = 0.5 * (log_mws_min + log_mws_max)
            x_text = 10**log_x_text
            y_text = slope * log_x_text + intercept
            
            # Select two points close to midpoint to get tangent slope in display coordinates
            x1 = 10**(log_x_text - 0.05)
            y1 = slope * np.log10(x1) + intercept
            x2 = 10**(log_x_text + 0.05)
            y2 = slope * np.log10(x2) + intercept
            
            trans = ax.transData.transform
            p1 = trans((x1, y1))
            p2 = trans((x2, y2))
            
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            if "label_text" in fit:
                label_text = fit["label_text"]
            else:
                label_text = fr"$\alpha = {slope:.2f}\log(x)$"
            
            ax.annotate(
                label_text,
                xy=(x_text, y_text),
                xytext=(0, 6),
                textcoords="offset points",
                rotation=angle,
                rotation_mode="anchor",
                color=color,
                weight="bold",
                fontsize=26,
                ha="center",
                va="bottom"
            )
    
    return fig


def plot_all_1d_landscapes() -> Any:
    """
    Plots all 1D landscapes per site variant.
    """
    results_path: str = "data/023_EXP_stair/results.pickle"
    
    if not os.path.exists(results_path):
        print(f"Results file not found for plotting: {results_path}")
        return None

    with open(results_path, "rb") as f:
        results_data: List[Dict[str, Any]] = pickle.load(f)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharex=True, sharey=True)
    
    sites_to_row: Dict[int, int] = {2: 0, 4: 1, 6: 2}
    radius_to_col: Dict[int, int] = {10: 0, 14: 1, 18: 2, 22: 3, 26: 4}
    
    for item in results_data:
        landscape_path: str = item.get("landscape_file", "")
        if not landscape_path or not os.path.exists(landscape_path):
            rel_candidate: str = os.path.join("data/023_EXP_stair/landscapes", os.path.basename(landscape_path))
            if os.path.exists(rel_candidate):
                landscape_path = rel_candidate
            else:
                continue
        
        sites: int = item["sites"]
        radius: int = item["radius"]
        if sites not in sites_to_row or radius not in radius_to_col:
            continue
        row_idx: int = sites_to_row[sites]
        col_idx: int = radius_to_col[radius]
        ax = axes[row_idx, col_idx]
        
        with open(landscape_path, "rb") as lf:
            lf_data: Dict[str, Any] = pickle.load(lf)
            fe_1d: np.ndarray = lf_data["free_energy_1d"]
            z_centers: np.ndarray = lf_data["z_centers"]
        
        # Smooth the 1D energy landscape with a Gaussian kernel (2nm std)
        # Since the bin size is 1 nm, sigma=2.0 represents 2 nm std.
        nan_mask: np.ndarray = np.isnan(fe_1d)
        
        diam: int = item["diameter"]
        
        # Get the color based only on pore diameter
        diam_colors: Dict[int, str] = {
            46: "#1f77b4",  # blue
            54: "#ff7f0e",  # orange
            62: "#2ca02c",  # green
            70: "#d62728"   # red
        }
        color: Tuple[float, float, float] = mcolors.to_rgb(diam_colors.get(diam, "#7f7f7f"))
        
        label: str = f"{diam} nm"
        
        if np.any(nan_mask):
            non_nan_indices: np.ndarray = np.where(~nan_mask)[0]
            nan_indices: np.ndarray = np.where(nan_mask)[0]
            fe_1d_filled: np.ndarray = np.copy(fe_1d)
            if len(non_nan_indices) > 0:
                fe_1d_filled[nan_mask] = np.interp(nan_indices, non_nan_indices, fe_1d[non_nan_indices])
            fe_1d_smoothed: np.ndarray = gaussian_filter1d(fe_1d_filled, sigma=2.0)
            
            fe_1d_solid: np.ndarray = np.copy(fe_1d_smoothed)
            fe_1d_solid[nan_mask] = np.nan
            
            ax.plot(z_centers, fe_1d_smoothed, color=color, alpha=0.7, linewidth=1.2, linestyle="--")
            ax.plot(z_centers, fe_1d_solid, color=color, alpha=0.7, linewidth=1.2, label=label)
        else:
            fe_1d_smoothed: np.ndarray = gaussian_filter1d(fe_1d, sigma=2.0)
            ax.plot(z_centers, fe_1d_smoothed, color=color, alpha=0.7, linewidth=1.2, label=label)
    
    for sites, row_idx in sites_to_row.items():
        for radius, col_idx in radius_to_col.items():
            ax = axes[row_idx, col_idx]
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_ylim(-1.5, 7)
            
            # Set labels on edges
            if row_idx == 2:
                ax.set_xlabel("z (nm)", fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(f"{sites} Sites\nFree Energy ($k_B T$)", fontsize=14)
            if row_idx == 0:
                mw: float = radius_a_to_kda(radius)
                ax.set_title(f"r = {radius} Å\n({mw:.1f} kDa)", fontsize=14)
                
    # Collect legend handles and labels from all subplots to ensure all diameters are represented
    handles_dict: Dict[str, Any] = {}
    for r in range(3):
        for c in range(5):
            h, l = axes[r, c].get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in handles_dict:
                    handles_dict[label] = handle
                    
    if handles_dict:
        sorted_labels: List[str] = sorted(handles_dict.keys(), key=lambda x: int(x.split()[0]))
        sorted_handles: List[Any] = [handles_dict[lbl] for lbl in sorted_labels]
        # Place the legend in the top-right subplot (row 0, col 4)
        leg = axes[0, 4].legend(sorted_handles, sorted_labels, title="Pore Diameter", loc="upper right", fontsize=10, title_fontsize=11)
        for line in leg.get_lines():
            line.set_linewidth(3.0)
            
    plt.suptitle("1D Free Energy Landscapes per Site and Molecular Weight", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig