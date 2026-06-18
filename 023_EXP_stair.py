# %% 
import os
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
if __name__ == "__main__":
    configs: List[Tuple[str, int, str, str]] = [
        ("ntr_variants_46R", 46, "", "10-30.pickle"),
        ("ntr_variants", 54, "_more", "10-70.pickle"),
        ("ntr_variants_62R", 62, "", "10-30.pickle"),
        ("ntr_variants_70R", 70, "", "10-30.pickle"),
    ]
    
    sites_list: List[int] = [2, 4, 6]
    radius_list: List[int] = [10, 14, 18, 22, 26]
    num_sim_folders: int = 30
    
    DATA_DIR: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/023_EXP_stair"
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
                base_coords_dir: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/1_single_sim_kap_coords"
                
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
if __name__ == "__main__":
    results_path: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/023_EXP_stair/results.pickle"
    
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results_data = pickle.load(f)
            
        # Extract the z=0 energy directly from the saved landscape files if not already present
        for item in results_data:
            if "z0_free_energy" not in item:
                landscape_path = item.get("landscape_file")
                if landscape_path and os.path.exists(landscape_path):
                    with open(landscape_path, "rb") as lf:
                        lf_data = pickle.load(lf)
                        fe_1d = lf_data["free_energy_1d"]
                        z_centers = lf_data["z_centers"]
                        z0_idx = np.argmin(np.abs(z_centers - 0.0))
                        item["z0_free_energy"] = float(fe_1d[z0_idx])
                        
        # Get sorted unique sites and diameters
        unique_sites: List[int] = sorted(list(set(item["sites"] for item in results_data if "z0_free_energy" in item)))
        unique_diams: List[int] = sorted(list(set(item["diameter"] for item in results_data if "z0_free_energy" in item)))
        num_subplots: int = len(unique_sites)
        
        if num_subplots == 0:
            print("No results found for 1D free energy at z=0.")
        else:
            fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4.5), sharey=True)
            if num_subplots == 1:
                axes = [axes]
                
            for idx, num_sites in enumerate(unique_sites):
                ax = axes[idx]
                site_results = [item for item in results_data if item["sites"] == num_sites and "z0_free_energy" in item]
                
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
                    
                    log_mws = np.log10(filtered_mws)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_mws, filtered_energies)
                    
                    ax.scatter(filtered_mws, filtered_energies, marker="o", s=40)
                    
                    fit_mws = np.linspace(min(filtered_mws), max(filtered_mws), 100)
                    fit_energies = slope * np.log10(fit_mws) + intercept
                    ax.plot(fit_mws, fit_energies, linestyle="--", linewidth=1.5,
                            label=f"{diam} nm (R²={r_value**2:.2f})")
                    
                ax.set_title(f"{num_sites} Sites", fontsize=12, fontweight="bold")
                ax.set_xlabel("Molecular Weight (kDa)", fontsize=10)
                ax.set_xscale("log")
                
                if idx == 0:
                    ax.set_ylabel("1D Free Energy at z=0 (kBT)", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(title="Diameter")
                
            plt.suptitle("1D Free Energy at z=0 vs Molecular Weight", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            
            plot_output_path: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/023_EXP_stair_z0_plot.png"
            os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
            plt.savefig(plot_output_path, bbox_inches="tight", dpi=300)
            print(f"\nPlot saved successfully to: {plot_output_path}")
            plt.show()

            # --- New Plot: All 1D Landscapes ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            
            sites_to_idx: Dict[int, int] = {2: 0, 4: 1, 6: 2}
            plotted_diams_per_ax: Dict[int, set] = {0: set(), 1: set(), 2: set()}
            
            for item in results_data:
                landscape_path = item.get("landscape_file")
                if not landscape_path or not os.path.exists(landscape_path):
                    continue
                
                sites = item["sites"]
                if sites not in sites_to_idx:
                    continue
                ax_idx = sites_to_idx[sites]
                ax = axes[ax_idx]
                
                with open(landscape_path, "rb") as lf:
                    lf_data = pickle.load(lf)
                    fe_1d = lf_data["free_energy_1d"]
                    z_centers = lf_data["z_centers"]
                
                # Smooth the 1D energy landscape with a Gaussian kernel (2nm std)
                # Since the bin size is 1 nm, sigma=2.0 represents 2 nm std.
                nan_mask = np.isnan(fe_1d)
                if np.any(nan_mask):
                    non_nan_indices = np.where(~nan_mask)[0]
                    nan_indices = np.where(nan_mask)[0]
                    fe_1d_filled = np.copy(fe_1d)
                    if len(non_nan_indices) > 0:
                        fe_1d_filled[nan_mask] = np.interp(nan_indices, non_nan_indices, fe_1d[non_nan_indices])
                    fe_1d_smoothed = gaussian_filter1d(fe_1d_filled, sigma=2.0)
                    fe_1d_smoothed[nan_mask] = np.nan
                else:
                    fe_1d_smoothed = gaussian_filter1d(fe_1d, sigma=2.0)
                
                diam = item["diameter"]
                radius = item["radius"]
                
                # Get the custom color based on diameter and radius
                color = get_variant_color(
                    diameter=diam,
                    radius=float(radius),
                    min_r=10.0,
                    max_r=26.0,
                )
                
                label = f"{diam} nm" if diam not in plotted_diams_per_ax[ax_idx] else None
                plotted_diams_per_ax[ax_idx].add(diam)
                
                ax.plot(z_centers, fe_1d_smoothed, color=color, alpha=0.7, linewidth=1.2, label=label)
            
            for sites, ax_idx in sites_to_idx.items():
                ax = axes[ax_idx]
                ax.set_xlabel("z (nm)", fontsize=11)
                if ax_idx == 0:
                    ax.set_ylabel("Free Energy (kBT)", fontsize=11)
                ax.set_title(f"{sites} Sites", fontsize=13, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.5)
                
                # Sort legend handles to keep them consistent (46, 54, 62, 70)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    sorted_pairs = sorted(zip(handles, labels), key=lambda x: int(x[1].split()[0]))
                    shandles, slabels = zip(*sorted_pairs)
                    ax.legend(shandles, slabels, title="NPC Diameter")
            
            plt.suptitle("1D Free Energy Landscapes per Site Variant", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            
            all_1d_output_path = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots/023_EXP_stair_all_1d_landscapes.png"
            plt.savefig(all_1d_output_path, bbox_inches="tight", dpi=300)
            print(f"All 1D landscapes plot saved to: {all_1d_output_path}")
            plt.show()
    else:
        print(f"Results file not found for plotting: {results_path}")
