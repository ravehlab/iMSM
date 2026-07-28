# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from typing import List, Tuple, Dict, Any
from iMSM.extensions.npc.npc_utils import radius_a_to_kda, z_order_get_only_state_to_idx_dict_nmc
from iMSM.extensions.npc.symkmeans import SKM

def get_stationary_distribution(P: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    """
    Computes the stationary distribution of a transition matrix P using the power method.
    """
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.allclose(pi, pi_next, atol=tol):
            break
        pi = pi_next
    return pi

def simulate_msm_trajectories(P: np.ndarray, n_diffusers: int, n_steps: int) -> np.ndarray:
    """
    Simulates n_diffusers trajectories of length n_steps using transition matrix P.
    Starting states are sampled from the stationary distribution.
    """
    n_states = P.shape[0]
    pi = get_stationary_distribution(P, 1000, 1e-10)
    
    # Sample initial states
    states = np.random.choice(n_states, size=n_diffusers, p=pi)
    
    trajectories = np.zeros((n_diffusers, n_steps), dtype=int)
    trajectories[:, 0] = states
    
    # Cumulative transition matrix for fast sampling
    P_cum = np.cumsum(P, axis=1)
    # Ensure the last column is exactly 1.0 to avoid precision issues
    P_cum[:, -1] = 1.0
    
    for t in range(1, n_steps):
        r = np.random.random(n_diffusers)
        for d in range(n_diffusers):
            current_state = trajectories[d, t-1]
            trajectories[d, t] = np.searchsorted(P_cum[current_state], r[d])
            
    return trajectories

def compute_msm_verticality_score(P: np.ndarray, centroids: np.ndarray, spoke_mapping_matrix: np.ndarray, n_diffusers: int, n_steps: int, threshold: float) -> Tuple[float, float, float]:
    """
    Simulates trajectories using transition matrix P and maps back to spokedness using centroids.
    Returns a tuple of (verticality_score, average_purity, average_similarity).
    """
    trajectories: np.ndarray = simulate_msm_trajectories(P, n_diffusers, n_steps)
    
    # spoke_mapping_matrix: (8, 458), centroids: (n_clusters, 458)
    centroid_spokedness: np.ndarray = np.dot(spoke_mapping_matrix, centroids.T) # shape: (8, n_clusters)
    
    # Map trajectories to spokedness of shape (n_diffusers, n_steps, 8)
    spokedness: np.ndarray = np.zeros((n_diffusers, n_steps, 8))
    for d in range(n_diffusers):
        for t in range(n_steps):
            state: int = int(trajectories[d, t])
            spokedness[d, t, :] = centroid_spokedness[:, state]
            
    return compute_verticality_score(spokedness, threshold)

# Set random seed for reproducibility in sampling
np.random.seed(42)

def load_embedded_coordinates(file_path: str) -> np.ndarray:
    """
    Loads embedded coordinate file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required embedded coordinate file not found: {file_path}")
        
    with open(file_path, "rb") as f:
        coords: np.ndarray = pickle.load(f)
    return coords

def get_spoke_mapping_matrix(state_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Creates an (8, n_states) mapping matrix. Each spoke row contains 1.0 for
    states (Nups and channels) belonging to that spoke, and 0.0 otherwise.
    """
    n_states = len(state_to_idx)
    mapping = np.zeros((8, n_states))
    for state, idx in state_to_idx.items():
        parts = state.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            spoke = int(parts[-1]) % 8
            mapping[spoke, idx] = 1.0
    return mapping

def circular_emd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Wasserstein-1 distance between two histograms on a cycle of 8 bins.
    Adjacent bins are distance 1 apart, and bin 7 is adjacent to bin 0.
    Returns the distance in units of 'spoke steps'.
    """
    p_arr: np.ndarray = np.asarray(p, dtype=float)
    q_arr: np.ndarray = np.asarray(q, dtype=float)
    p_norm: np.ndarray = p_arr / p_arr.sum()
    q_norm: np.ndarray = q_arr / q_arr.sum()
    F: np.ndarray = np.cumsum(p_norm - q_norm)
    return float(np.sum(np.abs(F - np.median(F))))

def compute_purity(x: np.ndarray) -> float:
    """
    Computes the purity of a distribution on 8 bins.
    Concentration in up to 2 ADJACENT bins (e.g., [0.5, 0.5] or [1.0, 0.0]) is 
    considered fully pure (1.0), and a completely flat distribution (0.125 each) is 0.0.
    
    Uses the maximum sum of any two adjacent elements on the cycle, scaled from 
    [0.25, 1.0] to [0.0, 1.0].
    """
    # Calculate sum of adjacent pairs on the cycle
    adj_sums: np.ndarray = x + np.roll(x, -1)
    max_adj_sum: float = float(np.max(adj_sums))
    purity: float = (max_adj_sum - 0.25) / 0.75
    return max(0.0, min(1.0, purity))

def compute_transition_score(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the transition score between two probability distributions p and q.
    
    The score is defined as:
        score = purity(p) * purity(q) * similarity(p, q)
    where:
        purity(x) is computed via compute_purity(x).
            Ranges from 0.0 (fully flat) to 1.0 (concentrated in <= 2 spokes).
        similarity(p, q) = 1.0 - circular_emd(p, q) / 4.0
            Ranges from 0.0 (opposite sides of the circle) to 1.0 (identical distribution).
            
    Therefore, the score ranges from:
        - Max: 1.0 (when p and q are identical distributions concentrated in <= 2 spokes)
        - Min: 0.0 (when p and q are delta distributions on opposite sides of the circle,
                    or any distribution with 0.0 purity)
    """
    purity_p: float = compute_purity(p)
    purity_q: float = compute_purity(q)
    dist: float = circular_emd(p, q)
    similarity: float = 1.0 - (dist / 4.0)
    return purity_p * purity_q * similarity

def compute_verticality_score(spokedness: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    Computes the verticality score, average purity, and average similarity from the spokedness array of shape (n_diffusers, n_sections, 8).
    For each diffuser, calculates:
      - purity: average purity of all valid steps.
      - similarity: average similarity of consecutive valid steps.
      - score: average transition score of consecutive valid steps.
    
    Returns a tuple of (average_score, average_purity, average_similarity) across all diffusers.
    """
    n_diffusers: int = spokedness.shape[0]
    n_sections: int = spokedness.shape[1]
    
    diffuser_scores: List[float] = []
    diffuser_purities: List[float] = []
    diffuser_similarities: List[float] = []
    
    for d in range(n_diffusers):
        valid_vectors: List[np.ndarray] = []
        valid_indices: List[int] = []
        
        for t in range(n_sections):
            s_t: np.ndarray = spokedness[d, t]
            w_t: float = float(np.sum(s_t))
            if w_t >= threshold:
                v_t: np.ndarray = s_t / w_t
                valid_vectors.append(v_t)
                valid_indices.append(t)
                
        if not valid_vectors:
            continue
            
        purity_vals: List[float] = [compute_purity(v) for v in valid_vectors]
        diffuser_purities.append(float(np.mean(purity_vals)))
        
        scores: List[float] = []
        similarities: List[float] = []
        
        for i in range(len(valid_vectors) - 1):
            if valid_indices[i+1] == valid_indices[i] + 1:
                v1: np.ndarray = valid_vectors[i]
                v2: np.ndarray = valid_vectors[i+1]
                p1: float = purity_vals[i]
                p2: float = purity_vals[i+1]
                
                dist: float = circular_emd(v1, v2)
                sim: float = 1.0 - (dist / 4.0)
                score: float = p1 * p2 * sim
                
                similarities.append(sim)
                scores.append(score)
                
        if similarities:
            diffuser_scores.append(float(np.mean(scores)))
            diffuser_similarities.append(float(np.mean(similarities)))
            
    final_score: float = float(np.mean(diffuser_scores)) if diffuser_scores else 0.0
    final_purity: float = float(np.mean(diffuser_purities)) if diffuser_purities else 0.0
    final_similarity: float = float(np.mean(diffuser_similarities)) if diffuser_similarities else 0.0
    
    return final_score, final_purity, final_similarity


def load_cartesian_spokedness(
    base_dir: str,
    num_folders: int,
    filename: str,
    max_frames: int,
    exclude_folders: List[int]
) -> np.ndarray:
    """
    Loads coordinate files, computes spokedness over 50-frame periods using the angle on the xy plane,
    and returns a spokedness array of shape (n_diffusers, n_sections, 8).
    Out-of-pore frames (|z| > 15) are ignored and do not contribute to spokedness.
    """
    spokedness_list: List[np.ndarray] = []
    
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
            
            # Slice frame dimension
            if coords.shape[-1] > max_frames:
                coords = coords[:, :, :max_frames]
                
            # Transpose to (kap_amount, n_frames, 3)
            coords = coords.transpose(0, 2, 1)
            kap_amount, n_frames, _ = coords.shape
            
            n_sections = max_frames // 50
            # spokedness for this folder: (kap_amount, n_sections, 8)
            folder_spokedness = np.zeros((kap_amount, n_sections, 8))
            
            for d in range(kap_amount):
                for s in range(n_sections):
                    section_coords = coords[d, s * 50:(s + 1) * 50]
                    # Filter: in the pore (|z| <= 15)
                    z = section_coords[:, 2]
                    in_pore = (z >= -15.0) & (z <= 15.0)
                    valid_coords = section_coords[in_pore]
                    
                    if len(valid_coords) > 0:
                        x = valid_coords[:, 0]
                        y = valid_coords[:, 1]
                        theta = np.arctan2(y, x)
                        spoke_idx = np.round(theta / (np.pi / 4)).astype(int) % 8
                        # Count spokes
                        counts = np.bincount(spoke_idx, minlength=8)
                        folder_spokedness[d, s, :] = counts / 50.0
                    else:
                        folder_spokedness[d, s, :] = 0.0
                        
            spokedness_list.append(folder_spokedness)
            
    return np.concatenate(spokedness_list, axis=0)


# %%
# MD Cartesian analysis block
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
    
    cartesian_results_list: List[Dict[str, Any]] = []
    
    print("Starting processing of NTR variants for Cartesian verticality...")
    
    for dir_name, diameter, suffix, pickle_name in configs:
        print(f"\n==================================================")
        print(f"Processing Cartesian diameter {diameter} nm in {dir_name}")
        print(f"==================================================")
        
        for sites in sites_list:
            for radius in radius_list:
                mol_name: str = f"{sites}_{radius}{suffix}"
                unique_name: str = f"{diameter}nm_{sites}_{radius}{suffix}"
                base_coords_dir: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/1_single_sim_kap_coords"
                
                if not os.path.exists(base_coords_dir):
                    print(f"Skipping {mol_name} (directory does not exist: {base_coords_dir})")
                    continue
                
                print(f"\n--- Loading Cartesian coordinates for molecule {unique_name} ---")
                exclude: List[int] = []
                if diameter == 70 and sites == 2:
                    exclude = [1, 2, 3, 4, 21, 22]
                try:
                    spokedness = load_cartesian_spokedness(
                        base_dir=base_coords_dir,
                        num_folders=num_sim_folders,
                        filename=pickle_name,
                        max_frames=200,
                        exclude_folders=exclude
                    )
                    
                    # Compute verticality score
                    vert_score, purity_score, similarity_score = compute_verticality_score(spokedness, threshold=0.1)
                    print(f"[{unique_name}] Cartesian Verticality score: {vert_score:.6f} (Purity: {purity_score:.6f}, Similarity: {similarity_score:.6f})")
                    
                    cartesian_results_list.append({
                        "diameter": diameter,
                        "sites": sites,
                        "radius": radius,
                        "molecule_name": unique_name,
                        "verticality_score": vert_score,
                        "purity_score": purity_score,
                        "similarity_score": similarity_score
                    })
                except Exception as e:
                    print(f"Error computing Cartesian coordinates for {unique_name}: {e}")
                    continue

    DATA_DIR: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality"
    os.makedirs(DATA_DIR, exist_ok=True)
    cartesian_results_path: str = os.path.join(DATA_DIR, "cartesian_results.pickle")
    with open(cartesian_results_path, "wb") as f:
        pickle.dump(cartesian_results_list, f)
    print(f"\nCartesian Results saved successfully to: {cartesian_results_path}")


# %% MD analysis block

if __name__ == "__main__":
    configs: List[Tuple[str, int, str, str]] = [
        ("ntr_variants_46R", 46, "", "10-30.pickle"),
        ("ntr_variants", 54, "_more", "10-70.pickle"),
        ("ntr_variants_62R", 62, "", "10-30.pickle"),
        ("ntr_variants_70R", 70, "", "10-30.pickle"),
    ]
    
    sites_list: List[int] = [2, 4, 6]
    radius_list: List[int] = [10, 14, 18, 22, 26]
    
    state_to_idx_dict = z_order_get_only_state_to_idx_dict_nmc()
    spoke_mapping_matrix = get_spoke_mapping_matrix(state_to_idx_dict)
    
    results_list: List[Dict[str, Any]] = []
    
    print("Starting processing of NTR variants to aggregate embedded coordinates...")
    
    for dir_name, diameter, suffix, pickle_name in configs:
        print(f"\n==================================================")
        print(f"Processing diameter {diameter} nm in {dir_name}")
        print(f"==================================================")
        
        for sites in sites_list:
            for radius in radius_list:
                mol_name: str = f"{sites}_{radius}{suffix}"
                unique_name: str = f"{diameter}nm_{sites}_{radius}{suffix}"
                embedded_file_path: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/4_embedded.pickle"
                
                if not os.path.exists(embedded_file_path):
                    print(f"Skipping {mol_name} (file does not exist: {embedded_file_path})")
                    continue
                
                print(f"\n--- Loading embedded coordinates for molecule {unique_name} ---")
                try:
                    embedded_coords: np.ndarray = load_embedded_coordinates(file_path=embedded_file_path)
                    
                    # Compute current spokedness vector of shape (n_diffusers, n_sections, 8)
                    current_spokedness: np.ndarray = np.tensordot(
                        spoke_mapping_matrix, embedded_coords, axes=(1, 1)
                    ).transpose(1, 2, 0)
                    
                    # Compute verticality score
                    vert_score, purity_score, similarity_score = compute_verticality_score(current_spokedness, threshold=0.1)
                    print(f"[{unique_name}] Verticality score: {vert_score:.6f} (Purity: {purity_score:.6f}, Similarity: {similarity_score:.6f})")
                    
                    results_list.append({
                        "diameter": diameter,
                        "sites": sites,
                        "radius": radius,
                        "molecule_name": unique_name,
                        "verticality_score": vert_score,
                        "purity_score": purity_score,
                        "similarity_score": similarity_score
                    })
                except Exception as e:
                    print(f"Error loading coordinates for {unique_name}: {e}")
                    continue

    DATA_DIR: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality"
    os.makedirs(DATA_DIR, exist_ok=True)
    results_path: str = os.path.join(DATA_DIR, "results.pickle")
    with open(results_path, "wb") as f:
        pickle.dump(results_list, f)
    print(f"\nResults saved successfully to: {results_path}")

# %% iMSM analysis block
if __name__ == "__main__":
    configs: List[Tuple[str, int, str, str]] = [
        ("ntr_variants_46R", 46, "", "10-30.pickle"),
        ("ntr_variants", 54, "_more", "10-70.pickle"),
        ("ntr_variants_62R", 62, "", "10-30.pickle"),
        ("ntr_variants_70R", 70, "", "10-30.pickle"),
    ]
    
    sites_list: List[int] = [2, 4, 6]
    radius_list: List[int] = [10, 14, 18, 22, 26]
    
    state_to_idx_dict = z_order_get_only_state_to_idx_dict_nmc()
    spoke_mapping_matrix = get_spoke_mapping_matrix(state_to_idx_dict)
    
    print("\nStarting processing of iMSM transition matrices...")
    imsm_results_list: List[Dict[str, Any]] = []
    
    for dir_name, diameter, suffix, pickle_name in configs:
        print(f"\n==================================================")
        print(f"Processing iMSM for diameter {diameter} nm in {dir_name}")
        print(f"==================================================")
        
        for sites in sites_list:
            for radius in radius_list:
                mol_name: str = f"{sites}_{radius}{suffix}"
                unique_name: str = f"{diameter}nm_{sites}_{radius}{suffix}"
                
                # Paths
                tm_path: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/6_transition_matrices_subsets/1.00fraction/0index/320clusters.pickle"
                clustering_path: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/5_clustering_subsets/1.00fraction/0index/320clusters.pickle"
                
                if not os.path.exists(tm_path) or not os.path.exists(clustering_path):
                    print(f"Skipping iMSM {mol_name} (TM or clustering file does not exist)")
                    continue
                
                print(f"\n--- Running iMSM simulation for molecule {unique_name} ---")
                try:
                    with open(tm_path, "rb") as f:
                        tm: np.ndarray = pickle.load(f)
                    
                    with open(clustering_path, "rb") as f:
                        clustering: SKM = pickle.load(f)
                    
                    centroids: np.ndarray = clustering.centroids
                    
                    # Run MSM trajectory simulations and compute verticality
                    vert_score, purity_score, similarity_score = compute_msm_verticality_score(
                        P=tm,
                        centroids=centroids,
                        spoke_mapping_matrix=spoke_mapping_matrix,
                        n_diffusers=100,
                        n_steps=200,
                        threshold=0.1
                    )
                    
                    print(f"[iMSM - {unique_name}] Verticality score: {vert_score:.6f} (Purity: {purity_score:.6f}, Similarity: {similarity_score:.6f})")
                    
                    imsm_results_list.append({
                        "diameter": diameter,
                        "sites": sites,
                        "radius": radius,
                        "molecule_name": unique_name,
                        "verticality_score": vert_score,
                        "purity_score": purity_score,
                        "similarity_score": similarity_score
                    })
                except Exception as e:
                    print(f"Error simulating iMSM for {unique_name}: {e}")
                    continue
                    
    DATA_DIR: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality"
    imsm_results_path: str = os.path.join(DATA_DIR, "imsm_results.pickle")
    with open(imsm_results_path, "wb") as f:
        pickle.dump(imsm_results_list, f)
    print(f"\niMSM Results saved successfully to: {imsm_results_path}")


# %% Plotting results
if __name__ == "__main__":
    cartesian_results_path: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality/cartesian_results.pickle"
    results_path: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality/results.pickle"
    imsm_results_path: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality/imsm_results.pickle"
    
    if os.path.exists(cartesian_results_path) and os.path.exists(results_path) and os.path.exists(imsm_results_path):
        with open(cartesian_results_path, "rb") as f:
            cartesian_results_list = pickle.load(f)
        with open(results_path, "rb") as f:
            results_list = pickle.load(f)
        with open(imsm_results_path, "rb") as f:
            imsm_results_list = pickle.load(f)
            
        unique_sites = sorted(list(set(item["sites"] for item in results_list)))
        unique_diams = sorted(list(set(item["diameter"] for item in results_list)))
        num_subplots = len(unique_sites)
        
        if num_subplots > 0:
            fig, axes = plt.subplots(3, num_subplots, figsize=(5 * num_subplots, 13.5), sharey=True)
            if num_subplots == 1:
                axes = np.array([axes])
                
            for idx, num_sites in enumerate(unique_sites):
                # Row 0: MD Cartesian
                ax_cart = axes[0, idx]
                ax_cart.axhline(1.0, color="gray", linestyle="-.", alpha=0.5, label="Max (1.0)")
                cart_site_results = [item for item in cartesian_results_list if item["sites"] == num_sites]
                for diam in unique_diams:
                    diam_site_results = [item for item in cart_site_results if item["diameter"] == diam]
                    if not diam_site_results:
                        continue
                    diam_site_results = sorted(diam_site_results, key=lambda x: x["radius"])
                    mws = [float(radius_a_to_kda(item["radius"])) for item in diam_site_results]
                    scores = [item["verticality_score"] for item in diam_site_results]
                    purities = [item.get("purity_score", np.nan) for item in diam_site_results]
                    similarities = [item.get("similarity_score", np.nan) for item in diam_site_results]
                    
                    line, = ax_cart.plot(mws, scores, linestyle="-", linewidth=1.8, label=f"{diam} nm")
                    color = line.get_color()
                    ax_cart.scatter(mws, scores, marker="^", s=40, color=color)
                    
                    if not np.any(np.isnan(purities)):
                        ax_cart.plot(mws, purities, linestyle="--", linewidth=1.2, color=color, alpha=0.7)
                    if not np.any(np.isnan(similarities)):
                        ax_cart.plot(mws, similarities, linestyle=":", linewidth=1.2, color=color, alpha=0.7)
                        
                ax_cart.set_title(f"MD Cartesian - {num_sites} Sites", fontsize=12, fontweight="bold")
                ax_cart.set_xscale("log")
                ax_cart.set_ylim(0.0, 1.05)
                if idx == 0:
                    ax_cart.set_ylabel("Verticality Score", fontsize=10)
                ax_cart.grid(True, linestyle="--", alpha=0.5)
                
                # Combined legend
                handles, labels = ax_cart.get_legend_handles_labels()
                metric_handles = [
                    Line2D([0], [0], color="black", linestyle="-", label="Score"),
                    Line2D([0], [0], color="black", linestyle="--", label="Purity"),
                    Line2D([0], [0], color="black", linestyle=":", label="Similarity")
                ]
                ax_cart.legend(handles=handles + metric_handles, title="Diameter & Metrics", loc="lower left", fontsize=8)

                # Row 1: MD Interaction
                ax_md = axes[1, idx]
                ax_md.axhline(1.0, color="gray", linestyle="-.", alpha=0.5, label="Max (1.0)")
                site_results = [item for item in results_list if item["sites"] == num_sites]
                for diam in unique_diams:
                    diam_site_results = [item for item in site_results if item["diameter"] == diam]
                    if not diam_site_results:
                        continue
                    diam_site_results = sorted(diam_site_results, key=lambda x: x["radius"])
                    mws = [float(radius_a_to_kda(item["radius"])) for item in diam_site_results]
                    scores = [item["verticality_score"] for item in diam_site_results]
                    purities = [item.get("purity_score", np.nan) for item in diam_site_results]
                    similarities = [item.get("similarity_score", np.nan) for item in diam_site_results]
                    
                    line, = ax_md.plot(mws, scores, linestyle="-", linewidth=1.8, label=f"{diam} nm")
                    color = line.get_color()
                    ax_md.scatter(mws, scores, marker="o", s=40, color=color)
                    
                    if not np.any(np.isnan(purities)):
                        ax_md.plot(mws, purities, linestyle="--", linewidth=1.2, color=color, alpha=0.7)
                    if not np.any(np.isnan(similarities)):
                        ax_md.plot(mws, similarities, linestyle=":", linewidth=1.2, color=color, alpha=0.7)
                        
                ax_md.set_title(f"MD Interaction - {num_sites} Sites", fontsize=12, fontweight="bold")
                ax_md.set_xscale("log")
                ax_md.set_ylim(0.0, 1.05)
                if idx == 0:
                    ax_md.set_ylabel("Verticality Score", fontsize=10)
                ax_md.grid(True, linestyle="--", alpha=0.5)
                
                # Combined legend
                handles, labels = ax_md.get_legend_handles_labels()
                metric_handles = [
                    Line2D([0], [0], color="black", linestyle="-", label="Score"),
                    Line2D([0], [0], color="black", linestyle="--", label="Purity"),
                    Line2D([0], [0], color="black", linestyle=":", label="Similarity")
                ]
                ax_md.legend(handles=handles + metric_handles, title="Diameter & Metrics", loc="lower left", fontsize=8)

                # Row 2: iMSM
                ax_imsm = axes[2, idx]
                ax_imsm.axhline(1.0, color="gray", linestyle="-.", alpha=0.5, label="Max (1.0)")
                imsm_site_results = [item for item in imsm_results_list if item["sites"] == num_sites]
                for diam in unique_diams:
                    diam_site_results = [item for item in imsm_site_results if item["diameter"] == diam]
                    if not diam_site_results:
                        continue
                    diam_site_results = sorted(diam_site_results, key=lambda x: x["radius"])
                    mws = [float(radius_a_to_kda(item["radius"])) for item in diam_site_results]
                    scores = [item["verticality_score"] for item in diam_site_results]
                    purities = [item.get("purity_score", np.nan) for item in diam_site_results]
                    similarities = [item.get("similarity_score", np.nan) for item in diam_site_results]
                    
                    line, = ax_imsm.plot(mws, scores, linestyle="-", linewidth=1.8, label=f"{diam} nm")
                    color = line.get_color()
                    ax_imsm.scatter(mws, scores, marker="s", s=40, color=color)
                    
                    if not np.any(np.isnan(purities)):
                        ax_imsm.plot(mws, purities, linestyle="--", linewidth=1.2, color=color, alpha=0.7)
                    if not np.any(np.isnan(similarities)):
                        ax_imsm.plot(mws, similarities, linestyle=":", linewidth=1.2, color=color, alpha=0.7)
                        
                ax_imsm.set_title(f"iMSM - {num_sites} Sites", fontsize=12, fontweight="bold")
                ax_imsm.set_xlabel("Molecular Weight (kDa)", fontsize=10)
                ax_imsm.set_xscale("log")
                ax_imsm.set_ylim(0.0, 1.05)
                if idx == 0:
                    ax_imsm.set_ylabel("Verticality Score", fontsize=10)
                ax_imsm.grid(True, linestyle="--", alpha=0.5)
                
                # Combined legend
                handles, labels = ax_imsm.get_legend_handles_labels()
                metric_handles = [
                    Line2D([0], [0], color="black", linestyle="-", label="Score"),
                    Line2D([0], [0], color="black", linestyle="--", label="Purity"),
                    Line2D([0], [0], color="black", linestyle=":", label="Similarity")
                ]
                ax_imsm.legend(handles=handles + metric_handles, title="Diameter & Metrics", loc="lower left", fontsize=8)
                
            plt.suptitle("Verticality Score vs Molecular Weight (MD Cartesian vs MD Interaction vs iMSM)", fontsize=14, fontweight="bold", y=0.98)
            plt.tight_layout()
            
            plot_output_dir = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots"
            os.makedirs(plot_output_dir, exist_ok=True)
            plot_output_path = os.path.join(plot_output_dir, "024_EXP_new_verticality_plot.png")
            plt.savefig(plot_output_path, bbox_inches="tight", dpi=300)
            print(f"Plot saved successfully to: {plot_output_path}")
            plt.show()
    else:
        print(f"Results files not found for plotting: {cartesian_results_path}, {results_path} or {imsm_results_path}")


# %% iMSM Successful vs Aborted Transport Time Analysis

def simulate_transports(
    P: np.ndarray,
    centroids: np.ndarray,
    n_target: int,
    max_attempts: int,
    max_steps: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Simulates transports starting at the cytoplasm (cyt component > 0.9) and continuing at least until they leave the cytoplasm.
    Returns (successful_paths, aborted_paths).
    """
    is_cyto: np.ndarray = centroids[:, -1] > 0.9
    is_nuc: np.ndarray = centroids[:, 0] > 0.9
    
    cyto_states: np.ndarray = np.where(is_cyto)[0]
    if len(cyto_states) == 0 or not np.any(is_nuc):
        return [], []
    
    pi: np.ndarray = get_stationary_distribution(P, 1000, 1e-10)
    pi_cyto: np.ndarray = pi[cyto_states]
    if pi_cyto.sum() > 0:
        pi_cyto = pi_cyto / pi_cyto.sum()
    else:
        pi_cyto = np.ones(len(cyto_states)) / len(cyto_states)
        
    successful_paths: List[np.ndarray] = []
    aborted_paths: List[np.ndarray] = []
    
    P_cum: np.ndarray = np.cumsum(P, axis=1)
    P_cum[:, -1] = 1.0
    
    attempts: int = 0
    while (len(successful_paths) < n_target or len(aborted_paths) < n_target) and attempts < max_attempts:
        attempts += 1
        s: int = int(np.random.choice(cyto_states, p=pi_cyto))
        path: List[int] = [s]
        
        steps: int = 0
        # 1. Continue at least until we leave the cytoplasm
        while is_cyto[s] and steps < max_steps:
            r: float = float(np.random.random())
            s = int(np.searchsorted(P_cum[s], r))
            path.append(s)
            steps += 1
            
        # 2. Continue until we reach nucleus or cytoplasm
        success: bool = False
        aborted: bool = False
        while steps < max_steps:
            if is_nuc[s]:
                success = True
                break
            if is_cyto[s]:
                aborted = True
                break
            r = float(np.random.random())
            s = int(np.searchsorted(P_cum[s], r))
            path.append(s)
            steps += 1
            
        if success and len(successful_paths) < n_target:
            successful_paths.append(np.array(path))
        elif aborted and len(aborted_paths) < n_target:
            aborted_paths.append(np.array(path))
            
    return successful_paths, aborted_paths


if __name__ == "__main__":
    # Toggles for plotting and filtering
    show_only_2_and_4: bool = False
    plot_metric: str = "verticality/time"  # Options: "time", "verticality", "verticality/time"
    
    configs: List[Tuple[str, int, str, str]] = [
        ("ntr_variants_46R", 46, "", "10-30.pickle"),
        ("ntr_variants", 54, "_more", "10-70.pickle"),
        ("ntr_variants_62R", 62, "", "10-30.pickle"),
        ("ntr_variants_70R", 70, "", "10-30.pickle"),
    ]
    
    sites_list: List[int] = [2, 4, 6]
    radius_list: List[int] = [10, 14, 18, 22, 26]
    
    state_to_idx_dict: Dict[str, int] = z_order_get_only_state_to_idx_dict_nmc()
    spoke_mapping_matrix: np.ndarray = get_spoke_mapping_matrix(state_to_idx_dict)
    
    DATA_DIR: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/024_EXP_new_verticality"
    transport_times_path: str = os.path.join(DATA_DIR, "transport_times.pickle")
    imsm_results_path: str = os.path.join(DATA_DIR, "imsm_results.pickle")
    
    # Load the overall iMSM verticality results computed earlier
    if os.path.exists(imsm_results_path):
        with open(imsm_results_path, "rb") as f:
            imsm_results_list = pickle.load(f)
    else:
        imsm_results_list = []
        
    # Load or simulate transport times
    if not os.path.exists(transport_times_path):
        print("\nStarting simulation of successful vs aborted transports for time analysis...")
        transport_times_list: List[Dict[str, Any]] = []
        
        for dir_name, diameter, suffix, pickle_name in configs:
            print(f"\n==================================================")
            print(f"Simulating transports for diameter {diameter} nm")
            print(f"==================================================")
            
            for sites in sites_list:
                for radius in radius_list:
                    mol_name: str = f"{sites}_{radius}{suffix}"
                    unique_name: str = f"{diameter}nm_{sites}_{radius}{suffix}"
                    
                    tm_path: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/6_transition_matrices_subsets/1.00fraction/0index/320clusters.pickle"
                    clustering_path: str = f"/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/data/{dir_name}/{mol_name}/5_clustering_subsets/1.00fraction/0index/320clusters.pickle"
                    
                    if not os.path.exists(tm_path) or not os.path.exists(clustering_path):
                        print(f"Skipping {mol_name} (TM or clustering file missing)")
                        continue
                    
                    print(f"--- Running transport simulation for {unique_name} ---")
                    try:
                        with open(tm_path, "rb") as f:
                            tm: np.ndarray = pickle.load(f)
                        with open(clustering_path, "rb") as f:
                            clustering: SKM = pickle.load(f)
                            
                        centroids: np.ndarray = clustering.centroids
                        is_cyto: np.ndarray = centroids[:, -1] > 0.9
                        is_nuc: np.ndarray = centroids[:, 0] > 0.9
                        
                        # Generate successful and aborted transports
                        success_paths, aborted_paths = simulate_transports(tm, centroids, 100, 10000, 20000)
                        print(f"Generated {len(success_paths)} successful and {len(aborted_paths)} aborted paths")
                        
                        # Compute times in pore (1 step = 5 us)
                        success_times = [
                            sum(1 for s in path if not is_cyto[s] and not is_nuc[s]) * 5.0 for path in success_paths
                        ]
                        aborted_times = [
                            sum(1 for s in path if not is_cyto[s] and not is_nuc[s]) * 5.0 for path in aborted_paths
                        ]
                        
                        s_mean: float = float(np.mean(success_times)) if success_times else np.nan
                        a_mean: float = float(np.mean(aborted_times)) if aborted_times else np.nan
                        
                        print(f"[{unique_name}] Successful mean time: {s_mean:.2f} us")
                        print(f"[{unique_name}] Aborted mean time: {a_mean:.2f} us")
                        
                        transport_times_list.append({
                            "diameter": diameter,
                            "sites": sites,
                            "radius": radius,
                            "molecule_name": unique_name,
                            "successful_mean_time": s_mean,
                            "aborted_mean_time": a_mean
                        })
                    except Exception as e:
                        print(f"Error simulating transports for {unique_name}: {e}")
                        continue
                        
        with open(transport_times_path, "wb") as f:
            pickle.dump(transport_times_list, f)
        print(f"\nTransport times saved to: {transport_times_path}")
    else:
        with open(transport_times_path, "rb") as f:
            transport_times_list = pickle.load(f)
            
    # Now plot the panels (2 rows: successful vs aborted; columns: sites)
    unique_sites: List[int] = sorted(list(set(item["sites"] for item in transport_times_list)))
    unique_diams: List[int] = sorted(list(set(item["diameter"] for item in transport_times_list)))
    
    if show_only_2_and_4:
        unique_sites = [s for s in unique_sites if s in (2, 4)]
        
    num_cols = len(unique_sites)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 9), sharey="row")
    categories: List[str] = ["successful", "aborted"]
    
    for row_idx, cat in enumerate(categories):
        for col_idx, num_sites in enumerate(unique_sites):
            # Handle case where there is only one column (1D axes array)
            if num_cols == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            site_results = [item for item in transport_times_list if item["sites"] == num_sites]
            for diam in unique_diams:
                diam_site_results = [item for item in site_results if item["diameter"] == diam]
                if not diam_site_results:
                    continue
                diam_site_results = sorted(diam_site_results, key=lambda x: x["radius"])
                
                mws: List[float] = [float(radius_a_to_kda(item["radius"])) for item in diam_site_results]
                
                if plot_metric == "time":
                    key_name = "successful_mean_time" if cat == "successful" else "aborted_mean_time"
                    values: List[float] = [item[key_name] for item in diam_site_results]
                elif plot_metric == "verticality":
                    # Look up overall verticality score from imsm_results_list
                    values = []
                    for item in diam_site_results:
                        match = next((x for x in imsm_results_list if x["diameter"] == diam and x["sites"] == num_sites and x["radius"] == item["radius"]), None)
                        values.append(match["verticality_score"] if match is not None else np.nan)
                else:  # verticality/time
                    values = []
                    key_name = "successful_mean_time" if cat == "successful" else "aborted_mean_time"
                    for item in diam_site_results:
                        match = next((x for x in imsm_results_list if x["diameter"] == diam and x["sites"] == num_sites and x["radius"] == item["radius"]), None)
                        if match is not None and item[key_name] > 0:
                            values.append(match["verticality_score"] / item[key_name])
                        else:
                            values.append(np.nan)
                
                marker = "s" if cat == "successful" else "o"
                line, = ax.plot(mws, values, linestyle="-", linewidth=1.8, label=f"{diam} nm")
                color = line.get_color()
                ax.scatter(mws, values, marker=marker, s=40, color=color)
                    
            ax.set_title(f"Successful (Unaborted) - {num_sites} Sites" if cat == "successful" else f"Aborted - {num_sites} Sites", fontsize=12, fontweight="bold")
            ax.set_xlim(2, 100)
            ax.set_xscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            if col_idx == 0:
                if plot_metric == "time":
                    y_label = "Average Time in Pore (us)"
                elif plot_metric == "verticality":
                    y_label = "Verticality Score"
                else:
                    y_label = "Verticality / Time (1/us)"
                ax.set_ylabel(y_label, fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Molecular Weight (kDa)", fontsize=10)
                
            if col_idx == 0:
                ax.legend(title="Diameter", loc="upper left" if plot_metric == "time" else "lower left", fontsize=8)
                
    if plot_metric == "time":
        metric_title = "Average Time Spent in Pore"
    elif plot_metric == "verticality":
        metric_title = "Verticality Score"
    else:
        metric_title = "Verticality divided by Time"
    plt.suptitle(f"iMSM {metric_title}: Successful vs Aborted Transports", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plot_output_dir: str = "/cs/usr/roi.eliasian/LabFolder/Master/NPC-markov/plots"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_output_path: str = os.path.join(plot_output_dir, f"024_EXP_new_verticality_transports_{plot_metric.replace('/', '_')}.png")
    plt.savefig(plot_output_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved successfully to: {plot_output_path}")
    plt.show()
