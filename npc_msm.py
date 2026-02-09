import os
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Path formatting helpers
def _cp(params: dict, *parts: object) -> Path:
    """Checkpoint path join helper (keeps stage code readable)."""
    return Path(params["CHECKPOINTS_PATH"]).joinpath(*(str(p) for p in parts))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_range_strings(params: dict) -> tuple[str, str, str]:
    t0_str = f"{int(params['LOAD_MD_START_TIME_NS'] / 1000)}"
    t1_str = f"{int(params['LOAD_MD_END_TIME_NS'] / 1000)}"
    return t0_str, t1_str, f"{t0_str}-{t1_str}"


def _subset_folder(params: dict) -> str:
    mode_string = "_simulations" if params['DATA_SUBSET_MODE'] == "simulation" else ""
    return f"{params['DATA_SUBSET']:.2f}fraction{mode_string}/{params['DATA_SUBSET_INDEX']}index"

###

@dataclass
class iMSMConfig:
    # --- General ---
    checkpoints_path: str
    mode: str = "normal" #options: "normal", "bootstrap", "subset"
    # normal: normal run
    # perform bootstrap resampling to generate transition matrices and to computer permeabilities
    # subset: use only a subset of data for clustering and MSM generation
    
    # --- Bootstrapping ---
    bootstrap_repeats: int = 0 # if n>0 and bootstrap_mode is True, will do bootstrap resampling to generate n matrices (using the same clustering)
    
    # --- Data Subsetting ---
    # If data_subset <1.0, use a data subset of that fractionality for the clustering stage onwards.
    # Will equally collect data from multiple simulations.
    # For example, if there are 30 simulations of length 100us, and data_subset is 0.1, and data_subset_mode is "time", will keep the data_subset_index'th 10us of each simulation.
    # if data_subset_index is None, will take the final 10us.
    # if data_subset_mode is "simulation", will take the data_subset_index'th 10% of simulations fully.
    data_subset: float = 1.0
    data_subset_index: int = -1
    data_subset_mode: str = "time" # options: "time", "simulation"
    
    # --- Stage 1 & 2: Loading ---
    load_md_kap_sites: int = 2
    load_md_kap_radius: int = 20
    load_md_kap_amount: int = 50
    load_md_base_path: str = ""
    load_md_sims_range: range = range(1, 31)
    load_md_start_time_ns: int = 10000
    load_md_end_time_ns: int = 30000
    load_md_step_ns: int = 100
    load_md_ignored_nup_types: List[str] = field(default_factory=list)
    
    # --- Stage 3: Categorization ---
    microstate_k: int = 5
    max_surface_dist_nm: float = 1.0
    split_nc: bool = True
    custom_fg_coords_path: Optional[str] = None # For sims w/ multiple kap types, to not redo loading of FGs
    
    # --- Stage 4: Embedding ---
    window_size_steps: int = 10 # Window time is WINDOW_SIZE_STEPS * LOAD_MD_STEP_NS
    
    # --- Stage 5: Clustering ---
    clustering_type: str = "sym-faiss"
    n_clusters: List[int] = field(default_factory=lambda: [40, 80, 160, 320, 640, 1280])
    
    # --- Stage 6: MSM ---
    tm_prior: float = 0
    
    # --- Stage 7: Stats ---
    state_choice_method: str = "prominent" # options: "prominent", "distance"
    distance_state_threshold_nm: float = 10 # only used if STATE_CHOICE_METHOD is "distance"
    
    def to_legacy_dict(self):
        """Converts snake_case config to the UPPER_CASE dict expected by stages."""
        return {k.upper(): v for k, v in asdict(self).items()}
    
    def override_from_dict(self, override_dict: dict):
        """Override config parameters from a dictionary."""
        if override_dict is None:
            return
        for key, value in override_dict.items():
            if hasattr(self, key.lower()):
                setattr(self, key.lower(), value)
            else:
                raise KeyError(f"Invalid parameter key: {key}")
    
    
def run(checkpoints_path, start_stage=1, end_stage=7, params_override=None):
    """
    Run the full MSM analysis pipeline:
    1. Load NTR RMF data.
    2. Load FG RMF data.
    3. Categorize trajectories to microstates.
    4. Embed microstates to histograms.
    5. Cluster histograms to mesostates.
    6. Generate mesostate transition matrix.
    7. Calculate stats (Permeability).
    
    Parameters:
    - checkpoints_path (str): Path to save checkpoints and results.
    - start_stage (int): Stage to start from (1-7).
    - end_stage (int): Stage to end at (1-7) (not inclusive).
    - params_override (dict): Dictionary to override default parameters.
    """
    config = iMSMConfig(checkpoints_path=checkpoints_path)
    
    # Override parameters if provided
    config.override_from_dict(params_override)
    
    params = config.to_legacy_dict()
    validate_params(params)

    # Run stages
    stages = {
        1 : stage_01_loadNTRs,
        2 : stage_02_loadFGs,
        3 : stage_03_categorize,
        4 : stage_04_embed,
        5 : stage_05_cluster,
        6 : stage_06_buildMSM,
        7 : stage_07_computePermeabilities
    }
    
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    for i in range(start_stage, end_stage):
        print(f"Running stage {i}...")
        stages[i](params)
        print(f"Stage {i} completed.")

def validate_params(params):
    if params['CLUSTERING_TYPE'] != "sym-faiss":
        raise ValueError("Only 'sym-faiss' clustering type is supported in this pipeline.")
    if params['STATE_CHOICE_METHOD'] not in ["prominent", "distance"]:
        raise ValueError("STATE_CHOICE_METHOD must be either 'prominent' or 'distance'.")
    if params['STATE_CHOICE_METHOD'] == "distance" and params['DISTANCE_STATE_THRESHOLD_NM'] <= 0:
        raise ValueError("DISTANCE_STATE_THRESHOLD_NM must be positive if STATE_CHOICE_METHOD is 'distance'.")
    if params['DATA_SUBSET_INDEX'] is not None and params['DATA_SUBSET_INDEX'] >= (1 / params['DATA_SUBSET']):
        raise ValueError("DATA_SUBSET_INDEX must be less than 1 / DATA_SUBSET (or None).")
    if params['MODE'] == "bootstrap" and params['BOOTSTRAP_REPEATS'] <= 0:
        raise ValueError("BOOTSTRAP_REPEATS must be greater than 0 when MODE is 'bootstrap'.")
    if params['DATA_SUBSET_MODE'] not in ["time", "simulation"]:
        raise ValueError("DATA_SUBSET_MODE must be either 'time' or 'simulation'.")
    if params['DATA_SUBSET_MODE'] == "simulation" and int(len(params['LOAD_MD_SIMS_RANGE']) * params['DATA_SUBSET']) < 1:
        raise ValueError("DATA_SUBSET is too small for the number of simulations in LOAD_MD_SIMS_RANGE when DATA_SUBSET_MODE is 'simulation'.")

def stage_01_loadNTRs(params):
    from _001_PRC_extract_trajectories import multi_load_kap_data
    
    _ensure_dir(_cp(params, "1_single_sim_kap_coords"))
    kap_results = multi_load_kap_data(
        input_rmf_path=params['LOAD_MD_BASE_PATH'],
        kap_radius=params['LOAD_MD_KAP_RADIUS'],
        kap_amount=params['LOAD_MD_KAP_AMOUNT'],
        start_t=params['LOAD_MD_START_TIME_NS'],
        end_t=params['LOAD_MD_END_TIME_NS'],
        step_t=params['LOAD_MD_STEP_NS'],
        sims_range=params['LOAD_MD_SIMS_RANGE'],
        frames_per_file=1,
        one_frame_from_each=True,
        get_nsites=0
    )
    for i, result in enumerate(kap_results):
        t0_str, t1_str, sim_time_str = _time_range_strings(params)
        sim_dir = _ensure_dir(_cp(params, "1_single_sim_kap_coords", i + 1))
        with open(sim_dir / f"{sim_time_str}.pickle", "wb") as f:
            pickle.dump(result, f)

def stage_02_loadFGs(params):
    from _001_PRC_extract_trajectories import multi_load_fg_data, FG_TYPES, N_CHAINS_PER_FG, N_BEADS_PER_FG
    _ensure_dir(_cp(params, "2_single_sim_fg_coords"))
    
    fg_types = FG_TYPES
    n_chains_per_fg = N_CHAINS_PER_FG
    n_beads_per_fg = N_BEADS_PER_FG
    
    for nup_type in params['LOAD_MD_IGNORED_NUP_TYPES']:
        if nup_type in fg_types:
            idx = fg_types.index(nup_type)
            del fg_types[idx]
            del n_chains_per_fg[idx]
            del n_beads_per_fg[idx]
    
    fg_results = multi_load_fg_data(
        input_rmf_path=params['LOAD_MD_BASE_PATH'],
        fg_types=FG_TYPES,
        n_chains_per_fg=N_CHAINS_PER_FG,
        n_beads_per_fg=N_BEADS_PER_FG,
        start_t=params['LOAD_MD_START_TIME_NS'],
        end_t=params['LOAD_MD_END_TIME_NS'],
        step_t=params['LOAD_MD_STEP_NS'],
        sims_range=params['LOAD_MD_SIMS_RANGE'],
        frames_per_file=1,
        one_frame_from_each=True,
    )
    for i, result in enumerate(fg_results):
        t0_str, t1_str, sim_time_str = _time_range_strings(params)
        sim_dir = _ensure_dir(_cp(params, "2_single_sim_fg_coords", i + 1))
        with open(sim_dir / f"{sim_time_str}-fgs.pickle", "wb") as f:
            pickle.dump(result, f)

def stage_03_categorize(params):
    from _004_PRC_interaction_chains import categorize_multiples
    
    _, _, sim_time_str = _time_range_strings(params)
    fgs_path = (
        params['CUSTOM_FG_COORDS_PATH']
        if params['CUSTOM_FG_COORDS_PATH']
        else str(_cp(params, "2_single_sim_fg_coords")) + "/"
    )
    categorize_multiples(
        sim_indexes=params['LOAD_MD_SIMS_RANGE'],
        sim_times=[sim_time_str],
        diffuser_coords_path_prefix=str(_cp(params, "1_single_sim_kap_coords")),
        fg_coords_path_prefix=fgs_path,
        step=1,
        save_file_path=str(_cp(params, "3_categorized.pickle")),
        k=params['MICROSTATE_K'],
        diffuser_radius_nm=params['LOAD_MD_KAP_RADIUS']/10,
        max_surface_distance_nm=params['MAX_SURFACE_DIST_NM'],
        split_nc=params['SPLIT_NC']
    )

def stage_04_embed(params):
    from _013_PRC_clustering import load_embed_save_2
    
    load_embed_save_2(
    window_size=params['WINDOW_SIZE_STEPS'],
    load_categorized_path=str(_cp(params, "3_categorized.pickle")),
    save_embedded_path=str(_cp(params, "4_embedded.pickle")),
    save_embedded_eighth_path=None,
    split_nc="nmc"
    )

def stage_05_cluster(params):
    if params['MODE'] == "normal":
        _stage_5_cluster_normal(params)
    elif params['MODE'] == "subset":
        _stage_5_cluster_subset(params)
    else:
        raise ValueError("Invalid MODE for stage 5 (cluster). Must be 'normal' or 'subset'.")

def _stage_5_cluster_subset(params):
    from _013_PRC_clustering import load_reduce_cluster_save_2
    subset = _subset_folder(params)
    _ensure_dir(_cp(params, "5_clustering_subsets", subset))
    _ensure_dir(_cp(params, "5_clustered_subsets", subset))
    load_reduce_cluster_save_2(
        pca_components=220,
        n_clusters=params['N_CLUSTERS'],
        load_embedded_path=str(_cp(params, "4_embedded.pickle")),
        save_pca_cluster_path=str(_cp(params, "5_clustering_subsets", subset, "#c#clusters.pickle")),
        save_clustered_path=str(_cp(params, "5_clustered_subsets", subset, "#c#clusters.pickle")),
        verbose=False,
        data_subset=params['DATA_SUBSET'],
        data_subset_index=params['DATA_SUBSET_INDEX'],
        data_subset_mode=params['DATA_SUBSET_MODE'],
        n_sims=len(params['LOAD_MD_SIMS_RANGE'])
        )

def _stage_5_cluster_normal(params):
    from _013_PRC_clustering import load_reduce_cluster_save_2
    _ensure_dir(_cp(params, "5_clustering"))
    _ensure_dir(_cp(params, "5_clustered"))
    load_reduce_cluster_save_2(
        pca_components=220,
        n_clusters=params['N_CLUSTERS'],
        load_embedded_path=str(_cp(params, "4_embedded.pickle")),
        save_pca_cluster_path=str(_cp(params, "5_clustering", "#c#clusters.pickle")),
        save_clustered_path=str(_cp(params, "5_clustered", "#c#clusters.pickle")),
        verbose=False,
        )

def stage_06_buildMSM(params):
    if params['MODE'] == "subset":
        _stage_06_buildMSM_subset(params)
    else:  
        for n_clusters in params['N_CLUSTERS']:
            with open(_cp(params, "5_clustering", f"{n_clusters}clusters.pickle"), "rb") as f:
                clustering = pickle.load(f)
                actual_n_clusters = clustering.centroids.shape[0]
            with open(_cp(params, "5_clustered", f"{n_clusters}clusters.pickle"), "rb") as f:
                clustered_data = pickle.load(f)
            if params['MODE'] == "normal":
                _stage_06_buildMSM_normal(params, n_clusters, actual_n_clusters, clustered_data)
            elif params['MODE'] == "bootstrap":
                _stage_06_buildMSM_bootstrap(params, n_clusters, actual_n_clusters, clustered_data)
            else:
                raise ValueError("Invalid MODE for stage 6 (MSM). Must be 'normal', 'bootstrap', or 'subset'.")

def _stage_06_buildMSM_subset(params):
    from _015_PRC_clustered_chains import generate_transition_matrix
    subset = _subset_folder(params)
    out_dir = _ensure_dir(_cp(params, "6_transition_matrices_subsets", subset))
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustering_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
            actual_n_clusters = clustering.centroids.shape[0]
        with open(_cp(params, "5_clustered_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered_data = pickle.load(f)
        transition_matrix = generate_transition_matrix(clustered_data[:, :], actual_n_clusters, prior=params['TM_PRIOR'])
        with open(out_dir / f"{n_clusters}clusters.pickle", "wb") as f:
            pickle.dump(transition_matrix, f)

def _stage_06_buildMSM_normal(params, n_clusters, actual_n_clusters, clustered_data):
    from _015_PRC_clustered_chains import generate_transition_matrix
    out_dir = _ensure_dir(_cp(params, "6_transition_matrices"))
    transition_matrix = generate_transition_matrix(clustered_data[:, :], actual_n_clusters, prior=params['TM_PRIOR'])
    with open(out_dir / f"{n_clusters}clusters.pickle", "wb") as f:
        pickle.dump(transition_matrix, f)

def _stage_06_buildMSM_bootstrap(params, n_clusters, actual_n_clusters, clustered_data):
    from _015_PRC_clustered_chains import generate_transition_matrix
    out_dir = _ensure_dir(_cp(params, "6_transition_matrices_bootstrap"))
    n_sims = len(params['LOAD_MD_SIMS_RANGE'])
    n_trajs_per_sim = clustered_data.shape[0] // (n_sims * 8) # 8 for eightwise symmetry
    for b in range(params['BOOTSTRAP_REPEATS']):
        # resample sims with replacement
        resampled_indices = np.random.choice(n_sims, n_sims, replace=True)
        resampled_clustered = []
        for sym_i in range(8):
            bottom_range = sym_i * n_sims * n_trajs_per_sim
            sample = np.concatenate([clustered_data[bottom_range + i*n_trajs_per_sim:bottom_range + (i+1)*n_trajs_per_sim, :] for i in resampled_indices], axis=0)
            resampled_clustered.append(sample)
        resampled_clustered = np.concatenate(resampled_clustered, axis=0)
        transition_matrix = generate_transition_matrix(resampled_clustered[:, :], actual_n_clusters, prior=params['TM_PRIOR'])
        with open(out_dir / f"{n_clusters}clusters_bootstrap{b+1}.pickle", "wb") as f:
            pickle.dump(transition_matrix, f)
    
    
def mm_permiability(tm, bottom_states=None, top_states=None):
    from utils import amount_to_concentration, markov_rate_flux, markov_rate_manual

    lagtime = 1e-6 # in seconds
    try:
        rate_1 = markov_rate_flux(tm, start_states=bottom_states, target_states=top_states) # units: 1/lagtime (probably 1000ns)
        rate_2 = markov_rate_flux(tm, start_states=top_states, target_states=bottom_states) # units: 1/lagtime (probably 1000ns)
    except ValueError as e:
        print(f"Error calculating rates: {e}")
        return 0
    rate_s = (1/lagtime) * (rate_1 + rate_2) # units: 1/s

    concentration_M = amount_to_concentration(1, box_side_a=800) # single molecule
    concentration_uM = concentration_M * 1e6
    # print(concentration_uM)
    
    # units : n_events / s / uM / NPC 
    perm = rate_s / concentration_uM
    
    return perm

def get_top_bottom_states(clustering, state_choice_method, distance_state_threshold_nm, clustered=None):
    bottom_indices = []
    top_indices = []
    if state_choice_method == "prominent":
        if clustered is None:
            raise ValueError("clustered data must be provided for 'prominent' state choice method.")
        unique, counts = np.unique(clustered, return_counts=True)
        top_2_indices = np.argsort(counts)[-2:][::-1]
        most_prominent = unique[top_2_indices]
        if len(most_prominent) < 2:
            raise ValueError("Not enough unique states to determine top and bottom states.")
        bottom_indices.append(most_prominent[0])
        top_indices.append(most_prominent[1])
    elif state_choice_method == "distance":
        from _016_EXP_clustered_master_figure import estimate_cluters_mu_cov, calc_coordinate_edges_dict
        from utils import get_sorted_anchor_coordinates_np
        anchor_coordinates = get_sorted_anchor_coordinates_np()[1]
        coordinate_edges = calc_coordinate_edges_dict(anchor_coordinates)
        mus, covs = estimate_cluters_mu_cov(2000, clustering.centroids, coordinate_edges, range(clustering.centroids.shape[0]))
        bottom_indices = np.where(mus[:, 1] <= -distance_state_threshold_nm)[0]
        top_indices = np.where(mus[:, 1] >= distance_state_threshold_nm)[0]
    elif state_choice_method == "nuc_cyt_treshold":
        bottom_indices = np.where(clustering.centroids[:,0] > distance_state_threshold_nm)[0]
        top_indices = np.where(clustering.centroids[:,-1] > distance_state_threshold_nm)[0]
    return bottom_indices, top_indices
            
def stage_07_computePermeabilities(params):
    if params['MODE'] == "normal":
        _stage_07_computePermeabilities_normal(params)
    elif params['MODE'] == "bootstrap":
        _stage_07_computePermeabilities_bootstrap(params)
    elif params['MODE'] == "subset":
        _stage_07_computePermeabilities_subset(params)
    else:    
        raise ValueError("Invalid MODE for stage 7. Must be 'normal' or 'bootstrap'.")



def _stage_07_computePermeabilities_bootstrap(params):
    permeabilities_bootstrap = [{} for _ in range(params['BOOTSTRAP_REPEATS'])]
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustered", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered = pickle.load(f)
        with open(_cp(params, "5_clustering", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
        bottom_indices, top_indices = get_top_bottom_states(clustering=clustering,
                                                    state_choice_method=params['STATE_CHOICE_METHOD'],
                                                    distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'],
                                                    clustered=clustered)    
        permeabilities = {}
        for b in range(params['BOOTSTRAP_REPEATS']):
            with open(_cp(params, "6_transition_matrices_bootstrap", f"{n_clusters}clusters_bootstrap{b+1}.pickle"), "rb") as f:
                tm = pickle.load(f)
            perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
            permeabilities_bootstrap[b][n_clusters] = perm
            if b == 0:
                print(f"Bootstrap {b+1} Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(_cp(params, "7_permeabilities_bootstrap.pickle"), "wb") as f:
        pickle.dump(permeabilities_bootstrap, f)

def _stage_07_computePermeabilities_normal(params):
    permeabilities = {}
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustered", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered = pickle.load(f)
        with open(_cp(params, "5_clustering", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
        with open(_cp(params, "6_transition_matrices", f"{n_clusters}clusters.pickle"), "rb") as f:
            tm = pickle.load(f)
        bottom_indices, top_indices = get_top_bottom_states(clustering=clustering,
                                                                state_choice_method=params['STATE_CHOICE_METHOD'],
                                                                distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'],
                                                                clustered=clustered)
        perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
        permeabilities[n_clusters] = perm
        print(f"Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(_cp(params, "7_permeabilities.pickle"), "wb") as f:
        pickle.dump(permeabilities, f)
        
        
def _stage_07_computePermeabilities_subset(params):
    subset = _subset_folder(params)
    out_dir = _ensure_dir(_cp(params, "7_permeabilities_subsets", subset))
    permeabilities = {}
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustered_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered = pickle.load(f)
        with open(_cp(params, "5_clustering_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
        with open(_cp(params, "6_transition_matrices_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            tm = pickle.load(f)
        bottom_indices, top_indices = get_top_bottom_states(clustering=clustering,
                                                                state_choice_method=params['STATE_CHOICE_METHOD'],
                                                                distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'],
                                                                clustered=clustered)
        perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
        permeabilities[n_clusters] = perm
        print(f"Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(out_dir / "7_permeabilities.pickle", "wb") as f:
        pickle.dump(permeabilities, f)