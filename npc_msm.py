import os
import pickle
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional


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
    # For example, if there are 30 simulations of length 100us, and data_subset is 0.1, will keep the data_subset_index'th 10us of each simulation.
    # if data_subset_index is None, will take the final 10us.
    data_subset: float = 1.0
    data_subset_index: int = -1
    
    # --- Stage 1 & 2: Loading ---
    load_md_kap_sites: int = 2
    load_md_kap_radius: int = 20
    load_md_kap_amount: int = 50
    load_md_base_path: str = ""
    load_md_sims_range: range = range(1, 31)
    load_md_start_time_ns: int = 10000
    load_md_end_time_ns: int = 30000
    load_md_step_ns: int = 100
    
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
    if params_override:
        # Convert keys to lowercase to match dataclass fields
        clean_overrides = {k.lower(): v for k, v in params_override.items()}
        for key, value in clean_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise KeyError(f"Invalid parameter key: {key}")
    
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
    
    os.makedirs(checkpoints_path, exist_ok=True)
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

def stage_01_loadNTRs(params):
    from _001_PRC_extract_trajectories import multi_load_kap_data
    
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/1_single_sim_kap_coords", exist_ok=True)
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
        os.makedirs(f"{params['CHECKPOINTS_PATH']}/1_single_sim_kap_coords/{i+1}", exist_ok=True)
        t0_str = f"{int(params['LOAD_MD_START_TIME_NS']/1000)}"
        t1_str = f"{int(params['LOAD_MD_END_TIME_NS']/1000)}"
        with open(f"{params['CHECKPOINTS_PATH']}/1_single_sim_kap_coords/{i+1}/{t0_str}-{t1_str}.pickle", "wb") as f:
            pickle.dump(result, f)

def stage_02_loadFGs(params):
    from _001_PRC_extract_trajectories import multi_load_fg_data, FG_TYPES, N_CHAINS_PER_FG, N_BEADS_PER_FG
    
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/2_single_sim_fg_coords", exist_ok=True)
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
        os.makedirs(f"{params['CHECKPOINTS_PATH']}/2_single_sim_fg_coords/{i+1}", exist_ok=True)
        t0_str = f"{int(params['LOAD_MD_START_TIME_NS']/1000)}"
        t1_str = f"{int(params['LOAD_MD_END_TIME_NS']/1000)}"
        with open(f"{params['CHECKPOINTS_PATH']}/2_single_sim_fg_coords/{i+1}/{t0_str}-{t1_str}-fgs.pickle", "wb") as f:
            pickle.dump(result, f)

def stage_03_categorize(params):
    from _004_PRC_interaction_chains import categorize_multiples
    
    t0_str = f"{int(params['LOAD_MD_START_TIME_NS']/1000)}"
    t1_str = f"{int(params['LOAD_MD_END_TIME_NS']/1000)}"
    fgs_path = params['CUSTOM_FG_COORDS_PATH'] if params['CUSTOM_FG_COORDS_PATH'] else f"{params['CHECKPOINTS_PATH']}/2_single_sim_fg_coords/"
    categorize_multiples(
        sim_indexes=params['LOAD_MD_SIMS_RANGE'],
        sim_times=[f"{t0_str}-{t1_str}"],
        diffuser_coords_path_prefix=f"{params['CHECKPOINTS_PATH']}/1_single_sim_kap_coords",
        fg_coords_path_prefix=fgs_path,
        step=1,
        save_file_path=f"{params['CHECKPOINTS_PATH']}/3_categorized.pickle",
        k=params['MICROSTATE_K'],
        diffuser_radius_nm=params['LOAD_MD_KAP_RADIUS']/10,
        max_surface_distance_nm=params['MAX_SURFACE_DIST_NM'],
        split_nc=params['SPLIT_NC']
    )

def stage_04_embed(params):
    from _013_PRC_clustering import load_embed_save_2
    
    load_embed_save_2(
    window_size=params['WINDOW_SIZE_STEPS'],
    load_categorized_path=f"{params['CHECKPOINTS_PATH']}/3_categorized.pickle",
    save_embedded_path=f"{params['CHECKPOINTS_PATH']}/4_embedded.pickle",
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
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/5_clustering_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index", exist_ok=True)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/5_clustered_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index", exist_ok=True)
    load_reduce_cluster_save_2(
        pca_components=220,
        n_clusters=params['N_CLUSTERS'],
        load_embedded_path=f"{params['CHECKPOINTS_PATH']}/4_embedded.pickle",
        save_pca_cluster_path=f"{params['CHECKPOINTS_PATH']}/5_clustering_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/#c#clusters.pickle",
        save_clustered_path=f"{params['CHECKPOINTS_PATH']}/5_clustered_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/#c#clusters.pickle",
        verbose=False,
        data_subset=params['DATA_SUBSET'],
        data_subset_index=params['DATA_SUBSET_INDEX']
        )

def _stage_5_cluster_normal(params):
    from _013_PRC_clustering import load_reduce_cluster_save_2
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/5_clustering", exist_ok=True)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/5_clustered", exist_ok=True)
    load_reduce_cluster_save_2(
        pca_components=220,
        n_clusters=params['N_CLUSTERS'],
        load_embedded_path=f"{params['CHECKPOINTS_PATH']}/4_embedded.pickle",
        save_pca_cluster_path=f"{params['CHECKPOINTS_PATH']}/5_clustering/#c#clusters.pickle",
        save_clustered_path=f"{params['CHECKPOINTS_PATH']}/5_clustered/#c#clusters.pickle",
        verbose=False,
        )

def stage_06_buildMSM(params):
    if params['MODE'] == "subset":
        _stage_06_buildMSM_subset(params)
    else:  
        for n_clusters in params['N_CLUSTERS']:
            with open(f"{params['CHECKPOINTS_PATH']}/5_clustering/{n_clusters}clusters.pickle", "rb") as f:
                clustering = pickle.load(f)
                actual_n_clusters = clustering.centroids.shape[0]
            with open(f"{params['CHECKPOINTS_PATH']}/5_clustered/{n_clusters}clusters.pickle", "rb") as f:
                clustered_data = pickle.load(f)
            if params['MODE'] == "normal":
                _stage_06_buildMSM_normal(params, n_clusters, actual_n_clusters, clustered_data)
            elif params['MODE'] == "bootstrap":
                _stage_06_buildMSM_bootstrap(params, n_clusters, actual_n_clusters, clustered_data)
            else:
                raise ValueError("Invalid MODE for stage 6 (MSM). Must be 'normal', 'bootstrap', or 'subset'.")

def _stage_06_buildMSM_subset(params):
    from _015_PRC_clustered_chains import generate_transition_matrix
    for n_clusters in params['N_CLUSTERS']:
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustering_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/{n_clusters}clusters.pickle", "rb") as f:
            clustering = pickle.load(f)
            actual_n_clusters = clustering.centroids.shape[0]
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustered_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/{n_clusters}clusters.pickle", "rb") as f:
            clustered_data = pickle.load(f)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_subsets", exist_ok=True)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_subsets/{params['DATA_SUBSET']:.2f}fraction/", exist_ok=True)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index", exist_ok=True)
    transition_matrix = generate_transition_matrix(clustered_data[:, :], actual_n_clusters, prior = params['TM_PRIOR'])
    with open(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/{n_clusters}clusters.pickle", "wb") as f:
        pickle.dump(transition_matrix, f)

def _stage_06_buildMSM_normal(params, n_clusters, actual_n_clusters, clustered_data):
    from _015_PRC_clustered_chains import generate_transition_matrix
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices", exist_ok=True)
    transition_matrix = generate_transition_matrix(clustered_data[:, :], actual_n_clusters, prior = params['TM_PRIOR'])
    with open(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices/{n_clusters}clusters.pickle", "wb") as f:
        pickle.dump(transition_matrix, f)

def _stage_06_buildMSM_bootstrap(params, n_clusters, actual_n_clusters, clustered_data):
    from _015_PRC_clustered_chains import generate_transition_matrix
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_bootstrap", exist_ok=True)
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
        transition_matrix = generate_transition_matrix(resampled_clustered[:, :], actual_n_clusters, prior = params['TM_PRIOR'])
        with open(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_bootstrap/{n_clusters}clusters_bootstrap{b+1}.pickle", "wb") as f:
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

def get_top_bottom_states(clustered, clustering, state_choice_method, distance_state_threshold_nm):
    bottom_indices = []
    top_indices = []
    if state_choice_method == "prominent":
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
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustered/{n_clusters}clusters.pickle", "rb") as f:
            clustered = pickle.load(f)
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustering/{n_clusters}clusters.pickle", "rb") as f:
            clustering = pickle.load(f)
        bottom_indices, top_indices = get_top_bottom_states(clustered=clustered,
                                                    clustering=clustering,
                                                    state_choice_method=params['STATE_CHOICE_METHOD'],
                                                    distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'])    
        permeabilities = {}
        for b in range(params['BOOTSTRAP_REPEATS']):
            with open(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_bootstrap/{n_clusters}clusters_bootstrap{b+1}.pickle", "rb") as f:
                tm = pickle.load(f)
            perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
            permeabilities_bootstrap[b][n_clusters] = perm
            if b == 0:
                print(f"Bootstrap {b+1} Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(f"{params['CHECKPOINTS_PATH']}/7_permeabilities_bootstrap.pickle", "wb") as f:
        pickle.dump(permeabilities_bootstrap, f)

def _stage_07_computePermeabilities_normal(params):
    permeabilities = {}
    for n_clusters in params['N_CLUSTERS']:
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustered/{n_clusters}clusters.pickle", "rb") as f:
            clustered = pickle.load(f)
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustering/{n_clusters}clusters.pickle", "rb") as f:
            clustering = pickle.load(f)
        with open(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices/{n_clusters}clusters.pickle", "rb") as f:
            tm = pickle.load(f)
        bottom_indices, top_indices = get_top_bottom_states(clustered=clustered,
                                                                clustering=clustering,
                                                                state_choice_method=params['STATE_CHOICE_METHOD'],
                                                                distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'])
        perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
        permeabilities[n_clusters] = perm
        print(f"Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(f"{params['CHECKPOINTS_PATH']}/7_permeabilities.pickle", "wb") as f:
        pickle.dump(permeabilities, f)
        
        
def _stage_07_computePermeabilities_subset(params):
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/7_permeabilities_subsets", exist_ok=True)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/7_permeabilities_subsets/{params['DATA_SUBSET']:.2f}fraction/", exist_ok=True)
    os.makedirs(f"{params['CHECKPOINTS_PATH']}/7_permeabilities_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index", exist_ok=True)
    permeabilities = {}
    for n_clusters in params['N_CLUSTERS']:
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustered_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/{n_clusters}clusters.pickle", "rb") as f:
            clustered = pickle.load(f)
        with open(f"{params['CHECKPOINTS_PATH']}/5_clustering_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/{n_clusters}clusters.pickle", "rb") as f:
            clustering = pickle.load(f)
        with open(f"{params['CHECKPOINTS_PATH']}/6_transition_matrices_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/{n_clusters}clusters.pickle", "rb") as f:
            tm = pickle.load(f)
        bottom_indices, top_indices = get_top_bottom_states(clustered=clustered,
                                                                clustering=clustering,
                                                                state_choice_method=params['STATE_CHOICE_METHOD'],
                                                                distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'])
        perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
        permeabilities[n_clusters] = perm
        print(f"Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(f"{params['CHECKPOINTS_PATH']}/7_permeabilities_subsets/{params['DATA_SUBSET']:.2f}fraction/{params['DATA_SUBSET_INDEX']}index/7_permeabilities.pickle", "wb") as f:
        pickle.dump(permeabilities, f)