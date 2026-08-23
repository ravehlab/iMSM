# %% imports
%load_ext autoreload
%autoreload 2

import mdtraj as md
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import importlib   
import os
from typing import Tuple

import iMSM.data_classes.input_data_classes
import iMSM.data_classes.config_data_class
import iMSM._1_categorize
import iMSM._2_embed
import iMSM._3_cluster
import iMSM._4_msm
import iMSM.main
import fg_visualizations

importlib.reload(iMSM.data_classes.input_data_classes)
importlib.reload(iMSM.data_classes.config_data_class)
importlib.reload(iMSM._1_categorize)
importlib.reload(iMSM._2_embed)
importlib.reload(iMSM._3_cluster)
importlib.reload(iMSM._4_msm)
importlib.reload(iMSM.main)
importlib.reload(fg_visualizations)

from iMSM.data_classes.input_data_classes import iMSMInput, iMSMInputSingleSim
from iMSM.data_classes.config_data_class import iMSMConfig
from iMSM._1_categorize import default_categorize
from iMSM._2_embed import default_embed
from iMSM.extensions.npc.npc_utils import infinitesimal_generator
from iMSM.main import run
from fg_visualizations import (
    visualize_cluster_compositions,
    visualize_transition_matrix,
    visualize_kap_states_and_rates,
    generate_vmd_state_scripts,
    reorder_imsm_states_by_y,
)


# %% setup imsm data

IMSM_CHECKPOINT_PATH: str = "data/nup_sims/fsfgx2/imsm/"
TRAJ_PATHS: list[str] = [
    "data/nup_sims/fsfgx2/output_from_0_to_2499_nowat_unwarped.dcd",
    "data/nup_sims/fsfgx2/output_from_2500_to_4999_nowat_unwarped.dcd",
    "data/nup_sims/fsfgx2/output_from_5000_to_7499_nowat_unwarped.dcd",
    "data/nup_sims/fsfgx2/output_from_7500_to_9999_nowat_unwarped.dcd",
    "data/nup_sims/fsfgx2/output_from_10000_to_12499_nowat_unwarped.dcd",
    "data/nup_sims/fsfgx2/output_from_12500_nowat_unwarped.dcd",
]
TOP_PATH: str = "data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb"

# Residues 1-861 are Kap95 (indices 0..860)
# Residues 862-986 are FSFG Chain 1 (indices 861..985)
# Residues 987-1111 are FSFG Chain 2 (indices 986..1110)

# 1. Load and concatenate the trajectories with topology
traj: md.Trajectory = md.load(TRAJ_PATHS, top=TOP_PATH)

# 2. Select ALL alpha carbon atom indices in the entire topology
all_ca_indices: np.ndarray = traj.topology.select("name CA")

# 3. Slice the list of indices by alpha carbon count
# [:861] grabs indices 0 through 860 (all 861 alpha carbons of Kap95)
kap_ca_indices: np.ndarray = all_ca_indices[:861]

# [861:986] grabs indices 861 to 985 (all 125 alpha carbons of first FSFG chain)
fg1_ca_indices: np.ndarray = all_ca_indices[861:986]

# [986:] grabs indices 986 to 1110 (all 125 alpha carbons of second FSFG chain)
fg2_ca_indices: np.ndarray = all_ca_indices[986:]

# Focus on the first FSFG chain for analysis
fg_ca_indices: np.ndarray = fg1_ca_indices

# 4. Extract the coordinates for those specific slices
kap_xyz_mdtraj: np.ndarray = traj.xyz[:, kap_ca_indices, :]
fg_xyz_mdtraj: np.ndarray = traj.xyz[:, fg_ca_indices, :]

# 5. Transpose to shape: (N_alphacarbons, 3, N_timepoints)
kap_coords: np.ndarray = np.transpose(kap_xyz_mdtraj, (1, 2, 0))
fg_coords: np.ndarray = np.transpose(fg_xyz_mdtraj, (1, 2, 0))

# 6. Shorten / slice time (optional)
first_frame: int | None = None  # e.g., 0, or None to start from the beginning
last_frame: int | None = None   # e.g., 2500, or None to include up to the end
stride: int = 1                 # e.g., 1 for all frames, 2 to skip every 2nd frame

frame_slice: slice = slice(first_frame, last_frame, stride)
kap_coords = kap_coords[:, :, frame_slice]
fg_coords = fg_coords[:, :, frame_slice]


def create_imsm_input(
    kap_coords: np.ndarray,
    focal_coords: np.ndarray,
    focal_component_name: str,
) -> iMSMInput:
    """Construct an iMSMInput instance for Kap95 and a given focal component."""
    full_coords: np.ndarray = np.concatenate((kap_coords, focal_coords), axis=0)
    components: np.ndarray = np.array(
        [f"kapC{i}" for i in range(kap_coords.shape[0])] + [focal_component_name]
    )
    trajectories: list[iMSMInputSingleSim] = [iMSMInputSingleSim(trajectory=full_coords)]
    return iMSMInput(
        trajectories=trajectories,
        components=components,
        focal_component=focal_component_name,
    )


def create_imsm_config(
    checkpoints_path: str,
    window_size: int,
    k_closest: int,
    max_surface_dist: float,
    n_clusters: int,
    merge_cluster_threshold: float,
    tm_prior: float,
    start_stage: int,
    end_stage: int,
) -> iMSMConfig:
    """Create an iMSMConfig instance with explicit parameters."""
    config: iMSMConfig = iMSMConfig(checkpoints_path=checkpoints_path)
    config.window_size = window_size
    config.k_closest = k_closest
    config.max_surface_dist = max_surface_dist
    config.n_clusters = n_clusters
    config.merge_cluster_threshold = merge_cluster_threshold
    config.tm_prior = tm_prior
    config.start_stage = start_stage
    config.end_stage = end_stage
    return config


# Construct 3 focal components: C88, C90, and Center of Mass of C88..C91
c88_fg_coords: np.ndarray = fg_coords[88:89, :, :]
c90_fg_coords: np.ndarray = fg_coords[90:91, :, :]
com_88_91_fg_coords: np.ndarray = np.mean(fg_coords[88:92, :, :], axis=0, keepdims=True)

c88_input: iMSMInput = create_imsm_input(
    kap_coords=kap_coords,
    focal_coords=c88_fg_coords,
    focal_component_name="fgC88",
)
c90_input: iMSMInput = create_imsm_input(
    kap_coords=kap_coords,
    focal_coords=c90_fg_coords,
    focal_component_name="fgC90",
)
com_88_91_input: iMSMInput = create_imsm_input(
    kap_coords=kap_coords,
    focal_coords=com_88_91_fg_coords,
    focal_component_name="fgC88_91_com",
)

imsm_runs: list[dict[str, object]] = [
    {
        "name": "fgC88",
        "input": c88_input,
        "checkpoint_path": os.path.join(IMSM_CHECKPOINT_PATH, "fgC88"),
        "focal_fg_alphacarbon": 88,
    },
    {
        "name": "fgC90",
        "input": c90_input,
        "checkpoint_path": os.path.join(IMSM_CHECKPOINT_PATH, "fgC90"),
        "focal_fg_alphacarbon": 90,
    },
    {
        "name": "fgC88_91_com",
        "input": com_88_91_input,
        "checkpoint_path": os.path.join(IMSM_CHECKPOINT_PATH, "fgC88_91_com"),
        "focal_fg_alphacarbon": (88, 91),
    },
]


# %% setup imsm parameters

# Shared configuration parameters
WINDOW_SIZE: int = 500  # number of frames to consider when generating interaction histograms.
K_CLOSEST: int = 5  # number of closest neighbors to consider when generating interaction histograms.
MAX_SURFACE_DIST: float = 1 # 10 angstrom # maximum distance between two components for them to be considered interacting.
N_CLUSTERS: int = 6  # number of clusters to use for the clustering step
MERGE_CLUSTER_THRESHOLD: float = 0.1  # merge clusters closer than threshold
TM_PRIOR: float = 0.01  # added to all entries counts matrix
START_STAGE: int = 1
END_STAGE: int = 5

NS_PER_FRAME: float = 0.96 * stride
n_frames: int = kap_coords.shape[2]
total_sim_time_ns: float = n_frames * NS_PER_FRAME
total_sim_time_us: float = total_sim_time_ns / 1000.0

fg_sequence: str = "".join([traj.topology.atom(idx).residue.code for idx in fg_ca_indices])
fsfg_starts: list[int] = [m.start() for m in re.finditer("FSFG", fg_sequence)]
splits: list[int] = [0] + fsfg_starts + [len(fg_sequence)]

print(f"Total Alpha Carbons found: {len(all_ca_indices)}")
print(f"Alpha Carbons in Kap95:    {len(kap_ca_indices)}")
print(f"Alpha Carbons in FG1:      {len(fg1_ca_indices)}")
print(f"Alpha Carbons in FG2:      {len(fg2_ca_indices)}")
print(f"FG1 sequence ({len(fg_sequence)} aa, split by FSFG motifs):")
for i in range(len(splits) - 1):
    fg_start: int = splits[i]
    fg_end: int = splits[i + 1] - 1
    ca_start: int = fg_ca_indices[fg_start]
    ca_end: int = fg_ca_indices[fg_end]
    seg: str = fg_sequence[fg_start : fg_end + 1]
    print(f"  Motif {i+1} | FG idx [{fg_start:3d}-{fg_end:3d}] | CA idx [{ca_start:3d}-{ca_end:3d}] | aa {fg_start+1:3d}-{fg_end+1:3d}: {seg}")
print(f"Number of frames: {n_frames}")
print(f"Total simulation time: {total_sim_time_ns:.2f} ns ({total_sim_time_us:.2f} µs)")
print(f"Kap coords shape: {kap_coords.shape}")
print(f"FG coords shape:  {fg_coords.shape}")

# %% run imsm
for item in imsm_runs:
    run_name: str = str(item["name"])
    run_input: iMSMInput = item["input"]  # type: ignore[assignment]
    run_chk_path: str = str(item["checkpoint_path"])
    print(f"\n================ Running iMSM for {run_name} ================")
    run_config: iMSMConfig = create_imsm_config(
        checkpoints_path=f"{run_chk_path}/",
        window_size=WINDOW_SIZE,
        k_closest=K_CLOSEST,
        max_surface_dist=MAX_SURFACE_DIST,
        n_clusters=N_CLUSTERS,
        merge_cluster_threshold=MERGE_CLUSTER_THRESHOLD,
        tm_prior=TM_PRIOR,
        start_stage=START_STAGE,
        end_stage=END_STAGE,
    )
    run(input=run_input, config=run_config)
    reorder_imsm_states_by_y(
        imsm_checkpoint_path=run_chk_path,
        top_path=TOP_PATH,
        kap_n_ca=861,
        heat5_range=(175, 195),
        heat6_range=(218, 232),
        tm_prior=TM_PRIOR,
    )

    
# %% visualize cluster compositions and transition matrix
for item in imsm_runs:
    run_name: str = str(item["name"])
    run_chk_path: str = str(item["checkpoint_path"])
    visualize_cluster_compositions(
        imsm_checkpoint_path=run_chk_path,
        kap_n_ca=kap_coords.shape[0],
        focal_component=run_name,
    )
    visualize_transition_matrix(
        imsm_checkpoint_path=run_chk_path,
        focal_component=run_name,
    )


# %% static spatial network visualization
for item in imsm_runs:
    run_name: str = str(item["name"])
    run_chk_path: str = str(item["checkpoint_path"])
    focal_ca: int | tuple[int, int] | list[int] | str = item["focal_fg_alphacarbon"]  # type: ignore[assignment]

    # visualize_kap_states_and_rates(
    #     top_path=TOP_PATH,
    #     imsm_checkpoint_path=run_chk_path,
    #     output_plot_path=f"plots/{run_name}_fsfgx2_kap_states_network.png",
    #     top_rate_percentile=25.0,
    #     kap_n_ca=861,
    #     repeat_colors=("#78909C", "#CFD8DC"),
    #     zoom=False,
    #     free_threshold=0.90,
    #     dt_ns=1000.0,
    #     auto_align_plane=True,
    #     rotation_x_deg=0.0,
    #     rotation_y_deg=0.0,
    #     focal_fg_alphacarbon=focal_ca,
    #     native_pdb_path=TOP_PATH,
    #     k_closest=K_CLOSEST,
    #     max_surface_dist=MAX_SURFACE_DIST,
    #     native_similarity_threshold=0.40,
    #     top_n_print=10,
    # )

    visualize_kap_states_and_rates(
        top_path=TOP_PATH,
        imsm_checkpoint_path=run_chk_path,
        output_plot_path=f"plots/{run_name}_fsfgx2_kap_states_network_zoomed.png",
        top_rate_percentile=10.0,
        kap_n_ca=861,
        repeat_colors=("#78909C", "#CFD8DC"),
        zoom=True,
        free_threshold=0.9,
        dt_ns=1000.0,
        auto_align_plane=True,
        rotation_x_deg=0.0,
        rotation_y_deg=0.0,
        rotation_z_deg=0.0,
        heat5_range=(175, 195),
        heat6_range=(218, 232),
        focal_fg_alphacarbon=focal_ca,
        native_pdb_path=TOP_PATH,
        k_closest=K_CLOSEST,
        max_surface_dist=MAX_SURFACE_DIST,
        native_similarity_threshold=0.75,
        top_n_print=10,
    )


# %% VMD scripts generation for Markov states
for item in imsm_runs:
    run_name: str = str(item["name"])
    run_chk_path: str = str(item["checkpoint_path"])
    focal_ca: int | tuple[int, int] | list[int] | str = item["focal_fg_alphacarbon"]  # type: ignore[assignment]

    generate_vmd_state_scripts(
        top_path=TOP_PATH,
        traj_paths=TRAJ_PATHS,
        imsm_checkpoint_path=run_chk_path,
        output_dir=f"vmd_states/{run_name}",
        kap_n_ca=861,
        focal_fg_alphacarbon=focal_ca,
        window_size=WINDOW_SIZE,
        stride=stride,
        first_frame=first_frame,
        ns_per_frame=NS_PER_FRAME,
        free_threshold=0.90,
        auto_align_plane=True,
        rotation_x_deg=0.0,
        rotation_y_deg=0.0,
        rotation_z_deg=0.0,
        heat5_range=(175, 195),
        heat6_range=(218, 232),
        vmd_zoom_scale=8.0,
    )