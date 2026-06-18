import numpy as np
import pickle
from collections import defaultdict
import concurrent.futures
import os
from scipy.spatial import cKDTree


def split_fg_trajectories_nc(fg_trajectories):
    """
    Args:
        fg_trajectories (Dictionary(string : np.array)): a coordinates dictionary of shape {fg_type : [n_chains, n_beads, 3, n_t]}
    Returns:
        fg_trajectories_nc Dictionary(string : List(np.array)): a dictionary of shape {fg_type_<N/C> : [n_chains, n_beads/2, 3, n_t]}
    """
    fg_trajectories_nc = {}
    for fg_type, coords in fg_trajectories.items():
        n_beads = coords.shape[1]
        fg_trajectories_nc[f"{fg_type}_N"] = coords[:, :n_beads//2, :, :]
        fg_trajectories_nc[f"{fg_type}_C"] = coords[:, n_beads//2:, :, :]
    return fg_trajectories_nc

def calc_formatted_strings_matrix(fg_coordinates):
    LONGEST_CHAIN_BEADS = 48
    total_chains = sum(coords.shape[0] for coords in fg_coordinates.values())
    all_formatted_strings = np.empty((total_chains, LONGEST_CHAIN_BEADS), dtype='U20')

    current_idx = 0
    for fg_type, coords in fg_coordinates.items():
        n_chains = coords.shape[0]
        # Pre-format all strings at once using vectorized operations
        chain_strings = np.array([f"{fg_type}_{i:02d}" for i in range(n_chains)])
        # Use broadcasting to fill all columns at once instead of looping
        all_formatted_strings[current_idx:current_idx + n_chains, :] = chain_strings[:, np.newaxis]
        
        current_idx += n_chains
    
    return all_formatted_strings
    
    

def categorize_diffusers(diffuser_coordinates, fg_coordinates, k, diffuser_radius_nm=3.5, max_surface_distance_nm=0.7, use_site_coords=False):
    """"Returns arrays containing the types and chain indices of the k closest FGs for each diffuser.

    Args:
        diffuser_coordinates (np.array): a coordinates array of shape [n_diffusers, 3] (unless use_site_coords is True, then shape [n_diffusers, n_sites, 3])
        fg_coordinates (Dictionary(string : np.array)): a coordinates dictionary of shape {fg_type : [n_chains, n_beads, 3]}
        k (int): Number of closest FGs to find for each diffuser
        use_site_coords (bool): If True, diffuser_coordinates is expected to have shape [n_diffusers, n_sites, 3] and the minimum distance from any site to any FG bead will be used.
        
    Returns:
        np.array: Array of shape [n_diffusers, k] containing strings in format "<fg_type>_<chain_i>"
        or "cyt"/"nuc" if no FG is within max_distance.
    """
    #todo implement use_site_coords
    # todo swithc nuc cyt channel
    FG_BEAD_RADIUS_NM=0.8
    KAP_SITE_RADIUS_NM=0.6
    
    max_distance = diffuser_radius_nm + max_surface_distance_nm + FG_BEAD_RADIUS_NM
    n_diffusers = diffuser_coordinates.shape[0]
    
    # Initialize results
    result = np.full((n_diffusers, k), "cyt", dtype='U20')
    result[diffuser_coordinates[:,2] < 0, :] = "nuc"
    
    # Handle channels
    angles_radians = np.arctan2(diffuser_coordinates[:,1], diffuser_coordinates[:,0])
    cyt_channel_mask = (diffuser_coordinates[:,2] < 15) & (diffuser_coordinates[:,2] > 5)
    mid_channel_mask = (diffuser_coordinates[:,2] < 5) & (diffuser_coordinates[:,2] > -5)
    nuc_channel_mask = (diffuser_coordinates[:,2] < 5) & (diffuser_coordinates[:,2] > -15)
    
    for i, angle in enumerate(np.arange(-np.pi, np.pi, np.pi/4)):
        angle_mask = (angles_radians >= angle) & (angles_radians < angle + np.pi/4)
        result[angle_mask & cyt_channel_mask, :] = f"cyt_channel_{int(i)}"
        result[angle_mask & mid_channel_mask, :] = f"mid_channel_{int(i)}"
        result[angle_mask & nuc_channel_mask, :] = f"nuc_channel_{int(i)}"
        
    
    # Build KDTree with all FG beads
    all_beads = []
    all_labels = []
    
    for fg_type, coords in fg_coordinates.items():
        n_chains, n_beads = coords.shape[:2]
        beads = coords.reshape(-1, 3) # [n_chains * n_beads, 3]
        all_beads.append(beads)
        
        # Create labels for each bead
        for chain_i in range(n_chains):
            for bead_j in range(n_beads):
                all_labels.append(f"{fg_type}_{chain_i:02d}")
        
    all_beads = np.vstack(all_beads)
    all_labels = np.array(all_labels)
    
    # Build KDTree
    tree = cKDTree(all_beads)
    
    # Query for k nearest neighbors within max_distance
    distances, indices = tree.query(diffuser_coordinates, k=k, distance_upper_bound=max_distance)
    
    # Handle the case where fewer than k neighbors are found
    for i in range(n_diffusers):
        valid_indices = indices[i][distances[i] < np.inf]
        if len(valid_indices) > 0:
            result[i, :len(valid_indices)] = all_labels[valid_indices]
    
    return result


def categorize_diffusers_over_time(diffuser_trajectories, fg_trajectories, k, step=1, diffuser_radius_nm=3.5, max_surface_distance_nm=0.7):
    """
    returns array of shape [n_diffusers, k(closest), time]
    """
    n_diffusers = diffuser_trajectories.shape[0]
    n_t = diffuser_trajectories.shape[2]
    n_t2 = list(fg_trajectories.values())[0].shape[3]
    if n_t != n_t2:
        raise Exception("Times not matching")
    n_columns = len(range(0, n_t, step))
    all_formatted_strings = None
    categorized_trajectories = np.zeros(shape=(n_diffusers, k, n_columns), dtype='U20')
    for t in range(0, n_t, step):
        cur_diffuser_trajectories = diffuser_trajectories[:, :, t]
        cur_fg_trajectories = {fg_type : fg_trajectories[:, :, :, t] for (fg_type, fg_trajectories) in fg_trajectories.items()}
        # if all_formatted_strings is None:
        #     all_formatted_strings = calc_formatted_strings_matrix(cur_fg_trajectories)
        categorized_trajectories[:,:,int(t / step)] = categorize_diffusers(cur_diffuser_trajectories, cur_fg_trajectories, k, diffuser_radius_nm=diffuser_radius_nm, max_surface_distance_nm=max_surface_distance_nm)
        
    return categorized_trajectories

def normalize_rows(counts_matrix):
    transition_matrix = np.zeros_like(counts_matrix)
    n_states = counts_matrix.shape[0]
    for i in range(n_states):
        row_sum = np.sum(counts_matrix[i, :])
        if row_sum == 0:
            print(f"Row sum is 0, setting row {i} to 0    :( ")
            transition_matrix[i, :] = 0
            continue
        transition_matrix[i, :] = counts_matrix[i, :] / row_sum
    return transition_matrix

def generate_counts_matrix(data, fg_types, n_chains_per_fg):
    # Get unique states
    states = []
    for i, fg_type in enumerate(fg_types):
        for j in range(n_chains_per_fg[i]):
            states.append(f"{fg_type}_{j:02d}")
    states = sorted(states)
    states.append("nuc")
    states.append("cyt")
    n_states = len(states)
    
    # Create state to index mapping
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    
    # Initialize transition counts
    counts = defaultdict(lambda: defaultdict(int))
    
    # Count transitions
    for sequence in data:
        for t in range(len(sequence) - 1):
            current_state = sequence[t]
            next_state = sequence[t + 1]
            counts[current_state][next_state] += 1
    return counts, states

def eightwise_symmetrize(data):
    """Given a matrix, makes it 8-wise symmetric"""
    if (data.shape[0] % 8 != 0) or (data.shape[1] % 8 != 0):
        raise Exception("invalid matrix shape")
    mask_base = np.array(np.eye(8, dtype=bool))
    masks = [np.roll(mask_base, shift=i, axis=0) for i in range(8)]
    new_data = np.zeros_like(data)
    for mask in masks:
        for i in range(int(data.shape[0] / 8)):
            for j in range(int(data.shape[1] / 8)):
                new_data[i*8:(i+1)*8, j*8:(j+1)*8][mask] = np.mean(data[i*8:(i+1)*8, j*8:(j+1)*8][mask])
    return new_data

def eightwise_symmetrize_vec(data):
    if data.ndim != 1:
        raise Exception("not a vector")
    if data.shape[0] % 8 != 0:
        raise Exception("invalid vector shape")
    for i in range(int(data.shape[0] / 8)):
        data[i*8:(i+1)*8] = np.mean(data[i*8:(i+1)*8])
    return data

def generate_transition_matrix(data, fg_types, n_chains_per_fg, init_1 = False, symmetrize = False, add_to_diag = 0):
    """
    Generate a transition matrix from an array of categorical time series.
    
    Parameters:
    -----------
    data : array-like
        Array of shape [n_diffusers, t] containing categorical data (strings)
        
    Returns:
    --------
    transition_matrix : numpy.ndarray
        Matrix of transition probabilities
    states : list
        List of unique states (categories)
    """
    counts, states = generate_counts_matrix(data, fg_types, n_chains_per_fg)
    n_states = len(states)
    
    # Create transition matrix
    if init_1: transition_matrix = np.ones((n_states, n_states)) * 0.1
    else: transition_matrix = np.zeros((n_states, n_states))
    transition_matrix += add_to_diag * np.eye(n_states)
    
    # Fill transition matrix with probabilities
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            transition_matrix[i, j] += counts[state_i][state_j]
    if symmetrize:
        transition_matrix[:-2, -1] = eightwise_symmetrize_vec(transition_matrix[:-2, -1])
        transition_matrix[:-2, -2] = eightwise_symmetrize_vec(transition_matrix[:-2, -2])
        transition_matrix[-1, :-2] = eightwise_symmetrize_vec(transition_matrix[-1, :-2])
        transition_matrix[-2, :-2] = eightwise_symmetrize_vec(transition_matrix[-2, :-2])
        transition_matrix[:-2, :-2] = eightwise_symmetrize(transition_matrix[:-2, :-2])
    transition_matrix = normalize_rows(transition_matrix)
    
    return transition_matrix, states


###########################################################################################
###########################################################################################
###########################################################################################


def categorize_multiples(sim_indexes, sim_times, diffuser_coords_path_prefix, fg_coords_path_prefix, step=1, save_file_path=None, k=1, diffuser_radius_nm=3.5, max_surface_distance_nm=0.7, split_nc=False):
    i_time_iterator = [(sim_i, time_i, time) for sim_i in sim_indexes for time_i, time in enumerate(sim_times)]


    arrays = [["temp" for _ in range(len(sim_times))] for _ in range(len(sim_indexes))]    
    sim_to_idx = {sim_i: idx for idx, sim_i in enumerate(sim_indexes)}

    def process_file_worker(i_time):
        sim_i, time_i, time = i_time
        print(f"{time}, {sim_i} ", end="")
        # print(f"path = {diffuser_coords_path_prefix}/{sim_i}/{time}.pickle")
        # print(f"fg path = {fg_coords_path_prefix}/{sim_i}/{time}-fgs.pickle")
        with open(f"{diffuser_coords_path_prefix}/{sim_i}/{time}.pickle", "rb") as f:
            diffuser_trajectories = pickle.load(f)
        with open(f"{fg_coords_path_prefix}/{sim_i}/{time}-fgs.pickle", "rb") as f:
            fg_trajectories = pickle.load(f)
        if split_nc:
            fg_trajectories = split_fg_trajectories_nc(fg_trajectories)
        categorized = categorize_diffusers_over_time(diffuser_trajectories, fg_trajectories, k=k, step=step, diffuser_radius_nm=diffuser_radius_nm, max_surface_distance_nm=max_surface_distance_nm)
        arrays[sim_to_idx[sim_i]][time_i] = categorized

    num_processes = len(os.sched_getaffinity(0))
    # num_processes = 1
    print(f"Using {num_processes} cores")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        executor.map(process_file_worker, i_time_iterator)
    print("")
    for i in range(len(arrays)):
        arrays[i] = np.concatenate(arrays[i], axis=2)
    all = np.concatenate(arrays, axis=0)

    if save_file_path is not None:
        with open(save_file_path, "wb") as f:
            if k == 1:
                pickle.dump(all[:, 0, :], f)
            else:
                pickle.dump(all, f)
    if k == 1:
        return all[:, 0, :]
    else:
        return all
    

def generate_transition_matrix_from_categorized(load_categorized_path,
                                                save_file_path,
                                                fg_types=['Nup2', 'Nsp1', 'Nup100', 'Nup116', 'Nup159', 'Nup49', 'Nup57', 'Nup145', 'Nup1', 'Nup60'],
                                                n_chains_per_fg=[16, 48, 16, 16, 16, 32, 32, 16, 8, 16],
                                                init_1=True,
                                                add_to_diag=0,
                                                series_len=None):
    with open(load_categorized_path, "rb") as f:
        categorized = pickle.load(f)
    if series_len is not None:
        categorized = categorized[:, :, -series_len:]
    tm, states = generate_transition_matrix(categorized, fg_types, n_chains_per_fg, symmetrize=True, init_1=init_1, add_to_diag=add_to_diag)
    with open(save_file_path, "wb") as f:
        pickle.dump((tm, states), f)