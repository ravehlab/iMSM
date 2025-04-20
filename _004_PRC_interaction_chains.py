import numpy as np
import pickle
from collections import defaultdict
import scipy as sp
import concurrent.futures
import os


def categorize_diffusers(diffuser_coordinates, fg_coordinates, k, max_distance=5):
    """"Returns arrays containing the types and chain indices of the k closest FGs for each diffuser.

    Args:
        diffuser_coordinates (np.array): a coordinates array of shape [n_diffusers, 3]
        fg_coordinates (Dictionary(string : np.array)): a coordinates dictionary of shape {fg_type : [n_chains, n_beads, 3]}
        k (int): Number of closest FGs to find for each diffuser
        
    Returns:
        np.array: Array of shape [n_diffusers, k] containing strings in format "<fg_type>_<chain_i>"
        or "cyt"/"nuc" if no FG is within max_distance.
    """
    n_diffusers = diffuser_coordinates.shape[0]
    # Initialize results array
    result = np.full((n_diffusers, k), "cyt", dtype='U20')
    result[diffuser_coordinates[:,2] < 0, :] = "nuc"
    
    # Pre-calculate total number of chains
    total_chains = sum(coords.shape[0] for coords in fg_coordinates.values())
    
    # Pre-allocate arrays for all chains
    all_min_distances = np.zeros((n_diffusers, total_chains))
    all_formatted_strings = np.empty(total_chains, dtype='U20')
    
    # Calculate minimum distances for all chains at once
    current_idx = 0
    for fg_type, coords in fg_coordinates.items():
        n_chains = coords.shape[0]
        
        # Generate formatted strings for this FG type
        chain_strings = np.array([f"{fg_type}_{i:02d}" for i in range(n_chains)])
        all_formatted_strings[current_idx:current_idx + n_chains] = chain_strings
        
        # Reshape arrays for broadcasting
        chains_reshaped = coords.reshape(n_chains, -1, 3)  # [n_chains, n_beads, 3]
        diffusers_expanded = diffuser_coordinates[:, np.newaxis, np.newaxis, :]  # [n_diffusers, 1, 1, 3]
        
        # Calculate distances to all beads for all chains at once
        distances = np.linalg.norm(chains_reshaped[np.newaxis, :, :, :] - diffusers_expanded, axis=3)
        
        # Find minimum distance per chain
        all_min_distances[:, current_idx:current_idx + n_chains] = np.min(distances, axis=2)
        
        current_idx += n_chains
    
    # Process each diffuser using vectorized operations
    for i in range(n_diffusers):
        # Find valid distances within max_distance
        valid_mask = all_min_distances[i] <= max_distance
        
        if np.any(valid_mask):
            # Get valid distances and their indices
            valid_distances = all_min_distances[i][valid_mask]
            valid_strings = all_formatted_strings[valid_mask]
            
            # Get indices of k smallest distances
            k_smallest_indices = np.argsort(valid_distances)[:k]
            result[i, :len(k_smallest_indices)] = valid_strings[k_smallest_indices]
    
    return result


def categorize_diffusers_over_time(diffuser_trajectories, fg_trajectories, k, step=1):
    n_diffusers = diffuser_trajectories.shape[0]
    n_t = diffuser_trajectories.shape[2]
    n_t2 = fg_trajectories["Nup57"].shape[3]
    if n_t != n_t2:
        raise Exception("Times not matching")
    n_columns = len(range(0, n_t, step))
    categorized_trajectories = np.zeros(shape=(n_diffusers, k, n_columns), dtype='U20')
    for t in range(0, n_t, step):
        cur_diffuser_trajectories = diffuser_trajectories[:, :, t]
        cur_fg_trajectories = {fg_type : fg_trajectories[:, :, :, t] for (fg_type, fg_trajectories) in fg_trajectories.items()}
        categorized_trajectories[:,:,int(t / step)] = categorize_diffusers(cur_diffuser_trajectories, cur_fg_trajectories, k)
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


def categorize_multiples(sim_indexes, sim_times, load_path_prefix="data/singles/", step=1, save_file_path=None):
    i_time_iterator = [(sim_i, time_i, time) for sim_i in sim_indexes for time_i, time in enumerate(sim_times)]


    arrays = [["temp" for _ in range(len(sim_times))] for _ in range(len(sim_indexes))]    

    def process_file(i_time):
        sim_i, time_i, time = i_time
        print(f"{time}, {sim_i} ", end="")
        with open(f"{load_path_prefix}/{sim_i}/{time}.pickle", "rb") as f:
            diffuser_trajectories = pickle.load(f)
        with open(f"{load_path_prefix}/{sim_i}/{time}-fgs.pickle", "rb") as f:
            fg_trajectories = pickle.load(f)
        categorized = categorize_diffusers_over_time(diffuser_trajectories, fg_trajectories, 1, step=step)
        arrays[sim_i - 1][time_i] = categorized

    num_processes = len(os.sched_getaffinity(0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        executor.map(process_file, i_time_iterator)

    for i in range(len(arrays)):
        arrays[i] = np.concatenate(arrays[i], axis=2)
    all = np.concatenate(arrays, axis=0)

    if save_file_path is not None:
        with open(save_file_path, "wb") as f:
            pickle.dump(all[:, 0, :], f)
    return all[:, 0, :]

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