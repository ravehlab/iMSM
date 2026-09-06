import numpy as np
import multiprocessing as mp
from functools import partial
import pickle
from collections import defaultdict
import os

def get_most_prominent_string(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    max_index = np.argmax(counts)
    most_prominent = unique_values[max_index]
    most_prominent_count = counts[max_index]
    fraction = most_prominent_count / len(arr)
    return most_prominent, fraction, len(unique_values)

def memory_categorization(categorized_trajectory, p, window_size):
    """
    Given categorized trajectories via 004_PRC_interaction_chains.ipynb.
    Categorizes them further to account for memory effects. 
    """
    # new_categorized = np.copy(categorized_trajectory)
    # for i in range(window_size, len(categorized_trajectory)):
    #     most_prominent, fraction, _ = get_most_prominent_string(categorized_trajectory[i-window_size:i])
    #     if fraction > p:
    #         new_categorized[i] = most_prominent + "_mem"
    # return new_categorized
    
    new_categorized = np.copy(categorized_trajectory)
    i = window_size
    len_categorized = len(categorized_trajectory)
    while i < len_categorized:
        most_prominent, fraction, _ = get_most_prominent_string(categorized_trajectory[i-window_size:i])
        if fraction > p:
            new_categorized[i-window_size:i] = most_prominent
            i += window_size
        else:
            i += 1
    return new_categorized

def memory_categorization_2(categorized_trajectory, n, window_size):
    new_categorized = np.copy(categorized_trajectory)
    i = window_size
    len_categorized = len(categorized_trajectory)
    while i < len_categorized:
        most_prominent, _, n_uniques = get_most_prominent_string(categorized_trajectory[i-window_size:i])
        if n_uniques <= n:
            new_categorized[i-window_size:i] = most_prominent
            i += window_size
        else:
            i += 1
        
        # if n_uniques <= n:
        #     if (prev_mem is not None) and (prev_mem in categorized_trajectory[i-window_size:i]):
        #         new_categorized[i] = prev_mem
        #     else:
        #         new_categorized[i] = most_prominent
        #         prev_mem = most_prominent
        # else:
        #     if prev_mem is not None:
        #         prev_mem = None
    return new_categorized


# Markov model given new categorization

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
    for i, fg_type in enumerate(fg_types):
        for j in range(n_chains_per_fg[i]):
            states.append(f"{fg_type}_{j:02d}_mem")
    states.append("nuc")
    states.append("cyt")
    states.append("nuc_mem")
    states.append("cyt_mem")
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
    if init_1: transition_matrix = np.ones((n_states, n_states)) * 1e-7
    else: transition_matrix = np.zeros((n_states, n_states))
    
    # Fill transition matrix with probabilities
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            transition_matrix[i, j] += counts[state_i][state_j]
    if symmetrize:
        transition_matrix[:-4, -1] = eightwise_symmetrize_vec(transition_matrix[:-4, -1])
        transition_matrix[:-4, -2] = eightwise_symmetrize_vec(transition_matrix[:-4, -2])
        transition_matrix[:-4, -3] = eightwise_symmetrize_vec(transition_matrix[:-4, -3])
        transition_matrix[:-4, -4] = eightwise_symmetrize_vec(transition_matrix[:-4, -4])
        transition_matrix[-1, :-4] = eightwise_symmetrize_vec(transition_matrix[-1, :-4])
        transition_matrix[-2, :-4] = eightwise_symmetrize_vec(transition_matrix[-2, :-4])
        transition_matrix[-3, :-4] = eightwise_symmetrize_vec(transition_matrix[-3, :-4])
        transition_matrix[-4, :-4] = eightwise_symmetrize_vec(transition_matrix[-4, :-4])
        transition_matrix[0:-4, 0:-4] = eightwise_symmetrize(transition_matrix[0:-4, 0:-4])
        # transition_matrix[216:-4, 216:-4] = eightwise_symmetrize(transition_matrix[216:-4, 216:-4])
        # transition_matrix[0:216, 0:216] = eightwise_symmetrize(transition_matrix[0:216, 0:216])
        
    transition_matrix = normalize_rows(transition_matrix)
    transition_matrix = (1 - add_to_diag) * transition_matrix + add_to_diag * np.eye(n_states)

    return transition_matrix, states


###################################################################################################
###################################################################################################
###################################################################################################

def process_trajectory(traj, p, window_size, memory_categorization_func):
    """Process a single trajectory with the memory categorization function."""
    return memory_categorization_func(traj, p, window_size)

def parallel_memory_categorization(trajectories, p, window_size, memory_categorization_func, num_processes=None):
    """Apply memory_categorization to all trajectories in parallel."""
    if num_processes is None:
        num_processes = len(os.sched_getaffinity(0))  # Use all available CPUs by default
    
    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Create a partial function with fixed p and window_size parameters
        process_func = partial(process_trajectory, p=p, window_size=window_size, 
                              memory_categorization_func=memory_categorization_func)
        
        # Apply the function to all trajectories in parallel
        results = pool.map(process_func, trajectories)
    
    # Convert results back to numpy array
    return np.array(results)