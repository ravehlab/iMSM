
# todo standard deviation estimator for transition matrix (see "Markov or Not Markov – This Should Be a Question")
import numpy as np
from collections import defaultdict

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

def generate_counts_matrix(data):    
    # Initialize transition counts
    counts = defaultdict(lambda: defaultdict(int))
    
    # Count transitions
    for sequence in data:
        for t in range(len(sequence) - 1):
            current_state = sequence[t]
            next_state = sequence[t + 1]
            counts[current_state][next_state] += 1
    return counts

def generate_transition_matrix(data, n_clusters, init_1 = False, add_to_diag = 0):
    """
    Generate a transition matrix from an array of categorical time series.
    
    Parameters:
    -----------
    data : array-like
        Array of shape [n_diffusers, t] containing categorical data, categories denoted by integers.
        categories should probably be sorted somehow (center of mass in fg space?). 
    n_clusters : int
        
    Returns:
    --------
    transition_matrix : numpy.ndarray
        Matrix of transition probabilities
    states : list
        List of unique states (categories)
    """
    counts = generate_counts_matrix(data)
    
    # Create transition matrix
    if init_1: transition_matrix = np.ones((n_clusters, n_clusters)) * 0.001
    else: transition_matrix = np.zeros((n_clusters, n_clusters))
    transition_matrix += add_to_diag * np.eye(n_clusters)
    
    # Fill transition matrix with probabilities
    for i in range(n_clusters):
        for j in range(n_clusters):
            transition_matrix[i, j] += counts[i][j]
    transition_matrix = normalize_rows(transition_matrix)
    
    return transition_matrix