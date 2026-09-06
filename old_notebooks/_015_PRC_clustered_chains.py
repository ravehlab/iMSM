
# todo standard deviation estimator for transition matrix (see "Markov or Not Markov – This Should Be a Question")
import numpy as np
from collections import defaultdict

import deeptime 

def normalize_rows(counts_matrix):
    transition_matrix = np.zeros_like(counts_matrix)
    n_states = counts_matrix.shape[0]
    rows_0 = 0
    for i in range(n_states):
        row_sum = np.sum(counts_matrix[i, :])
        if row_sum == 0:
            rows_0 += 1
            continue
        transition_matrix[i, :] = counts_matrix[i, :] / row_sum
    if rows_0 > 0:
        print(f"Number of rows with all zero counts: {rows_0}    :(")
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

def generate_transition_matrix(data, n_mesostates, prior=1):
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
    transition_matrix = np.zeros((n_mesostates, n_mesostates))

    # Fill transition matrix with counts
    for i in range(n_mesostates):
        for j in range(n_mesostates):
            transition_matrix[i, j] += counts[i][j]
            
    # Fix sum 0 rows
    # for i in range(n_mesostates):
    #     row_sum = np.sum(transition_matrix[i, :])
    #     if row_sum < 10:
    #         print(f"Row sum is less than 10, setting row {i} to sum of adjacent rows:( ")
    #         prev_shifted = np.concatenate([[0], transition_matrix[i-1, :-1]])  # Shift right, pad left with 0
    #         next_shifted = np.concatenate([transition_matrix[i+1, 1:], [0]])   # Shift left, pad right with 0
    #         transition_matrix[i, :] += (prev_shifted + next_shifted) / 2
    #         continue
            
    # Add prior
    # half_meso = n_mesostates // 2
    # mask = np.tril(np.ones((half_meso, half_meso)), k=0).astype(bool)
    # transition_matrix[:half_meso, :half_meso][mask] += prior
    # mask = np.triu(np.ones((half_meso, half_meso)), k=0).astype(bool)
    # transition_matrix[half_meso:, half_meso:][mask] += prior
    transition_matrix += prior
    # transition_matrix[0,0] += 10000000
    # transition_matrix[-1,-1] += 10000000
    # transition_matrix += np.eye(n_mesostates) * prior
    # transition_matrix += (np.diag(np.full(transition_matrix.shape[0]-1, prior * 0.005), k=1))
    # transition_matrix += (np.diag(np.full(transition_matrix.shape[0]-1, prior * 0.005), k=-1))

    # smooth the matrix with a 3x3 moving average on interior (leaves border rows/cols unchanged)
    # kern = np.ones((3, 3)) / 9.0
    # sub = transition_matrix[1:-1, 1:-1]
    # padded = np.pad(sub, 1, mode='edge')
    # smoothed = np.zeros_like(sub)
    # for i in range(sub.shape[0]):
    #     for j in range(sub.shape[1]):
    #         smoothed[i, j] = np.sum(padded[i:i+3, j:j+3] * kern)
    # transition_matrix[1:-1, 1:-1] = smoothed

    transition_matrix = normalize_rows(transition_matrix)
    
    return transition_matrix

def generate_bmsm(data, n_mesostates, prior=1, lagtime=1, reversible=False):
    # using effective count mode, recommended for bayesian MSMs here:
    # https://deeptime-ml.github.io/latest/api/generated/deeptime.markov.TransitionCountEstimator.html
    counts = deeptime.markov.TransitionCountEstimator(lagtime=lagtime, count_mode="effective", n_states=n_mesostates).fit(data).fetch_model().count_matrix
    # Add pseudo counts to the diagonal and superdiagonals
    pseudo_counts = prior * np.eye(n_mesostates)
    pseudo_counts += (np.diag(np.full(pseudo_counts.shape[0]-1, prior), k=1) * 0.00001)
    pseudo_counts += (np.diag(np.full(pseudo_counts.shape[0]-1, prior), k=-1) * 0.00001)
    counts += pseudo_counts
    mm = deeptime.markov.msm.BayesianMSM(reversible=reversible).fit(counts).fetch_model()
    return mm