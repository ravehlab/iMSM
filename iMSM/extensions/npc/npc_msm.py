
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

def generate_transition_matrix(data, n_mesostates, prior=1, reversible=False):
    """
    Generate a transition matrix from an array of categorical time series.
    
    Parameters:
    -----------
    data : array-like
        Array of shape [n_diffusers, t] containing categorical data, categories denoted by integers.
        categories should probably be sorted somehow (center of mass in fg space?). 
    n_mesostates : int
        
    Returns:
    --------
    transition_matrix : numpy.ndarray
        Matrix of transition probabilities
    """
    if reversible:
        counts = deeptime.markov.TransitionCountEstimator(lagtime=1, count_mode="sliding", n_states=n_mesostates).fit(data).fetch_model().count_matrix
        pseudo_counts = prior * np.eye(n_mesostates)
        counts += pseudo_counts
        # Use ML MSM for reversible matrices properly.
        mm = deeptime.markov.msm.MaximumLikelihoodMSM(reversible=True).fit(counts).fetch_model()
        return mm.transition_matrix

    counts = generate_counts_matrix(data)
    
    # Create transition matrix
    transition_matrix = np.zeros((n_mesostates, n_mesostates))

    # Fill transition matrix with counts
    for i in range(n_mesostates):
        for j in range(n_mesostates):
            transition_matrix[i, j] += counts[i][j]
            
    transition_matrix += prior
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