import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import seaborn as sns
import concurrent.futures
# --
import importlib
import utils
importlib.reload(utils)
from utils import reorder_transition_matrix, get_max_pb_file, z_order_interactionsmem_transition_matrix, z_order_interactions_transition_matrix, mean_first_passage_time
# --



def sim_mm(tm, length=1000, init_dist=None):
    # init_state = np.random.choice([0,len(tm) - 1], p=[0.5, 0.5]) # Start in nucleus or cytoplasm
    if init_dist is None:
        init_state = np.random.choice(list(range(len(tm))), p=[1/len(tm) for _ in range(len(tm))]) # Start in any state
    else:
        init_state = np.random.choice(list(range(len(tm))), p=init_dist)
    cur_state = init_state
    sim = [cur_state]
    for i in range(length):
        cur_state = np.random.choice(range(len(tm)), p=tm[cur_state])
        sim.append(cur_state)
    return sim

def _sim_mm_parallel(args):
        tm, series_len, init_dist = args
        series = sim_mm(tm, length=series_len - 1, init_dist=init_dist)
        return series

def sim_mm_parallel(tm, n_series, length=1000, init_dist=None):
    """returns a 2D array of shape (n_series, length) which contains mm runs. Utilizes multiple cores for speedup."""
    args_list = [(tm, length, init_dist)] * n_series
    max_workers = len(os.sched_getaffinity(0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        synthetic_data = np.array(list(executor.map(_sim_mm_parallel, args_list)))
    return synthetic_data
    
        
def calc_transports(sim, bottom_states, top_states):
    transports = 0
    cur = None
    for i in range(len(sim)-1):
        if cur == None:
            if sim[i+1] in bottom_states:
                cur = 0
                continue
            elif sim[i+1] in top_states:
                cur = 1
                continue
        elif cur == 0 and sim[i+1] in top_states:
            transports += 1
            cur = 1
        elif cur == 1 and sim[i+1] in bottom_states:
            transports += 1
            cur = 0
    return transports

def mm_permiability_from_path(matrix_path, n_sims=1000, sim_length=1000, bottom_states=None, top_states=None, init_dist=None, mem=False):
    with open(matrix_path, "rb") as f:
        tm, states = pickle.load(f)
    
    if mem: tm, states = z_order_interactionsmem_transition_matrix(tm, states)
    else: tm, states = z_order_interactions_transition_matrix(tm, states)
    
    # Simulate
    synthetic_data = sim_mm_parallel(tm, n_series=n_sims, length=sim_length, init_dist=init_dist)
    
    transports = []
    for i in range(n_sims):
        # sim = sim_mm(tm, length=sim_length, init_dist=init_dist)
        transports.append(calc_transports(synthetic_data[i], bottom_states, top_states))
        
    transport_events = np.sum(transports)
    sim_length_us = sim_length / 100
    sim_length_s = sim_length_us / 1e6
    concentration_uM = n_sims / 5 # approx
    
    # units : n_events / s / uM / NPC 
    perm = transport_events / (sim_length_s * concentration_uM)
    
    return perm

def mm_permiability_from_path_fast(matrix_path, mem=False, bottom_states=None, top_states=None):
    with open(matrix_path, "rb") as f:
        tm, states = pickle.load(f)
    
    if mem: tm, states = z_order_interactionsmem_transition_matrix(tm, states)
    else: tm, states = z_order_interactions_transition_matrix(tm, states)
    
    # Simulate
    mfpt = mean_first_passage_time(tm, states, bottom_states, top_states) # units: 100ns
    mfpt_s = 1e-7 * mfpt # units: s
    transport_events_per_s = 1 / mfpt_s # units: 1/s
    concentration_uM = 50/250 # approx, single molecule
    
    # units : n_events / s / uM / NPC 
    perm = transport_events_per_s / concentration_uM
    
    return perm

def md_permiability_from_path(traj_path, n_sims=1000, sim_length=1000, bottom_states=None, top_states=None):
    with open (traj_path, 'rb') as f:
        data = pickle.load(f)
        
    # convert data to numbers by z axis
    with open(f"data/anchor_coordinates.pickle", "rb") as f:
        anchor_coordinates = pickle.load(f)
    anchor_coordinates = dict(sorted(anchor_coordinates.items(), key=lambda x: x[1][2]))
    new_states = ["nuc"] + list(anchor_coordinates.keys()) + ["cyt"]
    state_to_index = {state: i for i, state in enumerate(new_states)}
    vectorized_lookup = np.vectorize(lambda x: state_to_index.get(x, 0))  # 0 is default if key not found
    data = vectorized_lookup(data)
    
    selected_indices = np.random.choice(data.shape[0], size=n_sims, replace=False)
    data = data[selected_indices]
    if data.shape[1] < sim_length:
        raise ValueError("The selected series length is longer than the data series length.")
    data = data[:, -sim_length:]
    
    transports = []
    for i in range(n_sims):
        transports.append(calc_transports(data[i], bottom_states, top_states))
    
    transport_events = np.sum(transports)
    sim_length_us = sim_length / 100
    sim_length_s = sim_length_us / 1e6
    concentration_uM = n_sims / 5 # approx
    
    # units : n_events / s / uM / NPC 
    perm = transport_events / (sim_length_s * concentration_uM)
    
    return perm


def plot_side_by_side_histograms(array1, array2, bins=10, labels=None, titles=None, figsize=(12, 5)):
    """
    Display histograms of two 1D numpy arrays side by side with frequencies shown as fractions.
    Uses the same y-axis range for both histograms for better comparison.
    
    Parameters:
    -----------
    array1 : numpy.ndarray
        First 1D array to plot
    array2 : numpy.ndarray
        Second 1D array to plot
    bins : int or sequence, optional
        Number of bins for the histograms (default: 10)
    labels : tuple of str, optional
        Labels for the arrays in the legend (default: None)
    titles : tuple of str, optional
        Titles for the subplots (default: None)
    figsize : tuple of float, optional
        Figure size in inches (width, height) (default: (12, 5))
    """
    # Input validation
    if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if array1.ndim != 1 or array2.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
        
    # Set default values
    if labels is None:
        labels = ('Array 1', 'Array 2')
    if titles is None:
        titles = ('Histogram of Array 1', 'Histogram of Array 2')
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)  # Added sharey=True
    
    # Compute histograms to get max density value
    hist1, bins1 = np.histogram(array1, bins=bins, density=True)
    hist2, bins2 = np.histogram(array2, bins=bins, density=True)
    
    # Determine global max density for consistent y-axis
    max_density = max(hist1.max(), hist2.max())
    
    # Plot normalized histogram for the first array (density=True for fractions)
    ax1.hist(array1, bins=bins, alpha=0.7, label=labels[0], color='blue', density=True)
    ax1.set_title(titles[0])
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency (fraction)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot normalized histogram for the second array (density=True for fractions)
    ax2.hist(array2, bins=bins, alpha=0.7, label=labels[1], color='green', density=True)
    ax2.set_title(titles[1])
    ax2.set_xlabel('Value')
    # No need for y-label on second plot since they share y-axis
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Set same y-axis limits for both plots
    y_lim = (0, max_density * 1.05)  # Add 5% padding
    ax1.set_ylim(y_lim)
    ax2.set_ylim(y_lim)
    
    plt.tight_layout()
    plt.show()
    



def get_mm_empirical_distribution(tm_path, n_series, series_len, mem=False, init_dist=None):
    
    
    with open(tm_path, "rb") as f:
        tm, states = pickle.load(f)
    
    # reorder matrix
    if mem: tm, states = z_order_interactionsmem_transition_matrix(tm, states)
    else: tm, states = z_order_interactions_transition_matrix(tm, states)
        
    synthetic_data = sim_mm_parallel(tm, length=series_len, init_dist=init_dist)
    if mem: synthetic_data[synthetic_data > 217] = synthetic_data[synthetic_data > 217] - 218 
    synthetic_data = synthetic_data.flatten()
    
    return np.histogram(synthetic_data, bins=218, density=True)[0]  # Returns the histogram of the data

def get_mm_stationary_distribution(tm_path, mem=False):
    with open(tm_path, "rb") as f:
        tm, states = pickle.load(f)
        
    # reorder matrix
    if mem: tm, states = z_order_interactionsmem_transition_matrix(tm, states)
    else: tm, states = z_order_interactions_transition_matrix(tm, states)
    
    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(tm.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    
    if mem: stationary = stationary[:218] + stationary[218:]
    return stationary
    
def get_md_empiricial_distribution(traj_path, n_series, series_len):
        # load real data
    with open (traj_path, 'rb') as f:
        data = pickle.load(f)

    k_closest = data.shape[1]
    n_series *= k_closest
    data = np.vstack([data[:, i, :] for i in range(data.shape[1])])
    
    # convert data to numbers by z axis
    with open(f"data/anchor_coordinates.pickle", "rb") as f:
        anchor_coordinates = pickle.load(f)
    anchor_coordinates = dict(sorted(anchor_coordinates.items(), key=lambda x: x[1][2]))
    new_states = ["nuc"] + list(anchor_coordinates.keys()) + ["cyt"]
    state_to_index = {state: i for i, state in enumerate(new_states)}
    vectorized_lookup = np.vectorize(lambda x: state_to_index.get(x, 0))  # 0 is default if key not found
    data = vectorized_lookup(data)

    selected_indices = np.random.choice(data.shape[0], size=n_series, replace=False)
    data = data[selected_indices]
    if data.shape[1] < series_len:
        raise ValueError("The selected series length is longer than the data series length.")
    data = data[:, -series_len:]
    data = data.flatten()

    
    return np.histogram(data, bins=218, density=True)[0]  # Returns the histogram of the data