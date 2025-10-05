import numpy as np
import scipy as sp
import pickle
import os
import re
import pydtmc
import deeptime


AVOGADRO = 6.02214076e23  # mol^-1

def get_sorted_anchor_coordinates():
    with open(f"data/anchor_coordinates.pickle", "rb") as f:
        anchor_coordinates = pickle.load(f)
        
    # Sort anchor coordinates by Z
    anchor_coordinates = dict(sorted(anchor_coordinates.items(), key=lambda x: x[1][2]))
    return anchor_coordinates

def get_sorted_anchor_coordinates_np():
    """
    return sorted anchor coordinates as a numpy array. shape - (216, 3)
    """
    anchor_coordinates = get_sorted_anchor_coordinates()
    # Convert to numpy array
    sorted_keys = list(anchor_coordinates.keys())
    sorted_values = np.array([anchor_coordinates[key] for key in sorted_keys])
    return sorted_keys, sorted_values

def reorder_transition_matrix(transition_matrix, old_order, new_order):
    """
    Reorders a transition matrix according to a new specified order of states.
    """ 
    # Input validation
    if set(old_order) != set(new_order):
        raise ValueError("Old and new order must contain the same elements")
    
    if len(old_order) != len(new_order):
        raise ValueError("Old and new order must have the same length")
        
    if transition_matrix.shape != (len(old_order), len(old_order)):
        raise ValueError("Transition matrix dimensions must match the length of state lists")
    
    # Create mapping from old positions to new positions
    old_to_new = {state: new_order.index(state) for state in old_order}
    
    # Create the reordered matrix
    n = len(old_order)
    reordered_matrix = np.zeros((n, n))
    
    # Fill in the reordered matrix
    for i, old_from_state in enumerate(old_order):
        for j, old_to_state in enumerate(old_order):
            new_i = old_to_new[old_from_state]
            new_j = old_to_new[old_to_state]
            reordered_matrix[new_i, new_j] = transition_matrix[i, j]
            
    return reordered_matrix

def z_order_interactions_transition_matrix(tm, states):
    """old version (no memory)"""
    anchor_coordinates = get_sorted_anchor_coordinates()
    
    for key in list(anchor_coordinates.keys()):
        if key not in states:
            states.append(key)
            tm = np.vstack([tm, np.zeros(tm.shape[1])])
            tm = np.hstack([tm, np.zeros((tm.shape[0], 1))])
    
    
    new_states = ["nuc"] + list(anchor_coordinates.keys()) + ["cyt"]
    tm = reorder_transition_matrix(tm, states, new_states)
    return tm, new_states

def z_order_get_only_state_to_idx_dict():
    anchor_coordinates = get_sorted_anchor_coordinates()
    
    states = ["nuc"] + [f"nuc_channel_{i}" for i in range(8)] + fg_types_nc + [f"cyt_channel_{i}" for i in range(8)] +["cyt"]
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    return state_to_idx

def z_order_get_only_state_to_idx_dict_nc():
    anchor_coordinates = get_sorted_anchor_coordinates()
    fg_types = list(anchor_coordinates.keys())
    fg_types_nc = []
    for fg in fg_types:
        type, number = fg.split("_", 1)
        fg_types_nc.append(f"{type}_N_{number}")
        fg_types_nc.append(f"{type}_C_{number}")
    states = ["nuc"] + [f"nuc_channel_{i}" for i in range(8)] + fg_types_nc + [f"cyt_channel_{i}" for i in range(8)] +["cyt"]
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    return state_to_idx

def z_order_get_only_state_to_idx_dict_nmc():
    anchor_coordinates = get_sorted_anchor_coordinates()
    fg_types = list(anchor_coordinates.keys())
    fg_types_nc = []
    for fg in fg_types:
        type, number = fg.split("_", 1)
        fg_types_nc.append(f"{type}_N_{number}")
        fg_types_nc.append(f"{type}_C_{number}")
    # print(fg_types_nc)

    states = ["nuc"] + \
             [f"nuc_channel_{i}" for i in range(8)] + \
             fg_types_nc[:216] + \
             [f"mid_channel_{i}" for i in range(8)] + \
             fg_types_nc[216:] + \
             [f"cyt_channel_{i}" for i in range(8)] + \
             ["cyt"]
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    return state_to_idx

def z_order_interactionsmem_transition_matrix(tm, states):
    """new version (memory). ordering is splitting the matrix into 2 parts, memory and non memory.
    each is ordered by z within itself."""
    
    anchor_coordinates = get_sorted_anchor_coordinates()
    
    for key in list(anchor_coordinates.keys()):
        if key not in states:
            states.append(key)
            tm = np.vstack([tm, np.zeros(tm.shape[1])])
            tm = np.hstack([tm, np.zeros((tm.shape[0], 1))])
        if key + "_mem" not in states:
            states.append(key + "_mem")
            tm = np.vstack([tm, np.zeros(tm.shape[1])])
            tm = np.hstack([tm, np.zeros((tm.shape[0], 1))])
    
    fgs_by_z = list(anchor_coordinates.keys())
    new_states = ["nuc"] + fgs_by_z + ["cyt"] + ["nuc_mem"] + [fg + "_mem" for fg in fgs_by_z] + ["cyt_mem"]
    tm = reorder_transition_matrix(tm, states, new_states)
    return tm, new_states

def infinitesimal_generator(P, dt=1.0):
    """
    Estimate the infinitesimal generator matrix Q from transition matrix P
    Implementation similar to: https://transitionmatrix.readthedocs.io/en/latest/_modules/transitionMatrix/model.html#TransitionMatrix.generator
    Units of Q are rate, i.e. 1/time. 
    expm(Q * dt) will result in P.
    
    Examples:
    If the units of P are transition probablities per 100ns,
    then the units of Q will be the transition rate 1 / (100ns * dt).
    
    
    Parameters:
    P : numpy.ndarray
        Transition matrix
    dt : float, optional
        The time scale parameter. 
    
    Returns:
    numpy.ndarray
        Infinitesimal generator matrix Q. 
    """
    Q = sp.linalg.logm(P) / dt
    return Q.real

def transition_from_generator(Q, dt=1.0):
    return sp.linalg.expm(Q * dt)

def get_max_pb_file(folder_path):
    """
    Returns the path to the .pb file with the highest number in the given folder.
    
    Args:
        folder_path (str): Path to the folder containing numbered .pb files
        
    Returns:
        str: Full path to the highest numbered .pb file, or None if no matching files
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"'{folder_path}' is not a valid directory")
    
    max_num = -1
    max_file = None
    
    # Regular expression to match files like "100.pb"
    pattern = re.compile(r'^(\d+)\.pb$')
    
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                max_file = filename
    
    return os.path.join(folder_path, max_file) if max_file else None

def mean_first_passage_time(P, states, start_states, target_states):
    mc = pydtmc.MarkovChain(P, states=states)
    return mc.mean_first_passage_times_between(start_states, target_states)

def stationary_distribution(P):
    mc = pydtmc.MarkovChain(P)
    return np.array(mc.pi[0])

# def markov_rate_with_flux(P, start_states, target_states):
#     # Based on Transition path theory. 
#     # See http://docs.markovmodel.org/lecture_tpt.html
#     mc = pydtmc.MarkovChain(P)
#     n_states = len(mc.states)
#     start_states_set = set(start_states)
#     anti_start_states = [i for i in range(n_states) if i not in start_states_set]
    
#     committor_forward = mc.committor_probabilities("forward" ,start_states, target_states)
#     committor_backward = mc.committor_probabilities("backward" ,start_states, target_states)
#     stationary = mc.pi[0]
    
#     # Vectorized:
#     # for i in start_states:
#     #     for j in anti_start_states:
#     #         flux += stationary[i] * committor_forward[j] * P[i, j]
#     flux = (
#         stationary[start_states][:, None]       # shape (|A|,1)
#       * P[np.ix_(start_states, anti_start_states)]  # shape (|A|,|nonA|)
#       * committor_forward[anti_start_states]            # broadcast to (|A|,|nonA|)
#     ).sum()

#     denom = (stationary * committor_backward).sum()
        
#     return flux / denom

def markov_rate_manual(P, start_states, target_states, n_samples=10000, n_steps=100000000):
    mc = deeptime.markov.msm.MarkovStateModel(P)
    times = []
    for i in range(n_samples):
        start = np.random.choice(start_states)
        traj = mc.simulate(n_steps = n_steps, start = start, stop = target_states)
        times.append(len(traj))
    mfpt = np.mean(times)
    return 1 / mfpt

def markov_rate_flux(P, start_states, target_states):
    mc = deeptime.markov.msm.MarkovStateModel(P)
    mfpt = mc.mfpt(start_states, target_states)
    rate = 1 / mfpt
    return rate

def concentration_to_amount(molar: float, box_side_a: float):
    volume = np.power(box_side_a, 3)
    return (molar * AVOGADRO * volume) / 1e+27

def amount_to_concentration(amount: float, box_side_a: float):
    # Units: Molar
    volume = np.power(box_side_a, 3)
    return amount / (AVOGADRO * volume * 1e-27)

def radius_a_to_kda(radius_a):
    """
    Like supplementary of raveh et al 2025.
    Based on  82. H. P. Erickson, Size and shape of protein molecules at the nanometer level determined by sedimentation, gel filtration, and electron microscopy. Biol. Proced. Online 11, 32–51 (2009).
    """
    return (radius_a / 6.6)**3

