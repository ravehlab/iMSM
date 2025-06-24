import numpy as np
import scipy as sp
import pickle
import os
import sys
import re
import uuid
import importlib.util
import pydtmc


# def import_reload(file_path, function_name):
#     # Generate a random module name to avoid conflicts
#     module_name = f"temp_module_{uuid.uuid4().hex[:8]}"
#     # Set up the spec
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     if spec is None:
#         raise FileNotFoundError(f"Could not find file: {file_path}")
#     # Create the module
#     module = importlib.util.module_from_spec(spec)
#     # Add the module to sys.modules
#     sys.modules[module_name] = module
#     # Execute the module
#     spec.loader.exec_module(module)
#     # Get the specific function
#     try:
#         function = getattr(module, function_name)
#         return function
#     except AttributeError:
#         raise AttributeError(f"Function '{function_name}' not found in {file_path}")

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
    
    states = ["nuc"] + list(anchor_coordinates.keys()) + ["cyt"]
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
    return sp.linalg.logm(P) / dt

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