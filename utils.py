import numpy as np
import scipy as sp
import os
import re

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