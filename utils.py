import numpy as np

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