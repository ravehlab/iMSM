import numpy as np
from dataclasses import dataclass

@dataclass 
class iMSMSingleSimEmbedding:
    trajectory: np.ndarray[np.float_] # N_diffusers x N_embedding_dimensions x N_embedding_timepoints

@dataclass
class iMSMEmbedding:
    trajectories: list[iMSMSingleSimEmbedding] # N_simulations
    unique_components: np.ndarray[np.str_] # N_unique_components, index corresponds to matching index in embedded vector