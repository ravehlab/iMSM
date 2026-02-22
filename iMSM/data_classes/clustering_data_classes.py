import numpy as np
from dataclasses import dataclass
from pyparsing import Any

@dataclass
class iMSMSingleSimClustering:
    trajectory: np.ndarray[np.int_] # N_trajectories x N_embedding_timepoints
    

@dataclass
class iMSMClustering:
    trajectories: list[iMSMSingleSimClustering] # N_simulations
    n_clusters: int
    unique_components: np.ndarray[np.str_]
    clustering: Any