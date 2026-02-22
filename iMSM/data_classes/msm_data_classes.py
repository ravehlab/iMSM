import numpy as np

from dataclasses import dataclass

@dataclass
class iMSMMSM:
    transition_matrix: np.ndarray[np.float_] # N_clusters x N_clusters, row normalized transition counts