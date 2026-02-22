import numpy as np
from dataclasses import dataclass

@dataclass
class iMSMInputSingleSim:
    """Data class for iMSM input data for a single simulation."""
    trajectory: np.ndarray[np.float_] # N_components x 3 x N_timepoints

@dataclass
class iMSMInput:
    """Data class for iMSM input data."""
    trajectories: list[iMSMInputSingleSim] # N_simulations
    components: np.ndarray[np.str_] # N_components
    focal_component: str