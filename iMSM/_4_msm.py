import numpy as np

from iMSM.data_classes.clustering_data_classes import iMSMClustering
from iMSM.data_classes.msm_data_classes import iMSMMSM
from iMSM.data_classes.config_data_class import iMSMConfig

def default_msm(clustering: iMSMClustering, config: iMSMConfig) -> iMSMMSM:
    tm_prior = config.tm_prior
    
    # append all single sim clustered trajectories into one array, shape: (N_diffusers x N_embedding_timepoints)
    clustered_trajectories = np.array([sim_clust.trajectory for sim_clust in clustering.trajectories])
    
    # construct transition count matrix
    n_clusters = clustering.n_clusters
    transition_counts = np.zeros((n_clusters, n_clusters), dtype=float)
    for sim_traj in clustered_trajectories:
        for i in range(sim_traj.shape[1] - 1):
            transition_counts[sim_traj[:, i], sim_traj[:, i + 1]] += 1
    # add tm_prior to transition counts
    transition_counts += tm_prior
    
    # row normalize transition counts to get transition matrix
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    
    return iMSMMSM(transition_matrix=transition_matrix)