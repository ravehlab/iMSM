from scipy import cluster
from sklearn.cluster import KMeans
import numpy as np

from iMSM.data_classes.embedding_data_classes import iMSMEmbedding, iMSMSingleSimEmbedding
from iMSM.data_classes.clustering_data_classes import iMSMClustering, iMSMSingleSimClustering
from iMSM.data_classes.config_data_class import iMSMConfig

def default_cluster(embedding: iMSMEmbedding, iMSMConfig: iMSMConfig) -> iMSMClustering:
    n_clusters = iMSMConfig.n_clusters
    trajectories = embedding.trajectories
    
    # append all single sim embeddings into one array, shape: (N_diffusers x N_embedding_dimensions x N_embedding_timepoints)
    embedded_trajectories = np.concatenate([sim_emb.trajectory for sim_emb in trajectories], axis=0)
    # reshape the array to (n_diffusers * N_embedding_timepoints, N_embedding_dimensions)
    reshaped_embedded_trajectories = np.zeros((embedded_trajectories.shape[0] * embedded_trajectories.shape[2], embedded_trajectories.shape[1])) 
    for i in range(embedded_trajectories.shape[0]):
        for j in range(embedded_trajectories.shape[2]):
            reshaped_embedded_trajectories[i * embedded_trajectories.shape[2] + j] = embedded_trajectories[i, :, j]
    
    # use k-means
    kmeans = KMeans(n_clusters=n_clusters)
    clustered = kmeans.fit_predict(reshaped_embedded_trajectories)
    
    # reshape clustered back to (N_diffusers x N_embedding_timepoints)
    reshaped_labels = np.zeros((embedded_trajectories.shape[0], embedded_trajectories.shape[2]), dtype=int)
    for i in range(embedded_trajectories.shape[0]):
        for j in range(embedded_trajectories.shape[2]):
            reshaped_labels[i, j] = clustered[i * embedded_trajectories.shape[2] + j]
        
    # break back into infividual sim trajectories
    clustered_trajectories = [iMSMSingleSimClustering(trajectory=np.zeros((embedded_trajectories.shape[0], embedded_trajectories.shape[2]), dtype=int))]
    counter = 0
    trajs_per_sim = trajectories[0].trajectory.shape[0]
    for i in range(reshaped_labels.shape[0]):
        if counter == trajs_per_sim:
            counter = 0
            clustered_trajectories.append(iMSMSingleSimClustering(trajectory=np.zeros((embedded_trajectories.shape[0], embedded_trajectories.shape[2]), dtype=int)))
        clustered_trajectories[-1].trajectory[counter] = reshaped_labels[i, :]
        counter += 1
        
        
    
    return iMSMClustering(trajectories=clustered_trajectories, n_clusters=n_clusters, unique_components=embedding.unique_components, clustering=kmeans)