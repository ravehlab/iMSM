from scipy import cluster
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
import numpy as np

from iMSM.data_classes.embedding_data_classes import iMSMEmbedding, iMSMSingleSimEmbedding
from iMSM.data_classes.clustering_data_classes import iMSMClustering, iMSMSingleSimClustering
from iMSM.data_classes.config_data_class import iMSMConfig

def merge_similar_clusters(labels: np.ndarray, data: np.ndarray, threshold: float) -> tuple[np.ndarray, int, np.ndarray]:
    unique_labels: np.ndarray = np.unique(labels)
    num_unique: int = int(unique_labels.size)
    dim: int = int(data.shape[1])
    
    if num_unique <= 1:
        single_centroid: np.ndarray = np.mean(data, axis=0, keepdims=True)
        return labels, num_unique, single_centroid
    
    centroids: np.ndarray = np.zeros((num_unique, dim), dtype=np.float64)
    for idx, label in enumerate(unique_labels):
        mask: np.ndarray = labels == label
        centroids[idx] = np.mean(data[mask], axis=0)
    
    dist_matrix: np.ndarray = squareform(pdist(centroids, metric="euclidean"))
    adjacency_matrix: np.ndarray = dist_matrix < threshold
    
    n_components: int
    comp_labels: np.ndarray
    n_components, comp_labels = connected_components(
        csgraph=adjacency_matrix, directed=False, return_labels=True
    )
    
    max_old_label: int = int(np.max(labels))
    lookup: np.ndarray = np.zeros(max_old_label + 1, dtype=np.int64)
    for idx, label in enumerate(unique_labels):
        lookup[label] = comp_labels[idx]
    
    new_labels: np.ndarray = lookup[labels]
    
    new_centroids: np.ndarray = np.zeros((n_components, dim), dtype=np.float64)
    for k in range(n_components):
        mask: np.ndarray = new_labels == k
        if np.any(mask):
            new_centroids[k] = np.mean(data[mask], axis=0)
            
    merged_count: int = num_unique - n_components
    print(f"Merged {merged_count} clusters (from {num_unique} to {n_components}) using distance threshold {threshold}.")
    return new_labels, int(n_components), new_centroids

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=iMSMConfig.seed)
    clustered = kmeans.fit_predict(reshaped_embedded_trajectories)
    
    if iMSMConfig.merge_cluster_threshold is not None:
        clustered, n_clusters, new_centroids = merge_similar_clusters(
            clustered, reshaped_embedded_trajectories, iMSMConfig.merge_cluster_threshold
        )
        if hasattr(kmeans, "cluster_centers_"):
            kmeans.cluster_centers_ = new_centroids
        if hasattr(kmeans, "n_clusters"):
            kmeans.n_clusters = n_clusters
        if hasattr(kmeans, "labels_"):
            kmeans.labels_ = clustered
    
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