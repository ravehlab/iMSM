from typing import Union, Type
from pandas.core import algorithms
from sklearn.base import ClusterMixin
import sklearn.cluster
from sklearn.decomposition import PCA
from utils import z_order_get_only_state_to_idx_dict
from utils_custom_kmeans import CustomKMeans
from banditpam import KMedoids
import numpy as np
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
import sklearn
import pickle
import os
import faiss

class PcaCluster:
    def __init__(self, pca_components: int, clustering: ClusterMixin):
        """for now only supports kmeans and bisecting kmeans"""
        self.pca = PCA(n_components=pca_components)
        self.clustering = clustering
        
    def fit(self, X, verbose: bool = False):
        X = self.pca.fit_transform(X)
        # X = np.sqrt(X) # hellinger distance instead of euclidean
        if verbose:
            # print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
            print(f"Total Explained variance: {np.sum(self.pca.explained_variance_ratio_)}")
            print(f"n components: {self.pca.n_components_}")
        self.clustering.fit(X, "L1")
        if verbose:
            print(f"mean intertia: {np.sqrt(self.clustering.inertia_ / len(X))}")
        # sort the centers by z in inverse tranformation
        centers = self.get_inverse_centers()
        sorted_indices = sort_cluster_centers(centers)
        self.clustering.cluster_centers_ = self.pca.transform(centers[sorted_indices])
        self.clustering.labels_ = list(range(len(sorted_indices)))
        return self
        
    def predict(self, X):
        X = self.pca.transform(X)
        # X = np.sqrt(X)  # hellinger distance instead of euclidean
        return self.clustering.predict(X)
    
    def get_inverse_centers(self):
        centers = self.clustering.cluster_centers_
        return self.pca.inverse_transform(centers)
    
    def get_inverse_centers_normalized(self):
        centers = self.get_inverse_centers()
        centers[centers < 0] = 0  # ensure no negative values
        return centers / centers.sum(axis=1, keepdims=True)
    
    def get_inverse_stationary(self, pi):
        """given stationary in the cluster space, return the stationary in the original space"""
        pi = np.array(pi)
        if len(pi.shape) == 1:
            pi = pi[:, np.newaxis]
        centers = self.get_inverse_centers_normalized()
        return (pi * centers).sum(axis=0)  # shape [n_states]

class ClusterAgglomerative:
    def __init__(self, clustering: ClusterMixin):
        """for now only supports kmeans and bisecting kmeans"""
        self.clustering = clustering
        self.centers = None
        self.y = None

    def fit(self, X):
        self.X = X.copy()
        # wasserstein_matrix = calculate_wasserstein_distance_matrix(X)
        y = self.clustering.fit_predict(X)

        self.centers = np.zeros((self.clustering.n_clusters_, X.shape[1]))
        for i in range(self.clustering.n_clusters_):
            center = np.mean(self.X[y == i], axis=0)
            center = center / np.linalg.norm(center)  # normalize the average
            self.centers[i] = center
            
        # sort the centers by z in inverse tranformation
        sorted_indices = sort_cluster_centers(self.centers)
        y = np.array([sorted_indices[i] for i in y])
        self.centers = self.centers[sorted_indices]
        self.y = y
        return self
    
    def predict(self):
        return self.y

    def get_inverse_centers(self):
        return self.centers
    
    def get_inverse_centers_normalized(self):
        centers = self.centers
        centers[centers < 0] = 0  # ensure no negative values
        return centers / centers.sum(axis=1, keepdims=True)
    
    def get_inverse_stationary(self, pi):
        """given stationary in the cluster space, return the stationary in the original space"""
        pi = np.array(pi)
        if len(pi.shape) == 1:
            pi = pi[:, np.newaxis]
        centers = self.get_inverse_centers_normalized()
        return (pi * centers).sum(axis=0)  # shape [n_states]
    
    
class ClusterFaiss:
    def __init__(self):
        """for now only supports kmeans and bisecting kmeans"""
        self.centers = None
        self.y = None

    def fit(self, clustering, X):
        X = np.sqrt(X) # hellinger distance
        clustering.train(X)

        self.centers = np.power(clustering.centroids, 2)
        _, y = clustering.assign(X)
        # sort the centers by z in inverse tranformation
        sorted_indices = sort_cluster_centers(self.centers)
        index_to_new_index = {old_index: new_index for new_index, old_index in enumerate(sorted_indices)}
        y = np.array([index_to_new_index[i] for i in y])
        self.centers = self.centers[sorted_indices]
        self.y = y
        
        return self
    
    def predict(self, X):
        return self.y

    def get_inverse_centers(self):
        return self.centers
    
    def get_inverse_centers_normalized(self):
        centers = self.centers
        centers[centers < 0] = 0  # ensure no negative values
        return centers / centers.sum(axis=1, keepdims=True)
    
    def get_inverse_stationary(self, pi):
        """given stationary in the cluster space, return the stationary in the original space"""
        pi = np.array(pi)
        if len(pi.shape) == 1:
            pi = pi[:, np.newaxis]
        centers = self.get_inverse_centers_normalized()
        return (pi * centers).sum(axis=0)  # shape [n_states]
        
    
        
def generate_radially_shifted(categorized_trajectories, shift):
    """
    categorized_trajectories: shape [n_diffusers, k(closest), time]
    """
    n_diffusers, n_closest, n_time = categorized_trajectories.shape
    shifted_trajectories = np.zeros_like(categorized_trajectories, dtype='U20') # string array
    for diffuser_i in range(n_diffusers):
        for closest_i in range(n_closest):
            for time_i in range(n_time):
                # shift the trajectory radially
                state = categorized_trajectories[diffuser_i, closest_i, time_i]
                if state == "cyt":
                    shifted_trajectories[diffuser_i, closest_i, time_i] = "cyt"
                    continue
                if state == "nuc":
                    shifted_trajectories[diffuser_i, closest_i, time_i] = "nuc"
                    continue
                split_state = state.split("_")
                fg = split_state[0]
                fg_i = int(split_state[1])
                fg_ring_i = fg_i // 8
                fg_within_ring_i = fg_i % 8
                shifted_trajectories[diffuser_i, closest_i, time_i] = f"{fg}_{(fg_ring_i * 8 + ((fg_within_ring_i + shift) % 8)):02d}"
    return shifted_trajectories

def multi_undivide_from_sections(embedded_sections, n_diffusers):
    """
    embedded_sections: shape [n_diffusers * n_sections]
    return: shape [n_diffusers, n_sections]
    """
    
                
def divide_to_sections(categorized_trajectory, window_size):
    """
    categorized_trajectories: shape [k(closest), time]
    return: shape [k * window_size, n_sections(=time//window_size)]
    """
    n_closest, n_time = categorized_trajectory.shape
    n_sections = n_time // window_size
    categorized_trajectory = categorized_trajectory[:, :n_sections * window_size]
    divided_trajectories = np.zeros((n_closest * window_size, n_sections), dtype='U20') # string array
    for i in range(n_sections):
        divided_trajectories[:, i] = categorized_trajectory[:, i * window_size:(i + 1) * window_size].flatten()
    return divided_trajectories
    
def multi_divide_to_sections(categorized_trajectories, window_size):
    """
    categorized_trajectories: shape [n_diffusers, k(closest), time]
    return: shape [n_diffusers, k * window_size, n_sections(=time//window_size)]
    """
    n_diffusers, n_closest, n_time = categorized_trajectories.shape
    divided_trajectories = np.zeros((n_diffusers, n_closest * window_size, n_time // window_size), dtype='U20') # string array
    for i in range(n_diffusers):
        divided_trajectories[i] = divide_to_sections(categorized_trajectories[i], window_size)
    return divided_trajectories
    
def embed_section(section, state_to_idx_dict):
    """
    return: shape [total_nups + 2(=218)]
    """
    indexed_section = np.zeros_like(section, dtype=int)
    for i, state in enumerate(section):
        indexed_section[i] = state_to_idx_dict[state]
    return np.bincount(indexed_section, minlength=218) / len(indexed_section)

def _histogram_mean(histogram):
        total_sum = 0
        total_count = 0
        for value, frequency in enumerate(histogram):
            total_sum += value * frequency
            total_count += frequency
        return total_sum / total_count

def sort_cluster_centers(centers):
    hist_means = np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        hist_means[i] = _histogram_mean(centers[i])
    sorted_indices = np.argsort(hist_means)
    # sorted_centers = centers[sorted_indices]
    return sorted_indices

def load_embed_save(window_size, load_categorized_path, save_embedded_path, save_embedded_eighth_path):
    with open(load_categorized_path, "rb") as f:
        categorized_trajectories = pickle.load(f)

    # Generate 8 shifted versions of categorized_trajectories    
    shifted_versions = [categorized_trajectories]
    for i in range(1, 8):
        shifted_versions.append(generate_radially_shifted(categorized_trajectories, i))
        
    ##########
    # Divide the trajectories into sections
    ##########

    divided_shifted_versions = [None] * 8
    for i in range(8):
        divided_shifted_versions[i] = multi_divide_to_sections(shifted_versions[i], window_size)
    # concatenate all shifted versions
    all_shifted_versions = np.concatenate(divided_shifted_versions, axis=0) # (n_diffusers * 8, k * window_size, n_sections)

    ##########
    # Embed Sections
    ##########

    state_to_idx_dict = z_order_get_only_state_to_idx_dict()
    embedded_sections = np.zeros((all_shifted_versions.shape[0], 218, all_shifted_versions.shape[2])) # (n_diffusers * 8, 218, n_sections)
    for i_diffuser in range(all_shifted_versions.shape[0]):
        for i_section in range(all_shifted_versions.shape[2]):
            # embed the section
            section = all_shifted_versions[i_diffuser, :, i_section]
            embedded_sections[i_diffuser, :, i_section] = embed_section(section, state_to_idx_dict)
            
    # dump data
    with open(save_embedded_path, "wb") as f:
        pickle.dump(embedded_sections, f)
    if save_embedded_eighth_path is not None:
        with open(save_embedded_eighth_path, "wb") as f:
            pickle.dump(embedded_sections[:int(len(embedded_sections)//8)], f)
        
        
def calculate_wasserstein_distance_matrix(embedded_sections):
    n, m = embedded_sections.shape
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = wasserstein_distance_nd(embedded_sections[i], embedded_sections[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

def wasserstein_distance_func(x, y):
    return wasserstein_distance_nd(x, y)

# def remove_near_duplicate_wasserstein(embedded_sections, threshold=1e-4):
#     unique_sections = []
#     for section in embedded_sections:
#         is_duplicate = False
#         for unique_section in unique_sections:
#             # Calculate Wasserstein distance between current section and each unique section
#             distance = wasserstein_distance_func(section, unique_section)
#             if distance < threshold:
#                 is_duplicate = True
#                 break
        
#         # Only add section if it's not a near duplicate of any existing unique section
#         if not is_duplicate:
#             unique_sections.append(section)
            
#     return np.array(unique_sections)

def load_reduce_cluster_save(pca_components, n_clusters: list[int], load_embedded_path, save_pca_cluster_path, save_clustered_path = None, verbose: bool = False, method="kmeans"):
    # reshape to [n_diffusers * 8 * n_sections, 218]
    with open(load_embedded_path, "rb") as f:
        embedded_sections = pickle.load(f) # (n_diffusers * 8, 218, n_sections)
        
    reshaped_embedded = np.zeros((embedded_sections.shape[0] * embedded_sections.shape[2], embedded_sections.shape[1])) # (n_diffusers * 8 * n_sections, 218)
    for i in range(embedded_sections.shape[0]):
        for j in range(embedded_sections.shape[2]):
            reshaped_embedded[i * embedded_sections.shape[2] + j] = embedded_sections[i, :, j]

    # cluster the embedded sections
    if method == "dbscan":
        n_jobs = len(os.sched_getaffinity(0))
        clustering = sklearn.cluster.DBSCAN(eps=1e-5, n_jobs=n_jobs, metric=jensenshannon)
        pca_cluster = ClusterAgglomerative(clustering=clustering)
        pca_cluster.fit(reshaped_embedded)
        y = pca_cluster.predict()
        n_clusters = len(set(y))
        cur_save_pca_cluster_path = save_pca_cluster_path.replace("#c#", f"{n_clusters}")
        with open(cur_save_pca_cluster_path, "wb") as f:
            pickle.dump(pca_cluster, f)
        
        reshaped_labels = np.zeros((embedded_sections.shape[0], embedded_sections.shape[2]), dtype=int) # (n_diffusers * 8, n_sections)
        for i in range(embedded_sections.shape[0]):
            for j in range(embedded_sections.shape[2]):
                reshaped_labels[i, j] = y[i * embedded_sections.shape[2] + j]
        
        if verbose:
            print(f"dbscan found {n_clusters} clusters")
        cur_save_clustered_path = save_clustered_path.replace("#c#", f"{n_clusters}")
        with open(cur_save_clustered_path, "wb") as f:
            pickle.dump(reshaped_labels, f)
        return
    
    for _n_clusters in n_clusters:
        if verbose:
            print(f"n_clusters: {_n_clusters}")
        if method in ["kmeans", "bisecting"]:
            if method == "bisecting":
                clustering = sklearn.cluster.BisectingKMeans(n_clusters=_n_clusters)
            elif method == "kmeans":
                clustering = sklearn.cluster.KMeans(n_clusters=_n_clusters)
            pca_cluster = PcaCluster(pca_components=pca_components, clustering=clustering)
            pca_cluster.fit(reshaped_embedded, verbose=verbose)
        elif method == "agglomerative":
            clustering = sklearn.cluster.SpectralClustering(n_clusters=_n_clusters)
            pca_cluster = ClusterAgglomerative(clustering=clustering)
            pca_cluster.fit(reshaped_embedded)
        elif method == "faiss":
            clustering = faiss.Kmeans(d=reshaped_embedded.shape[1], k=_n_clusters, spherical=True)
            clustering.train(reshaped_embedded)
            pca_cluster = ClusterFaiss()
            pca_cluster.fit(clustering, reshaped_embedded)

        cur_save_pca_cluster_path = save_pca_cluster_path.replace("#c#", f"{_n_clusters}")
        with open(cur_save_pca_cluster_path, "wb") as f:
            pickle.dump(pca_cluster, f)
        if save_clustered_path is not None:
            cur_save_clustered_path = save_clustered_path.replace("#c#", f"{_n_clusters}")
            if method in ["agglomerative", "faiss"]:
                y = pca_cluster.y
            else:
                y = pca_cluster.predict(reshaped_embedded)
                
                
            reshaped_labels = np.zeros((embedded_sections.shape[0], embedded_sections.shape[2]), dtype=int) # (n_diffusers * 8, n_sections)
            for i in range(embedded_sections.shape[0]):
                for j in range(embedded_sections.shape[2]):
                    reshaped_labels[i, j] = y[i * embedded_sections.shape[2] + j]
                
            with open(cur_save_clustered_path, "wb") as f:
                pickle.dump(reshaped_labels, f)
            
            
def load_cluster_trajectories_save(n_clusters: list[int], load_embedded_path, load_pca_cluster_path, save_clustered_path):
    """save clustered trajectories for the original data (multiplied by 8!!)"""
    with open(load_embedded_path, "rb") as f:
        embedded_sections = pickle.load(f)
    reshaped_embedded = np.zeros((embedded_sections.shape[0] * embedded_sections.shape[2], embedded_sections.shape[1])) # (n_diffusers * 8 * n_sections, 218)
    for i in range(embedded_sections.shape[0]):
        for j in range(embedded_sections.shape[2]):
            reshaped_embedded[i * embedded_sections.shape[2] + j] = embedded_sections[i, :, j]

    for _n_clusters in n_clusters:
        cur_load_pca_cluster_path = load_pca_cluster_path.replace("#c#", f"{_n_clusters}")
        with open(cur_load_pca_cluster_path, "rb") as f:
            pca_cluster = pickle.load(f)
        labels = pca_cluster.predict(reshaped_embedded)
        reshaped_labels = np.zeros((embedded_sections.shape[0], embedded_sections.shape[2]), dtype=int) # (n_diffusers * 8, n_sections)
        for i in range(embedded_sections.shape[0]):
            for j in range(embedded_sections.shape[2]):
                reshaped_labels[i, j] = labels[i * embedded_sections.shape[2] + j]
        cur_save_clustered_path = save_clustered_path.replace("#c#", f"{_n_clusters}")
        with open(cur_save_clustered_path, "wb") as f:
            pickle.dump(reshaped_labels, f)
       