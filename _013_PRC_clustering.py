from typing import Union, Type
from sklearn.base import ClusterMixin
from sklearn.decomposition import PCA
from utils import z_order_get_only_state_to_idx_dict
import numpy as np
import sklearn
import pickle

class PcaCluster:
    def __init__(self, pca_components: int, clustering: ClusterMixin):
        """for now only supports kmeans and bisecting kmeans"""
        self.pca = PCA(n_components=pca_components)
        self.clustering = clustering
        
    def fit(self, X, verbose: bool = False):
        X = self.pca.fit_transform(X)
        if verbose:
            # print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
            print(f"Total Explained variance: {np.sum(self.pca.explained_variance_ratio_)}")
            print(f"n components: {self.pca.n_components_}")
        self.clustering.fit(X)
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
        
        
def load_reduce_cluster_save(pca_components, n_clusters: list[int], load_embedded_path, save_pca_cluster_path, verbose: bool = False, bisecting: bool = False):
    # reshape to [n_diffusers * 8 * n_sections, 218]
    with open(load_embedded_path, "rb") as f:
        embedded_sections = pickle.load(f) # (n_diffusers * 8, 218, n_sections)
        
    reshaped_embedded = np.zeros((embedded_sections.shape[0] * embedded_sections.shape[2], embedded_sections.shape[1])) # (n_diffusers * 8 * n_sections, 218)
    for i in range(embedded_sections.shape[0]):
        for j in range(embedded_sections.shape[2]):
            reshaped_embedded[i * embedded_sections.shape[2] + j] = embedded_sections[i, :, j]

    # cluster the embedded sections
    for _n_clusters in n_clusters:
        if verbose:
            print(f"n_clusters: {_n_clusters}")
        if bisecting:
            clustering = sklearn.cluster.BisectingKMeans(n_clusters=_n_clusters)
        else:
            clustering = sklearn.cluster.KMeans(n_clusters=_n_clusters)
        pca_cluster = PcaCluster(pca_components=pca_components, clustering=clustering)
        pca_cluster.fit(reshaped_embedded, verbose=verbose)
        
        cur_save_pca_cluster_path = save_pca_cluster_path.replace("#c#", f"{_n_clusters}")
        with open(cur_save_pca_cluster_path, "wb") as f:
            pickle.dump(pca_cluster, f)
            
            
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
        reshaped_labels = labels.reshape(embedded_sections.shape[0], embedded_sections.shape[2])
        cur_save_clustered_path = save_clustered_path.replace("#c#", f"{_n_clusters}")
        with open(cur_save_clustered_path, "wb") as f:
            pickle.dump(reshaped_labels, f)
       