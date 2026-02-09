from math import e
from scipy.fftpack import shift
from sklearn.base import ClusterMixin
from deeptime.decomposition import TICA
import sklearn.cluster
from sklearn.decomposition import PCA
from utils import z_order_get_only_state_to_idx_dict, z_order_get_only_state_to_idx_dict_nc, z_order_get_only_state_to_idx_dict_nmc
import numpy as np
from scipy.stats import wasserstein_distance_nd
import sklearn
import pickle
import os
import faiss
from symkmeans import SKM

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
        self.centers = None
        self.y = None

    def fit(self, n_clusters, X):
        X = np.sqrt(X) # hellinger distance
        clustering = faiss.Kmeans(d=X.shape[1], k=n_clusters, spherical=True, verbose=False, gpu=True)
        clustering.train(X)        

        self.centers = np.power(clustering.centroids, 2) # to get a distribution that sums to 1
        # self.centers = clustering.centroids
        # self.centers = clustering.centroids
        _, y = clustering.assign(X)
        # sort the centers by z in inverse tranformation
        sorted_indices = sort_cluster_centers(self.centers)
        index_to_new_index = {old_index: new_index for new_index, old_index in enumerate(sorted_indices)}
        y = np.array([index_to_new_index[i] for i in y])
        self.centers = self.centers[sorted_indices]
        
        # Check for near-duplicate centers and remove them
        unique_centers = []
        unique_indices = []
        center_mapping = {}  # maps old index to new index

        for i, center in enumerate(self.centers):
            is_duplicate = False
            for j, unique_center in enumerate(unique_centers):
                # Calculate L2 distance between centers
                distance = np.linalg.norm(center - unique_center)
                if distance < 0.05:
                    is_duplicate = True
                    center_mapping[i] = j
                    break
            
            if not is_duplicate:
                unique_centers.append(center)
                unique_indices.append(i)
                center_mapping[i] = len(unique_centers) - 1

        # Update centers and labels
        self.centers = np.array(unique_centers)
        n_unique_clusters = len(unique_centers)

        # Update y labels to map to new cluster indices
        self.y = np.array([center_mapping[label] for label in y])

        print(f"Reduced from {n_clusters} to {n_unique_clusters} clusters after removing duplicates")
            
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
    
    
class TICAFaiss:
    def __init__(self):
        self.centers = None
        self.inverse_centers = None
        self.y = None

    def fit(self, n_clusters, X, X_timeseries):
        X_timeseries = [X_timeseries[i].T for i in range(X_timeseries.shape[0])]
        tica = TICA(lagtime=1, var_cutoff=0.95)
        tica.fit_from_timeseries(X_timeseries)
        transformed_X = tica.transform(X)
        # print("Transformed X shape:", transformed_X.shape)
        dim = transformed_X.shape[1]
        print(f"TICA dim: {dim}")
        clustering = faiss.Kmeans(d=transformed_X.shape[1], k=n_clusters, verbose=False, gpu=True)
        clustering.train(transformed_X)
        self.centers = clustering.centroids # shape [n_clusters, n_features] in tica space
        _, self.y = clustering.assign(transformed_X) # shape [n_samples,]

        # Calc inverse transform
        model = tica.fetch_model()
        U = model.instantaneous_coefficients
        U_pinv = np.linalg.pinv(U.T)  # Pseudoinverse of U⊤
        # Approximate inverse transform (back to feature space)
        # Note: This assumes identity basis functions (χ₀(x) = x)
        self.inverse_centers = (self.centers @ U_pinv.T[:dim, :]) + model.mean_0
        # normalize the centers to be a distribution
        self.inverse_centers[self.inverse_centers < 0] = 0
        self.inverse_centers = self.inverse_centers / self.inverse_centers.sum(axis=1, keepdims=True)
        
        # sort the centers by z in inverse tranformation
        sorted_indices = sort_cluster_centers(self.inverse_centers)
        index_to_new_index = {old_index: new_index for new_index, old_index in enumerate(sorted_indices)}
        self.y = np.array([index_to_new_index[i] for i in self.y])
        self.centers = self.centers[sorted_indices]
        self.inverse_centers = self.inverse_centers[sorted_indices]
        
        # Check for near-duplicate centers and remove them
        unique_centers = []
        unique_indices = []
        center_mapping = {}  # maps old index to new index

        for i, center in enumerate(self.inverse_centers):
            is_duplicate = False
            for j, unique_center in enumerate(unique_centers):
                # Calculate L2 distance between centers
                distance = np.linalg.norm(center - unique_center)
                if distance < 0.05:
                    is_duplicate = True
                    center_mapping[i] = j
                    break
            
            if not is_duplicate:
                unique_centers.append(center)
                unique_indices.append(i)
                center_mapping[i] = len(unique_centers) - 1

        # Update centers and labels
        self.inverse_centers = np.array(unique_centers)
        n_unique_clusters = len(unique_centers)

        # Update y labels to map to new cluster indices
        self.y = np.array([center_mapping[label] for label in self.y])

        print(f"Reduced from {n_clusters} to {n_unique_clusters} clusters after removing duplicates")
            
        
        return self
    
    def predict(self, X):
        # transformed_X = self.tica.transform(X)
        
        return self.y

    def get_inverse_centers(self):
        return self.inverse_centers
    
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
        
class TICAhdbscan:
    def __init__(self):
        self.centers = None
        self.inverse_centers = None
        self.y = None
        self.dim=None

    def fit(self, X, X_timeseries):
        X_timeseries = [X_timeseries[i].T for i in range(X_timeseries.shape[0])]
        tica = TICA(lagtime=1, var_cutoff=0.40)
        tica.fit_from_timeseries(X_timeseries)
        transformed_X = tica.transform(X)
        # print("Transformed X shape:", transformed_X.shape)
        self.dim = transformed_X.shape[1]
        print(f"TICA dim: {self.dim}")
        n_jobs = len(os.sched_getaffinity(0))
        clustering = sklearn.cluster.HDBSCAN(min_cluster_size=10, min_samples=160, n_jobs=None, leaf_size=80, cluster_selection_method='leaf', store_centers="centroid")
        clustering.fit(transformed_X)
        self.centers = clustering.centroids_ # shape [n_clusters, n_features] in tica space
        _, self.y = clustering.assign(transformed_X) # shape [n_samples,]
        
        # Calc inverse transform
        model = tica.fetch_model()
        U = model.instantaneous_coefficients[:,:self.dim]
        U_pinv = np.linalg.pinv(U.T)  # Pseudoinverse of U⊤
        # Approximate inverse transform (back to feature space)
        # Note: This assumes identity basis functions (χ₀(x) = x)
        self.inverse_centers = (U_pinv @ self.centers.T).T + model.mean_0
        # normalize the centers to be a distribution
        self.inverse_centers[self.inverse_centers < 0] = 0
        self.inverse_centers = self.inverse_centers / self.inverse_centers.sum(axis=1, keepdims=True)
        
        # sort the centers by z in inverse tranformation
        sorted_indices = sort_cluster_centers(self.inverse_centers)
        index_to_new_index = {old_index: new_index for new_index, old_index in enumerate(sorted_indices)}
        self.y = np.array([index_to_new_index[i] for i in self.y])
        self.centers = self.centers[sorted_indices]
        self.inverse_centers = self.inverse_centers[sorted_indices]
        
        return self
    
    def predict(self, X):
        return self.y

    def get_inverse_centers(self):
        return self.inverse_centers
    
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

        
def generate_radially_shifted(categorized_trajectories, shift, split_nc=False):
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
                if state in ["cyt", "nuc"]:
                    shifted_trajectories[diffuser_i, closest_i, time_i] = state
                    continue
                split_state = state.split("_")
                if split_state[1] == "channel":
                    channel_i = int(split_state[2])
                    shifted_trajectories[diffuser_i, closest_i, time_i] = f"{split_state[0]}_channel_{int((channel_i + shift) % 8)}"
                    continue
                
                fg = split_state[0]
                if split_nc:
                    fg_nc = split_state[1]
                    fg_i = int(split_state[2])
                else:
                    fg_i = int(split_state[1])
                fg_ring_i = fg_i // 8
                fg_within_ring_i = fg_i % 8
                if split_nc:
                    shifted_trajectories[diffuser_i, closest_i, time_i] = f"{fg}_{fg_nc}_{(fg_ring_i * 8 + ((fg_within_ring_i + shift) % 8)):02d}"
                else:
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
    
def embed_section(section, state_to_idx_dict, split_nc=None):
    """
    return: shape [total_nups + 4(=220) OR total_nups * 2 + 18(=450),]
    """
    indexed_section = np.zeros_like(section, dtype=int)
    for i, state in enumerate(section):
        indexed_section[i] = state_to_idx_dict[state]
    if split_nc == "nmc":
        return np.bincount(indexed_section, minlength=458) / len(indexed_section)
    elif split_nc == "nc":
        return np.bincount(indexed_section, minlength=450) / len(indexed_section)
    else:
        return np.bincount(indexed_section, minlength=220) / len(indexed_section)

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

def load_embed_save(window_size, load_categorized_path, save_embedded_path, save_embedded_eighth_path, split_nc=None):
    with open(load_categorized_path, "rb") as f:
        categorized_trajectories = pickle.load(f)

    # Generate 8 shifted versions of categorized_trajectories    
    shifted_versions = [categorized_trajectories]
    for i in range(1, 8):
        shifted_versions.append(generate_radially_shifted(categorized_trajectories, i, split_nc=split_nc))
        
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
    if split_nc == "nmc":
        state_to_idx_dict = z_order_get_only_state_to_idx_dict_nmc()
    elif split_nc == "nc":
        state_to_idx_dict = z_order_get_only_state_to_idx_dict_nc()
    else:
        state_to_idx_dict = z_order_get_only_state_to_idx_dict()
    embedded_sections = np.zeros((all_shifted_versions.shape[0], len(state_to_idx_dict), all_shifted_versions.shape[2])) # (n_diffusers * 8, 220, n_sections)
    for i_diffuser in range(all_shifted_versions.shape[0]):
        for i_section in range(all_shifted_versions.shape[2]):
            # embed the section
            section = all_shifted_versions[i_diffuser, :, i_section]
            section = embed_section(section, state_to_idx_dict, split_nc=split_nc)
            embedded_sections[i_diffuser, :, i_section] = section
            
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

def load_reduce_cluster_save(pca_components, n_clusters: list[int], load_embedded_path, save_pca_cluster_path, save_clustered_path = None, verbose: bool = False, method="kmeans", filter_common=False):
    # reshape to [n_diffusers * 8 * n_sections, 218]
    with open(load_embedded_path, "rb") as f:
        embedded_sections = pickle.load(f) # (n_diffusers * 8, 218, n_sections)
        
    
    if filter_common:
        n_trajs = embedded_sections.shape[0]
        traj_len = embedded_sections.shape[2]
        filtered_sections = []
        for i in range(n_trajs):
            # remove the sections in which over 50% of the mass is in cyt or nuc
            traj = embedded_sections[i]
            nuc_mask = traj[0] > 0.9
            cyt_mask = traj[-1] > 0.9
            if (np.sum(nuc_mask) > (0.9 * traj_len) or np.sum(cyt_mask) > (0.9 * traj_len)) and i%10 != 0:
                continue
            filtered_sections.append(traj)
        print(f"{len(filtered_sections)} / {n_trajs} trajectories remain after filtering common sections")
        embedded_sections = np.array(filtered_sections)
        
    reshaped_embedded = np.zeros((embedded_sections.shape[0] * embedded_sections.shape[2], embedded_sections.shape[1])) # (n_diffusers * 8 * n_sections, 218)
    for i in range(embedded_sections.shape[0]):
        for j in range(embedded_sections.shape[2]):
            reshaped_embedded[i * embedded_sections.shape[2] + j] = embedded_sections[i, :, j]

    # cluster the embedded sections
    if method == "hdbscan":
        pca_cluster = TICAhdbscan()
        pca_cluster.fit(reshaped_embedded, embedded_sections)
        y = pca_cluster.predict()
        n_clusters = pca_cluster.centers.shape[0]
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
            pca_cluster = ClusterFaiss()
            pca_cluster.fit(_n_clusters, reshaped_embedded)
        elif method == "ticafaiss":
            pca_cluster = TICAFaiss()
            pca_cluster.fit(_n_clusters, reshaped_embedded, embedded_sections)

        cur_save_pca_cluster_path = save_pca_cluster_path.replace("#c#", f"{_n_clusters}")
        with open(cur_save_pca_cluster_path, "wb") as f:
            pickle.dump(pca_cluster, f)
        if save_clustered_path is not None:
            cur_save_clustered_path = save_clustered_path.replace("#c#", f"{_n_clusters}")
            if method in ["agglomerative", "faiss", "ticafaiss"]:
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
       
       
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
# NEW       
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
def load_embed_save_2(window_size, load_categorized_path, save_embedded_path, split_nc=None, save_embedded_eighth_path=None):
    with open(load_categorized_path, "rb") as f:
        categorized_trajectories = pickle.load(f)
     
    ##########
    # Divide the trajectories into sections
    ##########
    divided = multi_divide_to_sections(categorized_trajectories, window_size)

    ##########
    # Embedded Sections
    ##########
    if split_nc == "nmc":
        state_to_idx_dict = z_order_get_only_state_to_idx_dict_nmc()
    elif split_nc == "nc":
        state_to_idx_dict = z_order_get_only_state_to_idx_dict_nc()
    else:
        state_to_idx_dict = z_order_get_only_state_to_idx_dict()
    embedded_sections = np.zeros((divided.shape[0], len(state_to_idx_dict), divided.shape[2])) # (n_diffusers * 8, 220, n_sections)
    for i_diffuser in range(divided.shape[0]):
        for i_section in range(divided.shape[2]):
            # embed the section
            section = divided[i_diffuser, :, i_section]
            section = embed_section(section, state_to_idx_dict, split_nc=split_nc)
            embedded_sections[i_diffuser, :, i_section] = section
            
    # dump data
    with open(save_embedded_path, "wb") as f:
        pickle.dump(embedded_sections, f)

def calc_shift_indices():
    z_order_dict = z_order_get_only_state_to_idx_dict_nmc()
    keys = list(z_order_dict.keys())
    keys = np.array(keys)[np.newaxis, np.newaxis, :]
    shift_indices = []
    for shift in range(8):
        shifted = generate_radially_shifted(keys, shift, split_nc="nmc")[0, 0, :]
        indices = []
        for key in keys[0, 0, :]:
            idx = np.where(shifted == key)[0][0]
            indices.append(idx)
        shift_indices.append(indices)
    return shift_indices

shift_indices = calc_shift_indices()
def sym_plus_n(X, n):
    n = n % 8
    X = X[:, shift_indices[n]]
    return X

def load_reduce_cluster_save_2(pca_components, n_clusters: list[int], load_embedded_path, save_pca_cluster_path, save_clustered_path = None, verbose: bool = False, data_subset=None, data_subset_index=None, data_subset_mode=None, n_sims=None):
    # reshape to [n_diffusers * n_sections, 218]
    with open(load_embedded_path, "rb") as f:
        embedded_sections = pickle.load(f) # (n_diffusers, 218, n_sections)
        
    if (data_subset is not None) and (data_subset != 1):
        if data_subset_mode == "time":
            print("Using data subset mode: time")
            n_sections = embedded_sections.shape[2]         
            new_n_sections = int(n_sections * data_subset)
            if data_subset_index is None or data_subset_index == -1:
                data_subset_index = (n_sections // new_n_sections) - 1
            embedded_sections = embedded_sections[:, :, data_subset_index * new_n_sections:(data_subset_index + 1) * new_n_sections]
        if data_subset_mode == "simulation":
            print("Using data subset mode: simulations")
            if n_sims is None:
                raise ValueError("n_sims must be provided when data_subset_mode is 'simulations'")
            n_diffusers = embedded_sections.shape[0]         
            diffusers_per_sim = n_diffusers // n_sims
            new_n_sims = int(n_sims * data_subset)
            if data_subset_index is None or data_subset_index == -1:
                data_subset_index = (n_sims // new_n_sims) - 1
            embedded_sections = embedded_sections[data_subset_index * new_n_sims * diffusers_per_sim:(data_subset_index + 1) * new_n_sims * diffusers_per_sim, :, :]
        else:
            raise ValueError("data_subset_mode must be either 'time' or 'simulation'")
        

    reshaped_embedded = np.zeros((embedded_sections.shape[0] * embedded_sections.shape[2], embedded_sections.shape[1])) # (n_diffusers * n_sections, 218)
    for i in range(embedded_sections.shape[0]):
        for j in range(embedded_sections.shape[2]):
            reshaped_embedded[i * embedded_sections.shape[2] + j] = embedded_sections[i, :, j]
    
    for _n_clusters in n_clusters:
        if verbose:
            print(f"n_clusters: {_n_clusters}")
        pca_cluster = SKM(d=reshaped_embedded.shape[1], k=_n_clusters, niter=10, sym_plus_n_func=sym_plus_n, nsym=8)
        pca_cluster.fit(reshaped_embedded)

        cur_save_pca_cluster_path = save_pca_cluster_path.replace("#c#", f"{_n_clusters}")
        with open(cur_save_pca_cluster_path, "wb") as f:
            pickle.dump(pca_cluster, f)
        if save_clustered_path is not None:
            cur_save_clustered_path = save_clustered_path.replace("#c#", f"{_n_clusters}")
            y = pca_cluster.y
            reshaped_labels = np.zeros((embedded_sections.shape[0] * 8, embedded_sections.shape[2]), dtype=int) # (n_diffusers * 8, n_sections)
            for i in range(embedded_sections.shape[0] * 8):
                for j in range(embedded_sections.shape[2]):
                    reshaped_labels[i, j] = y[i * embedded_sections.shape[2] + j]
            with open(cur_save_clustered_path, "wb") as f:
                pickle.dump(reshaped_labels, f)