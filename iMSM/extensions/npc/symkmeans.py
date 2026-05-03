import faiss
import numpy as np

class SKM():
    def __init__(self, d, k, sym_plus_n_func, nsym, niter=300, n_original_features=None):
        """
        d: int, dimensionality of the input data
        k: int, number of clusters, must be a multiple of nsym
        sym_plus_n_func: function, a function that takes in data and an integer n,
                         and returns the data transformed by the n-th symmetry operation
        nsym: int, number of symmetry operations
        niter: int, number of iterations for the k-means algorithm
        """
        self.d = d
        self.k = k
        self.niter = niter
        self.sym_plus_n_func = sym_plus_n_func
        self.nsym = nsym
        self.centroids = None
        self.centroids_z = None
        self.assignments = None
        self.n_original_features = n_original_features
        if k % nsym != 0:
            raise ValueError("Number of clusters k must be a multiple of nsym.")
        
    def sym_func(self, X):
        syms = [self.sym_plus_n_func(X, i) for i in range(self.nsym)]
        return np.vstack(syms)
    
    def remove_duplicate_centroids(self, centroids, assignments):
        # Check for near-duplicate centers and remove them
        unique_centers = []
        unique_indices = []
        center_mapping = {}  # maps old index to new index

        for i, center in enumerate(centroids):
            is_duplicate = False
            for j, unique_center in enumerate(unique_centers):
                # Calculate L1 distance between centers
                distance = np.sum(np.abs(center - unique_center))
                if distance < 0.005:
                    is_duplicate = True
                    center_mapping[i] = j
                    break

            if not is_duplicate:
                unique_centers.append(center)
                unique_indices.append(i)
                center_mapping[i] = len(unique_centers) - 1

        # Update centers and labels
        new_centroids = np.array(unique_centers)
        # Update assignments to map to new cluster indices
        new_assignments = np.array([center_mapping[label] for label in assignments])

        n_unique_clusters = len(unique_centers)
        print(f"Reduced from {self.k} to {n_unique_clusters} clusters after removing duplicates")
        return new_centroids, new_assignments

    def prune_underpopulated_centroids(self, centroids, assignments, min_cluster_size=10):
        """
        Remove clusters that have fewer than `min_cluster_size` points assigned.
        Reassign their members to the nearest centroid that satisfies the minimum size.
        """
        centroids = np.asarray(centroids)
        assignments = np.asarray(assignments)

        if centroids.ndim != 2:
            raise ValueError("Centroids must be a 2D array.")
        if assignments.ndim != 1:
            raise ValueError("Assignments must be a 1D array.")
        if centroids.shape[0] == 0:
            return centroids, assignments

        cluster_counts = np.bincount(assignments, minlength=centroids.shape[0])
        keep_mask = cluster_counts >= min_cluster_size

        if not np.any(keep_mask):
            raise ValueError(
                "No clusters meet the minimum size requirement. "
                "Consider lowering min_cluster_size or revisiting your clustering parameters."
            )

        valid_indices = np.where(keep_mask)[0]

        for idx in np.where(~keep_mask)[0]:
            member_mask = assignments == idx
            member_count = int(member_mask.sum())
            if member_count == 0:
                continue

            distances = np.linalg.norm(centroids[valid_indices] - centroids[idx], axis=1)
            target = valid_indices[int(np.argmin(distances))]
            assignments[member_mask] = target
            cluster_counts[target] += member_count

        kept_indices = valid_indices
        new_centroids = centroids[kept_indices]

        index_mapping = np.full(centroids.shape[0], -1, dtype=int)
        index_mapping[kept_indices] = np.arange(new_centroids.shape[0])
        new_assignments = index_mapping[assignments]

        if np.any(new_assignments == -1):
            raise RuntimeError("Found assignments pointing to pruned centroids after remapping.")

        print(f"Reduced from {centroids.shape[0]} to {new_centroids.shape[0]} clusters after pruning underpopulated centroids")

        return new_centroids, new_assignments
    
    def merge_heavy_nuc_cys_centroids(self, centroids, assignments, thresh=0.95):
        # merge all centroids where the 0th entry is > thresh to one where the 0th entry is 1
        # also merge all centroids where the 1st entry is > thresh to one where the -1st entry is 1
        centroids = np.asarray(centroids)
        assignments = np.asarray(assignments)

        if centroids.ndim != 2:
            raise ValueError("Centroids must be a 2D array.")
        if assignments.ndim != 1:
            raise ValueError("Assignments must be a 1D array.")
        if centroids.shape[0] == 0:
            return centroids, assignments
        if centroids.shape[1] < 2:
            raise ValueError("Centroids must have at least two features to merge heavy nuc/cys centroids.")
        if not (0.0 <= thresh <= 1.0):
            raise ValueError("thresh must lie in the interval [0, 1].")

        updated_centroids = centroids.copy()
        updated_assignments = assignments.copy()
        dtype = updated_centroids.dtype
        n_clusters = updated_centroids.shape[0]
        drop_indices = set()

        def _merge_group(candidate_indices, target_selector, canonical_vector):
            candidate_indices = np.array([idx for idx in candidate_indices if idx not in drop_indices], dtype=int)
            if candidate_indices.size == 0:
                return None

            target_idx = target_selector(candidate_indices)
            if target_idx is None:
                return None

            if canonical_vector is not None:
                updated_centroids[target_idx] = canonical_vector.astype(dtype, copy=False)

            for idx in candidate_indices:
                if idx == target_idx:
                    continue
                member_mask = updated_assignments == idx
                if member_mask.any():
                    updated_assignments[member_mask] = target_idx
                drop_indices.add(idx)

            return target_idx

        def _select_nuc_target(candidates):
            nuc_like = np.where(np.isclose(updated_centroids[:, 0], 1.0, atol=1e-6))[0]
            for idx in nuc_like:
                if idx in drop_indices:
                    continue
                return int(idx)
            if candidates.size:
                best = candidates[int(np.argmax(updated_centroids[candidates, 0]))]
                return int(best)
            return None

        def _select_cys_target(candidates):
            cys_like = np.where(np.isclose(updated_centroids[:, -1], 1.0, atol=1e-6))[0]
            for idx in cys_like:
                if idx in drop_indices:
                    continue
                return int(idx)
            if candidates.size:
                best = candidates[int(np.argmax(updated_centroids[candidates, -1]))]
                return int(best)
            return None

        nuc_indices = np.where(updated_centroids[:, 0] > thresh)[0]
        nuc_vector = np.zeros(updated_centroids.shape[1], dtype=dtype)
        nuc_vector[0] = 1.0
        nuc_target = _merge_group(nuc_indices, _select_nuc_target, nuc_vector)

        cys_indices = np.where(updated_centroids[:, -1] > thresh)[0]
        cys_vector = np.zeros(updated_centroids.shape[1], dtype=dtype)
        cys_vector[-1] = 1.0
        cys_target = _merge_group(cys_indices, _select_cys_target, cys_vector)

        if not drop_indices:
            return updated_centroids, updated_assignments

        keep_mask = np.ones(n_clusters, dtype=bool)
        keep_mask[list(drop_indices)] = False
        kept_indices = np.where(keep_mask)[0]
        new_centroids = updated_centroids[kept_indices]

        index_mapping = np.full(n_clusters, -1, dtype=int)
        index_mapping[kept_indices] = np.arange(kept_indices.size)
        new_assignments = index_mapping[updated_assignments]

        if np.any(new_assignments < 0):
            raise RuntimeError("Found assignments pointing to merged centroids after remapping.")

        merged_nuc = len(nuc_indices) - (1 if nuc_target is not None else 0)
        merged_cys = len(cys_indices) - (1 if cys_target is not None else 0)
        total_merged = merged_nuc + merged_cys
        print(
            f"Reduced from {n_clusters} to {new_centroids.shape[0]} clusters after merging "
            f"heavy nuc/cys centroids (nuc merged: {max(0, merged_nuc)}, cys merged: {max(0, merged_cys)})"
        )

        return new_centroids, new_assignments


    def fit(self, X, prune_thresh=0.95):
        if X.shape[0] < self.k:
            raise ValueError("Number of clusters k cannot be greater than number of samples.")
        if X.shape[1] != self.d:
            raise ValueError(f"Dimensionality of input data must be {self.d}.")
        
        sym_X = self.sym_func(X)
        clustering = faiss.Clustering(self.d, self.k)
        clustering.niter = 1
        clustering.verbose = False
        index = faiss.IndexFlatL2(self.d)
        
        new_centroids = self.sym_func(X[np.random.choice(X.shape[0], self.k, replace=False)])  
        for i in range(self.niter):
            clustering.train(sym_X, index)
            new_centroids = faiss.vector_to_array(clustering.centroids).reshape(self.k, self.d).copy()
            new_centroids = self.sym_func(new_centroids[:self.k // self.nsym]).astype(np.float32)
            # clustering.centroids = faiss.vector_to_array(faiss.swig_ptr(new_centroids))
            faiss.copy_array_to_vector(new_centroids.flatten(), clustering.centroids)
        self.centroids = new_centroids
        
        # Rebuild the index with the final centroids
        index.reset()
        index.add(self.centroids.astype(np.float32))
        
        _, self.assignments = index.search(sym_X, 1)
        self.assignments = self.assignments.flatten()

        if self.n_original_features is not None and self.n_original_features < self.d:
            self.centroids = self.centroids[:, :self.n_original_features]

        self.centroids, self.assignments = self.merge_heavy_nuc_cys_centroids(self.centroids, self.assignments, thresh=prune_thresh)
        self.centroids, self.assignments = self.prune_underpopulated_centroids(self.centroids, self.assignments, min_cluster_size=25)
        # self.centroids, self.assignments = self.remove_duplicate_centroids(self.centroids, self.assignments)
        self.y = self.assignments
        
        if self.n_original_features is not None and self.n_original_features < self.d:
            unique_labels = np.arange(self.centroids.shape[0])
            self.centroids_z = np.zeros((self.centroids.shape[0], self.d - self.n_original_features))
            for i, label in enumerate(unique_labels):
                mask = (self.y == label)
                if np.any(mask):
                    self.centroids_z[i] = sym_X[mask, self.n_original_features:].mean(axis=0)
                    
        return self
    
    def get_inverse_centers(self):
        return self.centroids
