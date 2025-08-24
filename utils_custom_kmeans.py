import numpy as np

# Custom Manhattan (L1) distance function (pairwise for two 1-D points)
def custom_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

# Vectorized helper to compute distance matrix between all samples X and centroids.
# Falls back to a Python loop if the provided distance function is not recognized / vectorizable.
def _distance_matrix(X, centroids, distance_function):
    # Fast paths for common / known metrics
    if distance_function is custom_distance:
        # Manhattan distance: |x-c| summed over features
        # Broadcast: (n_samples, 1, n_features) - (1, k, n_features) -> (n_samples, k, n_features)
        return np.sum(np.abs(X[:, None, :] - centroids[None, :, :]), axis=2)
    elif distance_function is None:
        # Should not happen, but guard
        raise ValueError("distance_function cannot be None")
    else:
        # Try vectorized call: if the function supports (X, centroids) directly returning (n_samples, k)
        try:
            dm = distance_function(X[:, None, :], centroids[None, :, :])
            if isinstance(dm, np.ndarray) and dm.shape == (X.shape[0], centroids.shape[0]):
                return dm
        except Exception:
            pass
        # Fallback to per-centroid loop (still vectorized over samples)
        dists = []
        for c in centroids:
            dists.append(np.apply_along_axis(distance_function, 1, X, c))
        return np.stack(dists, axis=1)

# Assign clusters based on custom / provided distance function (vectorized)
def assign_clusters(X, centroids, distance_function):
    dm = _distance_matrix(X, centroids, distance_function)
    return np.argmin(dm, axis=1)

# Compute new centroids as mean of assigned points (fully vectorized, handles empty clusters)
def compute_centroids(X, labels, k, old_centroids):
    n_features = X.shape[1]
    sums = np.zeros((k, n_features), dtype=X.dtype)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    # Accumulate sums per cluster
    np.add.at(sums, labels, X)
    # Avoid division by zero: for empty clusters, reinitialize to a random data point
    empty = counts == 0
    # For non-empty clusters compute mean
    centroids = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], old_centroids)
    if np.any(empty):
        # Reinitialize empty centroids to random existing points (k-means++ style improvement could be added)
        rand_indices = np.random.choice(X.shape[0], empty.sum(), replace=False)
        centroids[empty] = X[rand_indices]
    return centroids

# Main function to perform custom K-Means clustering (single initialization)
def k_means_custom(X, k, max_iter=300, distance_function=custom_distance, tol=1e-6, random_state=None):
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        init_indices = rng.choice(len(X), k, replace=False)
    else:
        init_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[init_indices].astype(float, copy=True)

    for _ in range(max_iter):
        labels = assign_clusters(X, centroids, distance_function)
        new_centroids = compute_centroids(X, labels, k, centroids)
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids
        if shift <= tol:
            break
    return labels, centroids

class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=300, distance_function=custom_distance, tol=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.cluster_centers_ = None
        self.distance_function = distance_function
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        self.labels_, self.cluster_centers_ = k_means_custom(
            X, self.n_clusters, self.max_iter, self.distance_function, self.tol, self.random_state
        )
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")
        return assign_clusters(X, self.cluster_centers_, self.distance_function)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def inertia_(self, X=None):
        # Sum of distances of samples to their assigned centroid
        if self.labels_ is None:
            raise ValueError("Model not fitted")
        if X is None:
            raise ValueError("Provide X used for fitting (not stored internally)")
        dm = _distance_matrix(np.asarray(X), self.cluster_centers_, self.distance_function)
        return float(np.sum(dm[np.arange(len(X)), self.labels_]))