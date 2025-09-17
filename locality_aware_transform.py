import numpy as np

class LocalityAwareTransform:
    """
    Observable transform that emphasizes local features and interactions
    while de-emphasizing long-range unlikely combinations.
    Includes weighted reconstruction for inverse transform.
    """
    
    def __init__(self, 
                 n_features=218,
                 window_size=16,       # Local interaction window
                 include_squares=True,  # Include squared terms
                 include_local_pairs=True,  # Include local pairwise products
                 decay_factor=0.9):    # Exponential decay for distant features
        self.n_features = n_features
        self.window_size = window_size
        self.include_squares = include_squares
        self.include_local_pairs = include_local_pairs
        self.decay_factor = decay_factor
        self._feature_map = None  # Will store mapping info after first call
        self._orig_dim = None
    
    def __call__(self, x):
        """Apply the locality-aware transform"""
        features = []
        feature_info = []  # Track what each feature represents
        
        # Store original dimensionality for inverse transform
        if self._feature_map is None:
            self._orig_dim = x.shape[1]
            self._feature_map = {'features': [], 'weights': []}
        
        # 1. Original features with exponential weighting
        weights = np.array([self.decay_factor ** i for i in range(x.shape[1])])
        weighted_original = x * weights[np.newaxis, :]
        features.append(weighted_original)
        
        if self._feature_map is not None:
            for i in range(x.shape[1]):
                feature_info.append({'type': 'weighted_original', 'indices': [i], 'weight': weights[i]})
        
        # 2. Focus intensively on the most local features
        if x.shape[1] >= self.n_features:
            local_features = x[:, :self.n_features] * 2.0
            features.append(local_features)
            
            if self._feature_map is not None:
                for i in range(self.n_features):
                    feature_info.append({'type': 'local_emphasis', 'indices': [i], 'weight': 2.0})
        
        # 3. Squared terms for local features only
        if self.include_squares:
            local_end = min(self.n_features, x.shape[1])
            local_squares = x[:, :local_end] ** 2
            features.append(local_squares)
            
            if self._feature_map is not None:
                for i in range(local_end):
                    feature_info.append({'type': 'squared', 'indices': [i], 'weight': 1.0})
        
        # 4. Local pairwise interactions within sliding windows
        if self.include_local_pairs:
            for start in range(0, min(self.n_features, x.shape[1]), self.window_size):
                end = min(start + self.window_size, x.shape[1], self.n_features)
                if end - start > 1:
                    local_chunk = x[:, start:end]
                    for i in range(local_chunk.shape[1]):
                        for j in range(i+1, local_chunk.shape[1]):
                            pair_product = (local_chunk[:, i] * local_chunk[:, j]).reshape(-1, 1)
                            features.append(pair_product)
                            
                            if self._feature_map is not None:
                                feature_info.append({'type': 'pairwise', 'indices': [start + i, start + j], 'weight': 1.0})
        
        # 5. Cumulative features (prefix sums)
        cumsum_features = np.cumsum(x[:, :self.n_features], axis=1)
        features.append(cumsum_features)
        
        if self._feature_map is not None:
            for i in range(min(self.n_features, x.shape[1])):
                feature_info.append({'type': 'cumsum', 'indices': list(range(i+1)), 'weight': 1.0})
        
        # 6. Local variance/spread measures
        for start in range(0, min(self.n_features, x.shape[1]), self.window_size):
            end = min(start + self.window_size, x.shape[1])
            if end - start > 1:
                local_chunk = x[:, start:end]
                local_var = np.var(local_chunk, axis=1, keepdims=True)
                local_mean = np.mean(local_chunk, axis=1, keepdims=True)
                features.extend([local_var, local_mean])
                
                if self._feature_map is not None:
                    feature_info.append({'type': 'local_var', 'indices': list(range(start, end)), 'weight': 1.0})
                    feature_info.append({'type': 'local_mean', 'indices': list(range(start, end)), 'weight': 1.0})
        
        # Store feature mapping for inverse transform
        if self._feature_map['features'] == []:
            self._feature_map['features'] = feature_info
            self._feature_map['weights'] = weights
        
        return np.hstack(features)
    
    def inverse_transform(self, transformed_vector, tica_model=None):
        """
        Estimate the original vector from a transformed vector using weighted reconstruction.
        
        Parameters:
        -----------
        transformed_vector : array-like, shape (n_transformed_features,)
            Vector in the transformed feature space
        tica_model : CovarianceKoopmanModel, optional
            The fitted TICA model to invert through
            
        Returns:
        --------
        original_estimate : array, shape (original_dim,)
            Estimated original vector
        confidence : array, shape (original_dim,)
            Confidence scores for each dimension (higher = more reliable)
        """
        if self._feature_map is None:
            raise ValueError("Transform must be applied at least once before inverse transform")
        
        # If TICA model provided, first invert through TICA space
        if tica_model is not None:
            # This is an approximation - TICA inverse is not exact
            # We use the eigenvectors to approximate the inverse transformation
            try:
                # Get the TICA components (right eigenvectors)
                components = tica_model.singular_vectors_right[:, :len(transformed_vector)]
                # Approximate inverse: project back through eigenvectors
                feature_space_vector = components @ transformed_vector
            except:
                # Fallback if TICA model doesn't have expected attributes
                feature_space_vector = transformed_vector
        else:
            feature_space_vector = transformed_vector
        
        return self._weighted_reconstruction(feature_space_vector)
    
    def _weighted_reconstruction(self, feature_vector):
        """Reconstruct using weighted combination of primary features"""
        original_estimate = np.zeros(self._orig_dim)
        confidence = np.zeros(self._orig_dim)
        
        # Focus on the most direct features first
        feature_idx = 0
        
        # 1. Weighted original features (most reliable)
        for i in range(self._orig_dim):
            if feature_idx < len(feature_vector):
                weight = self._feature_map['weights'][i]
                original_estimate[i] += feature_vector[feature_idx] / weight
                confidence[i] += 3.0 * weight  # High confidence for direct features
                feature_idx += 1
        
        # 2. Local emphasis features (second most reliable)
        for i in range(min(self.n_features, self._orig_dim)):
            if feature_idx < len(feature_vector):
                original_estimate[i] += feature_vector[feature_idx] / 2.0
                confidence[i] += 2.0
                feature_idx += 1
        
        # 3. Use squared features to refine estimates
        if self.include_squares:
            for i in range(min(self.n_features, self._orig_dim)):
                if feature_idx < len(feature_vector) and feature_vector[feature_idx] >= 0:
                    sqrt_val = np.sqrt(max(0, feature_vector[feature_idx]))
                    # Weight this less heavily since squares can be ambiguous
                    original_estimate[i] = (confidence[i] * original_estimate[i] + 0.5 * sqrt_val) / (confidence[i] + 0.5)
                    confidence[i] += 0.5
                    feature_idx += 1
        
        # 4. Skip pairwise and other complex features for simplicity in reconstruction
        # They contribute to the forward transform but are harder to invert reliably
        
        # Normalize confidence
        confidence = confidence / np.max(confidence) if np.max(confidence) > 0 else confidence
        
        return original_estimate, confidence