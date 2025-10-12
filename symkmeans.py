import faiss
import numpy as np

class SKM():
    def __init__(self, d, k, sym_plus_n_func, nsym, niter=300):
        self.d = d
        self.k = k
        self.niter = niter
        self.sym_plus_n_func = sym_plus_n_func
        self.nsym = nsym
        self.centroids = None
        self.assignments = None
        
    def sym_func(self, X):
        syms = [self.sym_plus_n_func(X, i) for i in range(self.nsym)]
        return np.vstack(syms)

    def fit(self, X):
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
        _, self.assignments = index.search(sym_X, 1)
        self.assignments = self.assignments.flatten()
        self.y = self.assignments
        return self
    
    def get_inverse_centers(self):
        return self.centroids
