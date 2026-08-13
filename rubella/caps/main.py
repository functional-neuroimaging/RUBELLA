import numpy as np
from .cluster import kmeans, define_distance, pairwise_correlations

def find_best_pairs(X,Y):
    sim = pairwise_correlations(X, Y)
    shape = sim.shape
    if not shape[0]==shape[1]:
        raise ValueError("A square similarity matrix is expected.")
    d = shape[0]
    idx_pairs = []
    top_sims=np.zeros(d)
    i_idx = list(range(d))
    j_idx = list(range(d))
    for c in range(d):
        i,j = np.where(sim==sim.max()) # next best pair
        idx_pair = [i_idx.pop(i[0]),j_idx.pop(j[0])]
        idx_pairs.append(idx_pair)
        top_sims[idx_pair[0]] = sim[i,j]
        # Remove pair
        sim = np.delete(sim, i, axis=0)
        sim = np.delete(sim, j, axis=1)
    return idx_pairs,top_sims

def reorder_hierarchical(centroids): # re-order the list of centroids using hierarchical clustering
    from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
    sim = np.corrcoef(centroids)
    # generate the linkage matrix
    Z = linkage(np.abs(sim), 'ward')
    leaf_order = leaves_list(Z)
    return leaf_order

def reorder_Cmodes(centroids,labels):
    sim = np.abs(np.corrcoef(centroids))
    sim = np.triu(sim, k=1)
    shape = sim.shape
    if not shape[0]==shape[1]:
        raise ValueError("A square similarity matrix is expected.")
    d = shape[0]
    idx_pairs = []
    idx_list = list(range(d))
    while(sim.shape[0]>1):
        i,j = np.where(sim==sim.max()) # next best pair

        # need to pop the highest index first
        if i>j:
            idx_pair = [idx_list.pop(i[0]),idx_list.pop(j[0])]
        else:
            idx_pair = [idx_list.pop(j[0]),idx_list.pop(i[0])]

        idx_pairs.append(idx_pair)
        # Remove pair
        sim = np.delete(sim, (i,j), axis=0) # delete rows from the CAP pair
        sim = np.delete(sim, (i,j), axis=1) # delete columns from the CAP pair
    if d%2==1: # if odd number of CAPs, add the single CAP
        idx_pairs.append(idx_list)

    counts = np.array([(labels==i).sum() for i in range(d)])
    Cmodes_counts = [counts[idx_pair].sum() for idx_pair in idx_pairs]
    # reorder according to occurence rate
    idx_pairs = [idx_pairs[i] for i in np.argsort(Cmodes_counts)[::-1]]

    # re-order according to polarity
    for i in range(len(idx_pairs)):
        polarity_idx = np.argsort(np.array([cap.mean() for cap in centroids[idx_pairs[i]]]))[::-1]
        idx_pairs[i] = [idx_pairs[i][idx] for idx in polarity_idx]
    reordered_idx = [idx for pair in idx_pairs for idx in pair] # flatten the list

    return reordered_idx        

class CAPs:
    # original code for this class taken from https://github.com/CoBrALab/mousefmri_acq_publication/blob/main/3_clustering/CAP_functions.py
    def __init__(self, n_clusters, init="k-means++", distance='correlation', n_rep=1, max_iter=300, tol = 0.001, random_state=None, centroid_order = 'Cmodes'):
        self.n_clusters = n_clusters
        self.init = init
        self.distance = distance
        self.n_rep = n_rep #the number of replicates is num of times to repeat with new initial cluster positions (like n_init in scikit-learn)
        self.max_iter = max_iter #number of iterations in each replicate
        self.tol = tol #convergence tolerance (Frobenius norm between consecutive centroids)
        self.random_state = random_state # if a integer is provided, then the series of random state is deterministic
        self.centroid_order = centroid_order
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.cross_rep_sim = None

    def fit_kmeans(self, X, copy_X=False):
        best_centroids, best_labels, best_inertia, centroids_rep = kmeans(
            X, 
            self.n_clusters, 
            init=self.init, 
            distance=self.distance, 
            n_rep=self.n_rep, 
            max_iter=self.max_iter, 
            tol = self.tol, 
            random_state=self.random_state, 
            copy_X=copy_X)

        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        if self.n_rep>1:
            self.cross_rep_robustness(centroids_rep)
        if not self.centroid_order is None:
            self.order_centroids_()
        return self

    def cross_rep_robustness(self, centroids_list):
        top_sims_l = []
        for i in range(1,len(centroids_list)):
            # robustness is evaluated relative to the best centroid at idx 0
            idx_pairs,top_sims = find_best_pairs(centroids_list[0], centroids_list[i])
            top_sims_l.append(top_sims)
        self.cross_rep_sim = np.array(top_sims_l)
    
    def order_centroids_(self):
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centroids have not been computed.")
        centroids = self.cluster_centers_.copy()
        labels = self.labels_.copy()
        
        if self.centroid_order == 'hierarchical':
            reordered_idx = reorder_hierarchical(centroids)
        elif self.centroid_order == 'Cmodes':
            reordered_idx = reorder_Cmodes(centroids, labels)
        elif self.centroid_order is None: # no re-ordering performed
            return
        else:
            raise ValueError("centroid_order must be 'hierarchical', 'Cmodes' or None.")

        centroids_reordered = centroids[reordered_idx]
        labels_reordered = labels.copy()
        for i in range(self.n_clusters):
            labels_reordered[labels==reordered_idx[i]] = i
        
        self.labels_ = labels_reordered
        self.cluster_centers_ = centroids_reordered
        if self.n_rep>1:
            self.cross_rep_sim = self.cross_rep_sim[:,reordered_idx]
            
    def transform(self, X, distance=None):  # derive cluster distance from new data
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centroids have not been computed.")
        if distance is None:
            distance = self.distance
        compute_distance = define_distance(distance)
        distances = compute_distance(X=X, Y=self.cluster_centers_)
        return distances
    
    def predict(self, X):  # assign cluster labels to new data
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

    def estimate_R2(self,X, output_recon=False):
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centroids have not been computed.")
        if self.labels_ is None or len(self.labels_)!=X.shape[0]:
            print("Computing new labels for X.")
            labels = self.predict(X)
        else:
            labels = self.labels_

        # reconstruct data from labelsXcentroid
        recon_X = np.zeros(X.shape)
        for i in range(self.n_clusters):
            recon_X[labels==i]=self.cluster_centers_[i,:]
        res = X - recon_X
        R2 = 1-res.var()/X.var()
        if output_recon:
            return R2,recon_X
        else:
            return R2
