import numpy as np
from .cluster import kmeans, define_distance, pairwise_correlations, pre_normalize_distance

def find_best_pairs(X, Y):
    sim = pairwise_correlations(X, Y)
    shape = sim.shape
    if not shape[0] == shape[1]:
        raise ValueError("A square similarity matrix is expected.")
    d = shape[0]
    idx_pairs = []
    top_sims = np.zeros(d)
    i_idx = list(range(d))
    j_idx = list(range(d))
    for c in range(d):
        i, j = np.where(sim == sim.max())  # next best pair
        i0, j0 = i[0], j[0]                # deterministically take the first match, in case of a tie
        idx_pair = [i_idx.pop(i0), j_idx.pop(j0)]
        idx_pairs.append(idx_pair)
        top_sims[idx_pair[0]] = sim[i0, j0]
        # Remove pair
        sim = np.delete(sim, i0, axis=0)
        sim = np.delete(sim, j0, axis=1)
    return idx_pairs, top_sims


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
    """Co-Activation Pattern (CAP) clustering via k-means, supporting correlation, cosine, or
    Euclidean distance.

    For 'correlation'/'cosine' distance, clustering is amplitude-blind by design: samples are
    compared based on their (per-sample) shape rather than absolute magnitude. Internally, this
    is achieved by z-scoring (correlation) or L2-normalizing (cosine) each sample before running
    k-means with Euclidean distance -- mathematically equivalent to correlation/cosine-based
    clustering, but able to use fast, compiled Euclidean distance routines. As a consequence,
    ``cluster_centers_`` is returned in that same normalized space (zero-mean/unit-variance per row
    for 'correlation', unit-norm per row for 'cosine'), not in the original data's raw units.
    For 'euclidean' distance, no normalization is applied and centroids remain in the original
    data scale.

    Parameters
    ----------
    n_clusters : int
        The number of clusters (CAPs) to identify.

    init : str or np.ndarray, default='k-means++'
        Method for centroid initialization: 'k-means++', 'random', or a pre-defined np.ndarray
        of shape (n_clusters, n_features) providing the initial centroids directly.

    distance : str, default='correlation'
        The distance metric used for clustering: one of 'correlation', 'cosine', or 'euclidean'.

    n_rep : int, default=1
        Number of times k-means is run with different centroid initializations (equivalent to
        sklearn's n_init). The repetition with the lowest inertia is kept as the final solution.

    max_iter : int, default=300
        Maximum number of iterations allowed within a single repetition.

    tol : float, default=0.001
        Convergence tolerance, expressed as the fraction of samples reassigned to a different
        cluster between consecutive iterations. A repetition is considered converged once this
        fraction drops to or below tol.

    random_state : int, np.random.RandomState, or None, default=None
        Controls the randomness of centroid initialization. Pass an int for reproducible results.

    centroid_order : str or None, default='Cmodes'
        Method used to reorder clusters after fitting: one of 'hierarchical', 'Cmodes', or None
        (no reordering). See order_centroids_().

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        The fitted cluster centroids, in the same (possibly normalized) space as `distance` --
        see the class description above.

    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignment for each sample in the data used to fit the model.

    inertia_ : float
        Sum of squared distances of samples to their assigned centroid, for the best repetition.

    cross_rep_sim : np.ndarray or None
        Similarity of each non-best repetition's centroids to the best repetition's centroids.
        Set only when n_rep > 1. See cross_rep_robustness().

    Notes
    -----
    This class was first inspired from the code found under
    https://github.com/CoBrALab/mousefmri_acq_publication/blob/main/3_clustering/CAP_functions.py

    Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
    """
    def __init__(self, n_clusters, init="k-means++", distance='correlation', n_rep=1, max_iter=300, tol = 0.001, random_state=None, centroid_order = 'Cmodes'):
        self.n_clusters = n_clusters
        self.init = init
        self.distance = distance
        self.n_rep = n_rep #the number of replicates is num of times to repeat with new initial cluster positions (like n_init in scikit-learn)
        self.max_iter = max_iter #number of iterations in each replicate
        self.tol = tol 
        self.random_state = random_state # if a integer is provided, then the series of random state is deterministic
        self.centroid_order = centroid_order
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.cross_rep_sim = None

    def fit_kmeans(self, X, copy_X=False):
        """Fit the k-means clustering model to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to cluster (e.g. timepoints x voxels).

        copy_X : bool, default=False
            If False, X is temporarily normalized in place during fitting (and restored to its
            original values afterward) to save memory. If True, a copy of X is made instead,
            leaving the caller's original array untouched throughout -- at the cost of doubling
            memory use for the duration of the fit.

        Returns
        -------
        self : CAPs
            The fitted CAPs object, with ``cluster_centers_``, ``labels_``, ``inertia_`` (and, if n_rep>1,
            cross_rep_sim) set.

        Notes
        -----
        Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
        """
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
            
    def transform(self, X):
        """Compute the distance from each sample in X to each fitted cluster centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to compute distances for. Can be passed in raw (un-normalized) form: for
            'correlation'/'cosine' distance, the underlying distance functions normalize both X
            and the centroids internally, so no pre-normalization is required here.

        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_clusters)
            The pairwise distance between each sample and each cluster centroid.

        Notes
        -----
        Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
        """
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centroids have not been computed.")
        compute_distance = define_distance(self.distance)
        distances = compute_distance(X=X, Y=self.cluster_centers_)
        return distances
    
    def predict(self, X):
        """Assign each sample in X to its nearest cluster.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to assign to clusters. Can be passed in raw (un-normalized) form -- see
            transform().

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            The index of the nearest cluster for each sample.

        Notes
        -----
        Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
        """
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

    def reconstruct_timeseries(self, X, prenormalize_X=True):
        """Reconstruct each sample as a scaled version of its assigned cluster centroid.
        A per-sample scalar coefficient is fit by least squares: 
        for each sample x_i assigned to cluster j with centroid m_j, the coefficient c_i 
        minimizing ||x_i - c_i * m_j||^2 is found.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to reconstruct.

        prenormalize_X : bool, default=True
            If True, X is normalized the same way as ``cluster_centers_`` (z-scored per sample for
            'correlation', L2-normalized per sample for 'cosine', unchanged for 'euclidean')
            before reconstruction, so that X and ``cluster_centers_`` are in a comparable scale.
            Leave this as True unless X has already been normalized this way beforehand. 
            This is especially critical for 'correlation' distance: leaving this False will 
            produce a misleading reconstruction, since a zero-mean centroid cannot explain a 
            non-zero-mean signal through a scalar multiple alone.

        Returns
        -------
        recon : np.ndarray of shape (n_samples, n_features)
            The reconstructed data, i.e. coeffs[:, None] * ``cluster_centers_[labels]``.

        coeffs : np.ndarray of shape (n_samples,)
            The per-sample scalar coefficient fit for each sample's assigned centroid.

        labels : np.ndarray of shape (n_samples,)
            The cluster assignment used for reconstruction (``self.labels_`` if it matches X in
            length, otherwise freshly computed via predict(X)).

        X : np.ndarray of shape (n_samples, n_features)
            The data actually used for the reconstruction/coefficient fit -- identical to the
            input if prenormalize_X=False, otherwise the normalized version.

        Notes
        -----
        Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
        """
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centroids have not been computed.")

        if prenormalize_X:  # normalize X to match the centroids, which are normalized
            X, _, _ = pre_normalize_distance(X.copy(), self.distance)

        if self.labels_ is None or len(self.labels_)!=X.shape[0]:
            print("Computing new labels for X.")
            labels = self.predict(X)
        else:
            labels = self.labels_

        coeffs = np.empty(X.shape[0])
        for j in range(self.n_clusters):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                continue

            A = self.cluster_centers_[j].reshape(-1, 1)                      # (P, 1) design
            B = X[idx].T                                         # (P, n_j) targets
            c_j, *_ = np.linalg.lstsq(A, B, rcond=None)          # correct for any ||m_j||
            coeffs[idx] = c_j

        recon = coeffs[:, None] * self.cluster_centers_[labels]
        return recon, coeffs, labels, X

    def estimate_R2(self,X, prenormalize_X=True):
        """Estimate how well the fitted clustering explains the variance in X.

        For each sample, R2 is computed from the residual of its scaled-centroid reconstruction
        (see reconstruct_timeseries()) relative to that sample's total sum of squares, then
        aggregated per cluster and overall.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to evaluate.

        prenormalize_X : bool, default=True
            Whether to normalize X to match ``cluster_centers_`` before reconstruction -- see
            reconstruct_timeseries(). Should be left as True for 'correlation'/'cosine'
            distance; leaving X un-normalized will produce a misleadingly low R2.

        Returns
        -------
        r2_per_cluster : np.ndarray of shape (n_clusters,)
            R2 computed over the samples assigned to each cluster. NaN for any cluster with no
            assigned samples.

        r2_overall : float
            R2 computed over all samples jointly.

        recon : np.ndarray of shape (n_samples, n_features)
            The reconstructed data returned by reconstruct_timeseries().

        Notes
        -----
        Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
        """
        recon, coeffs, labels, X = self.reconstruct_timeseries(X, prenormalize_X=prenormalize_X)

        resid = X - recon

        X_SS = np.sum(X ** 2, axis=1)
        res_SS = np.sum(resid ** 2, axis=1)

        k = self.n_clusters
        r2_per_cluster = np.full(k, np.nan)
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                continue
            r2_per_cluster[j] = 1.0 - res_SS[idx].sum() / X_SS[idx].sum()

        r2_overall = 1.0 - res_SS.sum() / X_SS.sum()

        return r2_per_cluster, r2_overall, recon