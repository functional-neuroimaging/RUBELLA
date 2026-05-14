import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def pairwise_correlations(X, Y):  # return a correlation between each row of X with each row of Y
    # implementation is faster than sklearn.metrics.pairwise_distances
    X_ = X.T.copy()
    Y_ = Y.T.copy()
    X_ -= X_.mean(axis=0)
    Y_ -= Y_.mean(axis=0)
    r_num = X_.T.dot(Y_)
    r_den = np.sqrt(np.sum((X_)**2, axis=0).reshape(-1,1)*np.sum((Y_)**2, axis=0).reshape(1,-1))
    r = r_num/r_den
    return r

def correlation_distances(X, Y):
    return 1 - pairwise_correlations(X, Y)

def cosine_similarity(X,Y): 
    # estimate cosine similarity across the columns of X and Y
    # implementation similar speed as sklearn.metrics.pairwise_distances
    X_ = X.T.copy()
    X_ /= np.sqrt((X_ ** 2).sum(axis=0)) # first normalize from l2-norm
    Y_ = Y.T.copy()
    Y_ /= np.sqrt((Y_ ** 2).sum(axis=0)) # first normalize from l2-norm

    Sc = (X_.T.dot(Y_)) # then derive vector product
    return Sc

def cosine_distances(X,Y): 
    return 1 - cosine_similarity(X,Y)

def define_distance(distance):
    if isinstance(distance, str) and distance == "correlation":
        return correlation_distances
    elif isinstance(distance, str) and distance == "euclidean":
        return euclidean_distances
    elif isinstance(distance, str) and distance == "cosine":
        return cosine_distances
    else:
        raise ValueError(f"distance must be one of 'correlation','euclidean'")

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
    def __init__(self, n_clusters, init="k-means++", distance='correlation', n_rep=1, max_iters=300, tol = 1e-4, random_state=None, centroid_order = 'Cmodes'):
        self.n_clusters = n_clusters
        self.init = init
        self.distance = distance
        self.n_rep = n_rep #the number of replicates is num of times to repeat with new initial cluster positions (like n_init in scikit-learn)
        self.max_iters = max_iters #number of iterations in each replicate
        self.tol = tol #convergence tolerance (Frobenius norm between consecutive centroids)
        self.random_state = random_state # if a integer is provided, then the series of random state is deterministic
        self.centroid_order = centroid_order
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.cross_rep_sim = None

    def fit_kmeans(self, X, implementation='sklearn', copy_X=False):
        best_labels, best_centroids, best_inertia = None, None, np.inf
        
        from sklearn.utils import check_random_state
        random_state = check_random_state(self.random_state)

        if copy_X:
            X = X.copy() # run everything on a copy to avoid modifying the original data
        
        if implementation=='sklearn':
            # for compatibility between sklearn's euclidean distance and 
            # a correlation/cosine distance optimization, the data must 
            # be normalized before running sklearn's kmeans
            if self.distance=='correlation':
                # for compatibility with correlation, data is z-scored at each frame
                X = X.T
                X_m = X.mean(axis=0)
                X -= X_m
                X_std = X.std(axis=0)
                X /= X_std
                X = X.T
            elif self.distance=='cosine':
                # for compatibility with cosine, data is normalized with l2-norm at each frame
                X = X.T
                X_l2 = np.sqrt((X ** 2).sum(axis=0))
                X /= X_l2
                X = X.T

        centroids_rep = []
        labels_rep = []
        for i in range(self.n_rep):
            print(f"Running kmeans repetition #{i+1}")

            centroids_init = init_centroids(X,self.n_clusters,self.init,random_state)

            if implementation=='sklearn':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_clusters, init=centroids_init, n_init=1, max_iter=self.max_iters, tol=self.tol, random_state=random_state, verbose=1, copy_x=False).fit(X)
                centroids = kmeans.cluster_centers_
                labels = kmeans.labels_
                inertia = kmeans.inertia_
            elif implementation=='custom':
                centroids, labels, inertia = custom_kmeans_single_rep(X, self.n_clusters, init=centroids_init, distance=self.distance, rep_num=i+1, max_iters=self.max_iters, tol = self.tol, random_state=random_state)

            # Update best solution if the current one has lower inertia
            if inertia < best_inertia:
                best_labels, best_centroids, best_inertia = labels, centroids, inertia
                centroids_rep = [centroids] + centroids_rep # the best centroid will be at idx 0
                labels_rep = [labels] + labels_rep # the best centroid will be at idx 0
            else:
                centroids_rep.append(centroids)
                labels_rep.append(labels)

        # restore the original X if modified, and recompute the centroids so that they match
        if implementation=='sklearn':
            if self.distance=='correlation':
                X = X.T
                X *= X_std
                X += X_m
                X = X.T
            elif self.distance=='cosine':
                X = X.T
                X *= X_l2
                X = X.T
            # recompute the centroids from the labels
            centroids_rep = []
            for labels in labels_rep:
                centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
                centroids_rep.append(centroids)
            best_centroids = centroids_rep[0]
        
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

    def var_explained(self,X, output_recon=False):
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centroids have not been computed.")
        # reconstruct data from labelsXcentroid
        recon_X = np.zeros(X.shape)
        for i in range(self.n_clusters):
            recon_X[self.labels_==i]=self.cluster_centers_[i,:]
        res = X - recon_X
        VE = 1-res.var()/X.var()
        if output_recon:
            return VE,recon_X
        else:
            return VE


def custom_kmeans_single_rep(X, n_clusters, init="k-means++", distance='correlation', rep_num=1, max_iters=300, tol = 1e-4, random_state=None):
    compute_distance = define_distance(distance)
    centroids = init_centroids(X,n_clusters,init,random_state)

    iter_count = 0 
    centers_squared_diff = 1
    #iteratively update the location of the centroid until the number of max iterations is reached, or centroid location barely changes (tolerance)
    while (centers_squared_diff > tol) and (iter_count < max_iters):
        # Assign each data point to the nearest centroid based on correlation distance
        distances = compute_distance(X=X, Y=centroids)
        labels = np.argmin(distances, axis=1)

        # Update centroids based on the mean of assigned data points
        centroids_new = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        #compute the tolerance (how different the new centroids are compared to the previous iteration) using squared Frobenius norm
        centers_squared_diff = np.sum((centroids_new - centroids) ** 2)
        centroids = centroids_new
        
        #update the iteration number
        iter_count = iter_count + 1
        
        #print the convergence situation
        if centers_squared_diff <= tol:
            print("\nInitialization #" + str(rep_num) + " converged on iteration " + str(iter_count) + " as tolerance of " + str(centers_squared_diff) + " was reached.")
        elif iter_count == max_iters:
            print("\nInitialization #" + str(rep_num) + " converged at max iteration " + str(iter_count) + " at tolerance of " + str(centers_squared_diff) + ".")
        else:
            print("\r",f"Initialization #{str(rep_num)} - iteration {str(iter_count)}: tolerance = {str(centers_squared_diff)}", end="")

    # Calculate within-cluster sums of point-to-centroid distances (inertia)
    ### Note that inertia computations are based on euclidean distance, even if the clustering is performed based on correlation/cosine distance
    inertia = np.sum([np.sum((X[labels == i] - centroids[i])**2) for i in range(n_clusters)])
    return centroids, labels, inertia


def init_centroids(X,n_clusters,init,random_state):
    if isinstance(init, str) and init == "k-means++":
        from sklearn.cluster._kmeans import kmeans_plusplus
        centroids, _ = kmeans_plusplus(
            X,n_clusters,random_state=random_state,
            x_squared_norms=None,
            n_local_trials=None,
        )
    elif isinstance(init, str) and init == "random":
        # centroids are initialized as a random volume from the input data
        centroids = X[random_state.choice(X.shape[0], n_clusters, replace=False)]
    elif isinstance(init, np.ndarray):
        expected_shape = (n_clusters,X.shape[1])
        if not init.shape==expected_shape:
            raise ValueError(f"The inputed cluster centroids of shape {init.shape} does not match the expected shape of {expected_shape}.")
        centroids = init
    else:
        raise ValueError(f"init must be 'k-means++', 'random' or a numpy array providing a cluster initialization. Instead {init} was found.")
    return centroids
