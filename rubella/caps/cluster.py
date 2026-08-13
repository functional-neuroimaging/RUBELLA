import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

'''
Similarity/distance metrics
'''
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


def pre_normalize_distance(X, distance):
    # to leverage euclidean distance computational speed,
    # data is normalized so that correlation/cosine 
    # distance equivalence is attained with 
    # euclidean distance
    if distance=='correlation':
        # for compatibility with correlation, data is z-scored at each frame
        X = X.T
        Xm = X.mean(axis=0)
        X -= Xm
        Xvar = X.std(axis=0)
        X /= Xvar
        X = X.T
    elif distance=='cosine':
        # for compatibility with cosine, data is normalized with l2-norm at each frame
        X = X.T
        Xvar = np.sqrt((X ** 2).sum(axis=0))
        X /= Xvar
        X = X.T
        Xm = None
    elif distance=='euclidean':
        Xvar = Xm = None
    return X, Xvar, Xm

'''
KMeans
'''

def kmeans(X, n_clusters, init="k-means++", distance='correlation', n_rep=1, max_iter=300, tol = 0.001, random_state=None, copy_X=False):
    best_labels, best_centroids, best_inertia = None, None, np.inf
    
    random_state = check_random_state(random_state)

    if copy_X:
        X = X.copy() # run everything on a copy to avoid modifying the original data

    # data is normalized to leverage euclidean distance computational speed
    # while preserving equivalence for correlation/cosine distance
    X, Xvar, Xm = pre_normalize_distance(X, distance)
    
    centroids_rep = []
    labels_rep = []
    for i in range(n_rep):
        print(f"Running kmeans repetition #{i+1}")

        centroids_init = init_centroids(X,n_clusters,init,random_state)
        centroids, labels, inertia = kmeans_single_rep(X, n_clusters, init=centroids_init, distance=distance, rep_num=i+1, max_iter=max_iter, tol = tol, random_state=random_state)

        # Update best solution if the current one has lower inertia
        if inertia < best_inertia:
            best_labels, best_centroids, best_inertia = labels, centroids, inertia
            centroids_rep = [centroids] + centroids_rep # the best centroid will be at idx 0
            labels_rep = [labels] + labels_rep # the best centroid will be at idx 0
        else:
            centroids_rep.append(centroids)
            labels_rep.append(labels)

    if not copy_X:
        # restore the original X if modified
        if distance=='correlation':
            X = X.T
            X *= Xvar
            X += Xm
            X = X.T
        elif distance=='cosine':
            X = X.T
            X *= Xvar
            X = X.T
    return best_centroids, best_labels, best_inertia, centroids_rep


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


def kmeans_single_rep(X, n_clusters, init="k-means++", distance='correlation', rep_num=1, max_iter=300, tol=0.001, random_state=None):
    M = init_centroids(X, n_clusters, init, random_state)
    M, _, _ = pre_normalize_distance(M, distance)

    n_samples = X.shape[0]
    # initial assignment from the starting centroids, used as the baseline for the first
    # reassignment-fraction comparison
    labels = euclidean_distances(X=X, Y=M).argmin(1)

    iter_count = 0
    reassigned_frac = 1.0
    # iteratively update centroid locations until max_iter is reached, or the fraction of
    # samples that changed cluster assignment drops to/below tol
    while (reassigned_frac > tol) and (iter_count < max_iter):
        # Update centroids based on the mean of currently assigned data points
        M = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        M, _, _ = pre_normalize_distance(M, distance)

        # Reassign labels based on the updated centroids
        new_labels = euclidean_distances(X=X, Y=M).argmin(1)
        reassigned_frac = np.sum(new_labels != labels) / n_samples
        labels = new_labels

        # update the iteration number
        iter_count = iter_count + 1

        # print the convergence situation
        if reassigned_frac <= tol:
            print("\nInitialization #" + str(rep_num) + " converged on iteration " + str(iter_count) + ", with " + f"{reassigned_frac:.2%}" + " of labels reassigned.")
        elif iter_count == max_iter:
            print("\nInitialization #" + str(rep_num) + " reached max iteration " + str(iter_count) + ", with " + f"{reassigned_frac:.2%}" + " of labels reassigned.")
        else:
            print("\r", f"Initialization #{str(rep_num)} - iteration {str(iter_count)}: {reassigned_frac:.2%} of labels reassigned", end="")

    # Calculate within-cluster mean of point-to-centroid distances (inertia)
    inertia = np.mean([np.mean((X[labels == i] - M[i])**2) for i in range(n_clusters)])
    return M, labels, inertia