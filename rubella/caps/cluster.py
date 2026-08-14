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
    """Normalize each row (sample) of X so that squared Euclidean distance on the normalized
    data is exactly linear in correlation or cosine distance on the original data. This lets
    k-means be run with fast, compiled Euclidean distance routines while remaining equivalent
    to clustering directly on correlation/cosine distance.

    For 'correlation', each row is z-scored (its mean subtracted, then divided by its standard
    deviation). For two z-scored rows a, b of length n (each satisfying sum(a**2) = sum(b**2)
    = n, under the population-std convention used here), squared Euclidean distance works out
    to an exact affine function of correlation distance:

        ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 2n - 2*a.b = 2n - 2*n*corr(a, b)
                    = 2n * (1 - corr(a, b))

    For 'cosine', each row is divided by its L2 norm, so ||a|| = ||b|| = 1, and squared
    Euclidean distance reduces to:

        ||a - b||^2 = 2 - 2*a.b = 2 * (1 - cosine_similarity(a, b))

    For 'euclidean', X is returned unchanged (no equivalence transform applies).

    In both cases, squared Euclidean distance is a positive affine function of the target
    distance (scale factor 2n or 2, zero offset), so nearest-centroid assignment (argmin over
    centroids, for a fixed sample) is identical whether computed on squared Euclidean distance,
    plain (non-squared) Euclidean distance, or the correlation/cosine distance directly.

    NOTE: for 'correlation'/'cosine', this function mutates X in place (via a transposed view)
    rather than returning a new array, as a memory-saving optimization for potentially large
    arrays. Pass a copy (X.copy()) if the caller needs to preserve the original values of X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The data to normalize, one row per sample.

    distance : str
        One of 'correlation', 'cosine', or 'euclidean'.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        The normalized data. For 'correlation'/'cosine' this is the same underlying array as
        the input, mutated in place (see note above); for 'euclidean' it is the unmodified input.

    Xvar : np.ndarray of shape (n_samples,) or None
        The per-row scaling factor that was divided out (std for 'correlation', L2 norm for
        'cosine'). None for 'euclidean'. Needed to invert the normalization later.

    Xm : np.ndarray of shape (n_samples,) or None
        The per-row mean that was subtracted out. Only computed for 'correlation'; None
        otherwise. Needed to invert the normalization later.
        
    Notes
    -----
    Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
    """
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
    else:
        raise ValueError("Distance must be one of 'correlation', 'cosine', or 'euclidean'.")
    return X, Xvar, Xm

'''
KMeans
'''

def kmeans(X, n_clusters, init="k-means++", distance='correlation', n_rep=1, max_iter=300, tol = 0.001, random_state=None, copy_X=False):
    """Run k-means clustering, with support for correlation, cosine, or Euclidean distance.

    For 'correlation'/'cosine' distance, X is normalized once up front (see
    pre_normalize_distance()) and clustering is then run internally as Euclidean-distance
    k-means, which is mathematically equivalent to clustering on correlation/cosine distance but
    much faster, since it can use sklearn's compiled Euclidean distance routines. The returned
    centroids remain in this normalized space rather than the original data's raw units -- see
    the CAPs class docstring in main.py for details.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The data to cluster.

    n_clusters : int
        The number of clusters.

    init : str or np.ndarray, default='k-means++'
        Centroid initialization method: 'k-means++', 'random', or a pre-defined array of shape
        (n_clusters, n_features). See init_centroids().

    distance : str, default='correlation'
        One of 'correlation', 'cosine', or 'euclidean'.

    n_rep : int, default=1
        Number of independent repetitions (each with a different centroid initialization) to
        run. The repetition with the lowest inertia is returned as the best solution.

    max_iter : int, default=300
        Maximum number of iterations per repetition.

    tol : float, default=0.001
        Convergence tolerance for each repetition, expressed as the fraction of samples
        reassigned between consecutive iterations. See kmeans_single_rep().

    random_state : int, np.random.RandomState, or None, default=None
        Controls the randomness of centroid initialization across repetitions.

    copy_X : bool, default=False
        If False, X is normalized in place and restored to its original values before this
        function returns, to avoid holding two copies of a potentially large array in memory.
        If True, a copy of X is made up front instead, leaving the caller's original array
        untouched throughout.

    Returns
    -------
    best_centroids : np.ndarray of shape (n_clusters, n_features)
        The centroids of the best (lowest-inertia) repetition.

    best_labels : np.ndarray of shape (n_samples,)
        The cluster assignment of the best repetition.

    best_inertia : float
        The inertia (sum of squared distances to assigned centroid) of the best repetition.

    centroids_rep : list of np.ndarray
        The centroids from every repetition, with the best repetition's centroids at index 0.
        Used e.g. by CAPs.cross_rep_robustness() to assess consistency across repetitions.
        
    Notes
    -----
    Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
    """
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
        centroids = init.copy()  # avoid mutating the caller's array via pre_normalize_distance's in-place ops        
    else:
        raise ValueError(f"init must be 'k-means++', 'random' or a numpy array providing a cluster initialization. Instead {init} was found.")
    return centroids


def kmeans_single_rep(X, n_clusters, init="k-means++", distance='correlation', rep_num=1, max_iter=300, tol=0.001, random_state=None):
    """Run a single k-means repetition (one centroid initialization) to convergence.

    Centroids are recomputed each iteration as the mean of their currently-assigned samples,
    then re-normalized (see pre_normalize_distance()) to keep them in the same space as X, so
    that Euclidean distance between X and the centroids stays equivalent to correlation/cosine
    distance throughout fitting.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The data to cluster. Must already be normalized to match `distance` (see
        pre_normalize_distance()) -- this function does not normalize X itself, since X is
        shared and reused unchanged across every repetition in kmeans().

    n_clusters : int
        The number of clusters.

    init : str or np.ndarray, default='k-means++'
        Centroid initialization method, or a pre-defined array of initial centroids -- see
        init_centroids(). In normal use this is called by kmeans(), which always passes a
        specific centroid array already selected for this repetition.

    distance : str, default='correlation'
        One of 'correlation', 'cosine', or 'euclidean'. Determines how centroids are
        re-normalized after each update (see pre_normalize_distance()).

    rep_num : int, default=1
        Repetition index, used only to label progress output.

    max_iter : int, default=300
        Maximum number of iterations before stopping, regardless of convergence.

    tol : float, default=0.001
        Convergence tolerance, expressed as the fraction of samples reassigned to a different
        cluster between consecutive iterations. The loop stops once this fraction drops to or
        below tol, or once max_iter is reached.

    random_state : int, np.random.RandomState, or None, default=None
        Passed through to init_centroids() for the initial centroid selection.

    Returns
    -------
    M : np.ndarray of shape (n_clusters, n_features)
        The final cluster centroids, in the same (possibly normalized) space as X.

    labels : np.ndarray of shape (n_samples,)
        The final cluster assignment for each sample.

    inertia : float
        The sum of squared distances between each sample and its assigned centroid.
        
    Notes
    -----
    Docstring generated with Claude Sonnet 5 (Anthropic), claude.ai — August 2026.    
    """
    M = init_centroids(X, n_clusters, init, random_state)
    M, _, _ = pre_normalize_distance(M, distance)

    n_samples = X.shape[0]
    # initial assignment from the starting centroids, used as the baseline for the first
    # reassignment-fraction comparison. 
    labels = euclidean_distances(X=X, Y=M, squared=True).argmin(1) # squared=True skips sklearn's sqrt step for computational efficiency

    iter_count = 0
    reassigned_frac = 1.0
    # iteratively update centroid locations until max_iter is reached, or the fraction of
    # samples that changed cluster assignment drops to/below tol
    while (reassigned_frac > tol) and (iter_count < max_iter):
        # Update centroids based on the mean of currently assigned data points
        M = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        M, _, _ = pre_normalize_distance(M, distance)

        # Reassign labels based on the updated centroids
        new_labels = euclidean_distances(X=X, Y=M, squared=True).argmin(1) # squared=True skips sklearn's sqrt step for computational efficiency
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

    # Calculate within-cluster sum of point-to-centroid distances (inertia)
    inertia = np.sum([np.sum((X[labels == i] - M[i])**2) for i in range(n_clusters)])
    return M, labels, inertia