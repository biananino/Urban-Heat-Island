# -*- coding: utf-8 -*-
# This file contains functions to calculate spatial weights

import numpy as np
import scipy
from scipy.spatial import KDTree


def row_normalize(distance_matrix):
    """Normalize rows of a sparse distance matrix such that row sums equal 1.

    Useful for converting weighted sums into weighted averages. Zero rows (sum=0) are
    left unchanged to avoid division by zero.

    Parameters
    ----------
    distance_matrix : scipy.sparse.csr_matrix or scipy.sparse.coo_matrix
        Input sparse matrix (weight matrix).

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-normalized sparse matrix in CSR format. Each non-zero row sums to 1.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> W = csr_matrix([[1, 2], [0, 0], [4, 0]])
    >>> W_normalized = row_normalize(W)
    >>> W_normalized.toarray()
    array([[0.33333333, 0.66666667],
           [0.        , 0.        ],
           [1.        , 0.        ]])
    """
    distance_matrix = distance_matrix.tocsr()
    row_sums = distance_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for zero rows
    distance_matrix = distance_matrix.multiply(1.0 / row_sums)
    return distance_matrix

def distance_band(tree1, tree2, threshold, binary=False, distance_metric=2, alpha=-1):
    """Computes a spatial weight matrix between points in two KD-trees within a distance band.

    For each point in `tree1`, calculates weights for neighboring points in `tree2` within
    `threshold` distance. Weights can be binary (1/0) or distance-decayed (using `alpha`).
    Output is a sparse matrix suitable for calculating spatial lag.

    Parameters
    ----------
    tree1, tree2 : scipy.spatial.KDTree
        KD-trees containing point coordinates for source (`tree1`) and target (`tree2`) points.
    threshold : float
        Maximum distance for neighboring points to be included in the weight matrix.
    binary : bool, optional (default=False)
        If True, all weights are binary (1 if distance < `threshold`, else 0).
        If False, applies distance decay using `alpha`.
    distance_metric : int, optional (default=2)
        Minkowski p-norm parameter (e.g., 2=Euclidean, 1=Manhattan).
    alpha : float, optional (default=-1)
        Distance decay exponent (passed to `distance_decay`).
        - If `alpha < 0`, weights are `distance ** alpha`.
        - If `alpha >= 0`, no decay (weights = 1 if `binary=False`).
        Ignored if `binary=True`.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse weight matrix (shape: `(tree1.n, tree2.n)`), where:
        - Rows correspond to points in `tree1`.
        - Columns correspond to points in `tree2`.
        - Non-zero values are weights (binary or distance-decayed).

    Notes
    -----
    1. Zero distances (self-pair) have weights equals to zero.
    To set a weight for self-pairs modify the diagonal after.

    Examples
    --------
    >>> from scipy.spatial import KDTree
    >>> import numpy as np
    >>> # Create KD-trees from point coordinates
    >>> points1 = np.array([[0, 0], [1, 1]])
    >>> points2 = np.array([[0, 0.5], [1, 1.1], [2, 2]])
    >>> tree1, tree2 = KDTree(points1), KDTree(points2)
    >>> # Binary weights (threshold=1.5)
    >>> W_binary = distance_band(tree1, tree2, threshold=1.5, binary=True)
    >>> W_binary.toarray()
    array([[1, 1, 0],
           [1, 1, 1]])
    >>> # Distance-decayed weights (alpha=-1)
    >>> W_decay = distance_band(tree1, tree2, threshold=1.5, binary=False, alpha=-1)
    >>> W_decay.toarray()
    array([[ 2.        ,  0.67267279,  0.        ],
           [ 0.89442719, 10.        ,  0.70710678]])
    """
    dist_matrix = tree1.sparse_distance_matrix(
        tree2, threshold, p=distance_metric, output_type="coo_matrix"
    )
    if binary is True:
        dist_matrix.data = np.where(dist_matrix.data, 1, 0)
    elif binary is False:
        dist_matrix.data = np.where(
            alpha >= 0,
            dist_matrix.data,
            np.float_power(dist_matrix.data, alpha, where=dist_matrix.data > 0),
        )
    return dist_matrix


# noinspection PyTypeChecker
def kernel_weights(tree1, tree2, kernel, bandwidth, eps=1e-7, distance_metric=2):
    """Computes a spatial weight matrix between points in two KD-trees.

    For each point in `tree1`, calculates weights for neighboring points in `tree2` within
    `bandwidth` distance, applying the specified kernel function to decay weights with distance.
    Output is a sparse matrix suitable for calculating spatial lag.

    Parameters
    ----------
    tree1, tree2 : scipy.spatial.KDTree
        KD-trees containing point coordinates for source (`tree1`) and target (`tree2`) points.
    kernel : str
        Kernel function to apply. One of:
        - 'triangular': Linear decay (1 - distance/bandwidth)
        - 'uniform': Constant weight within bandwidth
        - 'quadratic': Epanechnikov kernel (3/4)*(1 - (distance/bandwidth)²)
        - 'quartic': Biweight kernel (15/16)*(1 - (distance/bandwidth)²)²
        - 'gaussian': Normal kernel
    bandwidth : float
        Maximum distance for neighboring points.
        Points beyond (bandwidth + eps) are excluded.
    eps : float, optional (default=1e-7)
        Small buffer added to bandwidth for numerical stability in distance comparisons.
    distance_metric : int, optional (default=2)
        Minkowski p-norm parameter (2=Euclidean, 1=Manhattan).

        Returns
    -------
    scipy.sparse.coo_matrix
        Sparse weight matrix of shape (tree1.n, tree2.n) where:
        - Rows correspond to points in `tree1`
        - Columns correspond to points in `tree2`
        - Non-zero values are kernel-weighted values

    Examples
    --------
    >>> from scipy.spatial import KDTree
    >>> import numpy as np
    >>> # Create sample point sets
    >>> points1 = np.array([[0, 0], [1, 1]])
    >>> points2 = np.array([[0, 0.5], [1, 1.1], [2, 2]])
    >>> tree1, tree2 = KDTree(points1), KDTree(points2)

    >>> # Triangular kernel with bandwidth=1.5
    >>> W_tri = kernel_weights(tree1, tree2, 'triangular', 1.5)
    >>> W_tri.toarray()
    array([[0.66666667, 0.00892875, 0.        ],
           [0.25464401, 0.93333333, 0.05719096]])

    >>> # Gaussian kernel with bandwidth=1.0
    >>> W_gauss = kernel_weights(tree1, tree2, 'gaussian', 1.0)
    >>> np.round(W_gauss.toarray(), 4)
    array([[0.3521, 0.    , 0.    ],
           [0.    , 0.397 , 0.    ]])


    >>> # Quartic kernel with bandwidth=2.0
    >>> W_quart = kernel_weights(tree1, tree2, 'quartic', 2.0)
    >>> np.round(W_quart.toarray(), 4)
    array([[0.824 , 0.1877, 0.    ],
           [0.4431, 0.9328, 0.2344]])
    """
    threshold = bandwidth + eps
    dist_matrix = tree1.sparse_distance_matrix(
        tree2, threshold, p=distance_metric, output_type="coo_matrix"
    )
    dist_matrix.data = dist_matrix.data / bandwidth
    match kernel:
        case "triangular":
            dist_matrix.data = np.where(
                dist_matrix.data <= 1, triangular_kernel(dist_matrix.data), 0
            )
        case "uniform":
            dist_matrix.data = np.where(
                dist_matrix.data <= 1, uniform_kernel(dist_matrix.data), 0
            )
        case "quadratic":
            dist_matrix.data = np.where(
                dist_matrix.data <= 1, quadratic_kernel(dist_matrix.data), 0
            )
        case "quartic":
            dist_matrix.data = np.where(
                dist_matrix.data <= 1, quartic_kernel(dist_matrix.data), 0
            )
        case "gaussian":
            dist_matrix.data = gaussian_kernel(dist_matrix.data)
    return dist_matrix


def triangular_kernel(z):
    """Triangular kernel function for distance weighting.

    Parameters
    ----------
    z : float or array-like
        Normalized distance (scaled to [0, 1] range)

    Returns
    -------
    float or array-like
        Weight values computed as (1 - z)

    Examples
    --------
    >>> triangular_kernel(0.2)
    0.8
    >>> triangular_kernel(np.array([0, 0.5, 1.0]))
    array([1. , 0.5, 0. ])

    Notes
    -----
    Mathematical form:
        K(z) = 1 - z
    """
    return 1 - z


def uniform_kernel(z):
    """Uniform (flat) kernel function for distance weighting.

    Parameters
    ----------
    z : float (unused)
        Normalized distance (scaled to [0, 1] range)

    Returns
    -------
    float
        Weight value computed as (1/2) for all inputs

    Examples
    --------
    >>> uniform_kernel(0.7)
    0.5
    >>> uniform_kernel(np.array([0.1, 0.9]))
    0.5

    Notes
    -----
    Mathematical form:
        K(z) = 1/2
    """
    return 1/2


def quadratic_kernel(z):
    """Quadratic (Epanechnikov) kernel function for distance weighting.

    Parameters
    ----------
    z : float or array-like
        Normalized distance (scaled to [0, 1] range)

    Returns
    -------
    float or array-like
        Weight values computed as (3/4)*(1 - z²)

    Examples
    --------
    >>> quadratic_kernel(0.5)
    0.5625
    >>> quadratic_kernel(np.linspace(0, 1, 3))
    array([0.75  , 0.5625, 0.    ])

    Notes
    -----
    Mathematical form:
        K(z) = (3/4) * (1 - z²)
    """
    return (3 / 4) * (1 - z**2)


def quartic_kernel(z):
    """Quartic (biweight) kernel function for distance weighting.

    Parameters
    ----------
    z : float or array-like
        Normalized distance (scaled to [0, 1] range)

    Returns
    -------
    float or array-like
        Weight values computed as (15/16)*(1 - z²)²

    Examples
    --------
    >>> quartic_kernel(0.6)
    0.384
    >>> quartic_kernel(np.array([0, 0.5, 1.0]))
    array([0.9375    , 0.52734375, 0.        ])
    """
    return (15 / 16) * (1 - z**2) ** 2


def gaussian_kernel(z):
    """Gaussian kernel function for distance weighting.

    Parameters
    ----------
    z : float or array-like
        Normalized distance (scaled to [0, 1] range)

    Returns
    -------
    float or array-like
        Weight values computed using standard normal PDF

    Examples
    --------
    >>> gaussian_kernel(0.5)
    0.3520653267642995
    >>> gaussian_kernel(np.array([0, 0.5, 1.0]))
    array([0.39894228, 0.35206533, 0.24197072])

    Notes
    -----
    Mathematical form:
        K(z) = sqrt(2π) * exp(-z²/2)
    """
    return ((2 * np.pi) ** (-1 / 2)) * np.exp((-(z**2)) / 2)


def knn_matrix(tree, coordinates, k = 1, max_distance = np.inf, binary = True, distance_metric = 2):
    """Calculate a sparse distance matrix containing only the k-nearest neighbors.

    For each point in coordinates, finds its k-nearest neighbors in the KD-tree and
    constructs a sparse distance matrix. Note that when self-querying self-pairs count
    towards k.

    Parameters
    ----------
    tree : scipy.spatial.KDTree
        A KD-tree containing reference points to query against
    coordinates : array-like
        Query points coordinates (n-dimensional)
    k : int, optional
        Number of nearest neighbors to find (default: 1)
    max_distance : float, optional
        Maximum distance to consider for neighbors (default: infinity)
    binary : bool, optional
        If True, all non-zero distances are set to 1 (default: True)
    distance_metric : int, optional
        Which Minkowski p-norm to use (default: 2, Euclidean distance)

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse distance matrix in COO format where:
        - Rows represent query points
        - Columns represent reference points from the tree
        - Non-zero values represent distances (or 1s if binary=True)

    Notes
    -------
    When self-querying and using binary = True the diagonal will consist
    of ones.

    Examples
    --------
    >>> from scipy.spatial import KDTree
    >>> import numpy as np

    # Create a KD-tree with 3 points
    >>> points = np.array([[0, 0], [1, 1], [2, 2]])
    >>> tree = KDTree(points)

    # Query with same points, k=1 neighbors, binary=True
    >>> matrix = knn_matrix(tree, points, k=1)
    >>> matrix.toarray()
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    # Query with k=2 neighbors, binary=False
    #The 0s at the diagonal at different from the zeros in the third column
    >>> matrix = knn_matrix(tree, points, k=2, binary=False)
    >>> np.round(matrix.toarray(), 2)
    array([[0.  , 1.41, 0.  ],
           [1.41, 0.  , 0.  ],
           [0.  , 1.41, 0.  ]])

    # With max_distance constraint
    >>> matrix = knn_matrix(tree, points, k=2, max_distance=1.5, binary=True)
    >>> matrix.toarray()
    array([[1., 1., 0.],
           [1., 1., 0.],
           [0., 1., 1.]])
    """
    distances, neighbours = tree.query(coordinates, k = k, p = distance_metric,
                                     distance_upper_bound = max_distance) #Query tree for KNN
    matrix_nrows = len(coordinates)
    matrix_ncols = tree.n

    mask = np.isfinite(distances)   #To delete infinite distances when they occur
    row = np.repeat(np.arange(matrix_nrows), k)[mask.ravel()]
    col = neighbours.ravel()[mask.ravel()]
    data = distances.ravel()[mask.ravel()]
    if binary is True:
        data = np.ones_like(data)
    #Construct matrix
    distance_matrix = scipy.sparse.coo_matrix( (data, (row, col) ), shape = (matrix_nrows, matrix_ncols))
    return distance_matrix

if __name__ == "__main__":
    import doctest

    doctest.testmod()
