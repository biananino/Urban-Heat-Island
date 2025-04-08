import pandas as pd
import scipy.spatial as ss
from weights import knn_matrix, distance_band, row_normalize, kernel_weights

# This file contains functions to calculate spatial lag between a raster data array and
# a GeoDataFrame containing points.


def calculate_knn_lag(band_data, gdf, k=1, normalize=True):
    """
    Calculate spatial lag values using K-nearest neighbors (KNN) weights.

    Parameters
    ----------
    band_data : xarray.DataArray
        Input raster data with coordinates and bands.
    gdf : GeoDataFrame
        Reference points where lag values will be calculated.
    k : int, optional
        Number of nearest neighbors to consider (default=1).
    normalize : bool, optional
        If True, normalizes weights to sum to 1 for each point. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - Original coordinates ('X', 'Y') from input GeoDataFrame
        - Lagged band values (columns named '{band}_K{k}')

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> import xarray as xr
    >>> from shapely.geometry import Point

    >>> # Create test raster data (2x2 grid)
    >>> x_coords=[0.5, 1.5]
    >>> y_coords=[0.5, 1.5]
    >>> data=np.array([[1, 2], [3, 4]])  # Single band with 2x2 values
    >>> band_data = xr.DataArray(
    ...     data,
    ...     dims=('x', 'y'),
    ...     coords={
    ...         'x': x_coords,
    ...         'y': y_coords,
    ...     },
    ...     name='values'
    ... )

    >>> # Add a spatial ref coordinate (it's always added by rioxarray)
    >>> band_data = band_data.assign_coords({'spatial_ref': 0})

    >>> # Create test points
    >>> gdf = gpd.GeoDataFrame({
    ...     'geometry': [Point(0,0), Point(1,1), Point(2,2)]
    ... })

    >>> # Test K=3 without normalization. Ties are broken by index.
    >>> result = calculate_knn_lag(band_data, gdf, k=3, normalize=False)
    >>> result['values_K3'].values
    array([6., 6., 9.])

    >>> # Test K=3 with normalization. Ties are broken by index.
    >>> result = calculate_knn_lag(band_data, gdf, k=3, normalize=True)
    >>> result['values_K3'].values
    array([2., 2., 3.])
    """
    # Preprocess raster data
    raster_coords = band_data.to_dataframe().reset_index()[["x", "y"]].values
    raster_bands = (
        band_data.to_dataframe().reset_index().drop(columns=["x", "y", "spatial_ref"])
    )
    raster_columns = raster_bands.columns
    raster_bands = raster_bands.values
    vector_coords = pd.DataFrame({'X': gdf.geometry.x, 'Y': gdf.geometry.y})
    # Build tree
    raster_tree = ss.KDTree(raster_coords)
    # Get KNN
    knn = knn_matrix(raster_tree, vector_coords, k=k, binary=True)
    if normalize is True:
        knn = row_normalize(knn)
    # Lag
    knn_bands = pd.DataFrame(knn @ raster_bands)
    knn_bands.columns = map(lambda x: f"{x}_K{k}", list(raster_columns))
    # Join data
    final_data = pd.concat([vector_coords, knn_bands], axis=1)
    return final_data


def calculate_band_lag(
    band_data,
    gdf,
    threshold,
    binary=False,
    distance_metric=2,
    alpha=-1,
    normalize=True,
    n_points=False,
):
    """
    Calculate spatial lag values for raster bands within a specified distance threshold.

    For each point in a GeoDataFrame, computes a distance-weighted (or binary) spatial
    lag of raster values within a given neighborhood.

    Parameters
    ----------
    band_data : xarray.DataArray
        Input raster data with coordinates and bands.
    gdf : GeoDataFrame
        Reference points where lag values will be calculated.
    threshold : float
        Distance threshold for neighborhood inclusion.
    binary : bool, optional (default=False)
        If True, all weights are binary (1 if distance < `threshold`, else 0).
        If False, applies distance decay using `alpha`.
    distance_metric : float, optional (default=2)
        Minkowski p-norm parameter (e.g., 2=Euclidean, 1=Manhattan).
    alpha : float, optional (default=-1)
        Distance decay exponent (passed to `distance_decay`).
        - If `alpha < 0`, weights are `distance ** alpha`.
        - If `alpha >= 0`, no decay (weights = 1 if `binary=False`).
        Ignored if `binary=True`.
    normalize : bool, optional
        If True, normalizes weights to sum to 1 for each point. Default is True.
    n_points : bool, optional
        If True, adds a column with the count of points within the threshold.
        Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - Original coordinates ('X', 'Y') from input GeoDataFrame
        - Lagged band values (columns named '{band}_R{threshold}')
        - Optional 'n_points' column if n_points=True

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> import xarray as xr
    >>> from shapely.geometry import Point
    >>> from itertools import product

    >>> # Create test raster data (4x4 grid)
    >>> x_coords = [5,10,15,20]
    >>> y_coords = [5,10,15,20]
    >>> # Single band with 4x4 values
    >>> data = np.array([[1,2,3,4],[5,6,7,8],
    ...        [9,10,11,12],[13,14,15,16]], dtype = 'float64')
    >>> band_data = xr.DataArray(
    ...     data,
    ...     dims=('x', 'y'),
    ...     coords={
    ...         'x': x_coords,
    ...         'y': y_coords,
    ...     },
    ...     name='values'
    ... )

    >>> # Add a spatial ref coordinate (it's always added by rioxarray)
    >>> band_data = band_data.assign_coords({'spatial_ref': 0})

    >>> # Create same test points as band points
    >>> gdf = gpd.GeoDataFrame({
    ...      'geometry': [Point(z) for z in product(x_coords,y_coords)]
    ... })

    >>> # Binary lag without normalization.
    >>> result = calculate_band_lag(band_data, gdf, threshold = 7, binary = True,
    ...                             normalize = False)
    >>> result['values_R7'].values
    array([ 7., 10., 13., 11., 16., 24., 28., 23., 28., 40., 44., 35., 23.,
           38., 41., 27.])

    >>> # Distance decay lag.
    >>> result = calculate_band_lag(band_data, gdf, threshold = 7, alpha = -1,
    ...                             normalize = False)
    >>> result['values_R7'].values
    array([1.4, 2. , 2.6, 2.2, 3.2, 4.8, 5.6, 4.6, 5.6, 8. , 8.8, 7. , 4.6,
           7.6, 8.2, 5.4])

    >>> # Lag with gravity model weights
    >>> result = calculate_band_lag(band_data, gdf, threshold = 7, alpha = -2,
    ...                             normalize = False)
    >>> result['values_R7'].values
    array([0.28, 0.4 , 0.52, 0.44, 0.64, 0.96, 1.12, 0.92, 1.12, 1.6 , 1.76,
           1.4 , 0.92, 1.52, 1.64, 1.08])
    """
    # Preprocess raster data
    raster_coords = band_data.to_dataframe().reset_index()[["x", "y"]].values
    raster_bands = (
        band_data.to_dataframe().reset_index().drop(columns=["x", "y", "spatial_ref"])
    )
    raster_columns = raster_bands.columns
    raster_bands = raster_bands.values
    vector_coords = pd.DataFrame({'X': gdf.geometry.x, 'Y': gdf.geometry.y})
    # Build tree
    raster_tree = ss.KDTree(raster_coords)
    gdf_tree = ss.KDTree(vector_coords)
    # Get distance band
    dist = distance_band(
        gdf_tree, raster_tree, threshold, binary, distance_metric, alpha
    )
    if normalize is True:
        dist = row_normalize(dist)
    # Lag
    dist_bands = pd.DataFrame(dist @ raster_bands)
    dist_bands.columns = map(lambda x: f"{x}_R{threshold}", list(raster_columns))
    if n_points is True:
        # Add number of points
        dist_bands["n_points"] = dist.getnnz(axis=1)
    # Join data
    final_data = pd.concat([vector_coords, dist_bands], axis=1)
    return final_data


def calculate_kernel_lag(
    band_data,
    gdf,
    kernel,
    bandwidth,
    eps=1e-7,
    distance_metric=2,
    normalize=False,
    n_points=False,
):
    """Calculate kernel-weighted spatial lag values for raster bands within a specified
    bandwidth.

    For each point in a GeoDataFrame, computes a kernel-weighted spatial lag of raster
    values using a specified kernel function and bandwidth.

    Parameters
    ----------
    band_data : xarray.DataArray
        Input raster data with coordinates and bands.
    gdf : GeoDataFrame
        Reference points where lag values will be calculated.
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
    distance_metric : float, optional (default=2)
        Minkowski p-norm parameter (e.g., 2=Euclidean, 1=Manhattan).
    normalize : bool, optional
        If True, normalizes weights to sum to 1 for each point. Default is False.
    n_points : bool, optional
        If True, adds a column with the count of points within the threshold.
        Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - Original coordinates ('X', 'Y') from input GeoDataFrame
        - Kernel-weighted lagged band values
        (columns named '{band}_{kernel}_{bandwidth}')
        - Optional 'n_points' column if n_points=True

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> import xarray as xr
    >>> from shapely.geometry import Point
    >>> from itertools import product

    >>> # Create test raster data (4x4 grid)
    >>> x_coords = [5,10,15,20]
    >>> y_coords = [5,10,15,20]
    >>> # Single band with 4x4 values
    >>> data = np.array([[1,2,3,4],[5,6,7,8],
    ...        [9,10,11,12],[13,14,15,16]], dtype = 'float64')
    >>> band_data = xr.DataArray(
    ...     data,
    ...     dims=('x', 'y'),
    ...     coords={
    ...         'x': x_coords,
    ...         'y': y_coords,
    ...     },
    ...     name='values'
    ... )

    >>> # Add a spatial ref coordinate (it's always added by rioxarray)
    >>> band_data=band_data.assign_coords({'spatial_ref': 0})

    >>> # Create same test points as band points
    >>> gdf=gpd.GeoDataFrame({
    ...      'geometry': [Point(z) for z in product(x_coords,y_coords)]
    ... })

    >>> # Quadratic kernel.
    >>> result = calculate_kernel_lag(band_data, gdf, bandwidth=7, kernel='quadratic')
    >>> np.round(result['values_quadratic_7'].values, 3)
    array([ 3.321,  5.173,  7.026,  7.041,  9.628, 13.316, 15.536, 14.449,
           17.036, 22.194, 24.413, 21.857, 18.199, 24.459, 26.311, 21.918])

    >>> # Gaussian kernel
    >>> result = calculate_kernel_lag(band_data, gdf, bandwidth=7, kernel='gaussian')
    >>> np.round(result['values_gaussian_7'].values, 3)
    array([ 2.563,  3.889,  5.215,  4.996,  6.941,  9.812, 11.448, 10.301,
           12.246, 16.354, 17.989, 15.606, 12.296, 17.332, 18.658, 14.729])
    """
    # Preprocess raster data
    raster_coords = band_data.to_dataframe().reset_index()[["x", "y"]].values
    raster_bands = (
        band_data.to_dataframe().reset_index().drop(columns=["x", "y", "spatial_ref"])
    )
    raster_columns = raster_bands.columns
    raster_bands = raster_bands.values
    vector_coords = pd.DataFrame({'X': gdf.geometry.x, 'Y': gdf.geometry.y})
    # Build tree
    raster_tree = ss.KDTree(raster_coords)
    gdf_tree = ss.KDTree(vector_coords)
    # Get kernel weights
    kern = kernel_weights(
        gdf_tree,
        raster_tree,
        kernel=kernel,
        bandwidth=bandwidth,
        eps=eps,
        distance_metric=distance_metric,
    )
    if normalize is True:
        kern = row_normalize(kern)
    # Lag
    kern_bands = pd.DataFrame(kern @ raster_bands)
    kern_bands.columns = map(
        lambda x: f"{x}_{kernel}_{bandwidth}", list(raster_columns)
    )
    if n_points is True:
        # Add number of points
        kern_bands["n_points"] = kern.getnnz(axis=1)
    # Join data
    final_data = pd.concat([vector_coords, kern_bands], axis=1)
    return final_data


def calculate_ring_lag(
    band_data,
    gdf,
    inner_radius,
    outer_radius,
    distance_metric=2,
    normalize=True,
    n_points=False,
):
    """
    Calculate spatial lag values for raster bands within a ring-shaped neighborhood.

    For each point in a GeoDataFrame, computes a distance-weighted spatial lag of raster
    values within a ring defined by an inner and outer radius.

    Parameters
    ----------
    band_data : xarray.DataArray
        Input raster data with coordinates and bands.
    gdf : GeoDataFrame
        Reference points where lag values will be calculated.
    inner_radius : float
        Inner radius of the ring (exclusive lower bound).
    outer_radius : float
        Outer radius of the ring (inclusive upper bound).
    distance_metric : float, optional (default=2)
        Minkowski p-norm parameter (e.g., 2=Euclidean, 1=Manhattan).
    normalize : bool, optional
        If True, normalizes weights to sum to 1 for each point. Default is True.
    n_points : bool, optional
        If True, adds a column with the count of points within the ring.
        Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - Original coordinates ('X', 'Y') from input GeoDataFrame
        - Lagged band values (columns named '{band}_{inner_radius}_{outer_radius}')
        - Optional 'n_points' column if n_points=True

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> import xarray as xr
    >>> from shapely.geometry import Point
    >>> from itertools import product

    >>> # Create test raster data (4x4 grid)
    >>> x_coords = [5,10,15,20]
    >>> y_coords = [5,10,15,20]
    >>> # Single band with 4x4 values
    >>> data = np.array([[1,2,3,4],[5,6,7,8],
    ...        [9,10,11,12],[13,14,15,16]], dtype = 'float64')
    >>> band_data = xr.DataArray(
    ...     data,
    ...     dims=('x', 'y'),
    ...     coords={
    ...         'x': x_coords,
    ...         'y': y_coords,
    ...     },
    ...     name='values'
    ... )

    >>> # Add a spatial ref coordinate (it's always added by rioxarray)
    >>> band_data=band_data.assign_coords({'spatial_ref': 0})

    >>> # Create same test points as band points
    >>> gdf=gpd.GeoDataFrame({
    ...      'geometry': [Point(z) for z in product(x_coords,y_coords)]
    ... })

    >>> # Ring between 7 and 15
    >>> result = calculate_ring_lag(band_data, gdf, inner_radius = 7, outer_radius = 15)
    >>> np.round(result['values_7_15'].values, 3)
    array([7.875, 8.889, 8.556, 8.125, 9.222, 9.636, 9.182, 9.111, 7.889,
           7.818, 7.364, 7.778, 8.875, 8.444, 8.111, 9.125])
    """
    # Preprocess raster data
    raster_coords = band_data.to_dataframe().reset_index()[["x", "y"]].values
    raster_bands = (
        band_data.to_dataframe().reset_index().drop(columns=["x", "y", "spatial_ref"])
    )
    raster_columns = raster_bands.columns
    raster_bands = raster_bands.values
    vector_coords = pd.DataFrame({'X': gdf.geometry.x, 'Y': gdf.geometry.y})
    # Build tree
    raster_tree = ss.KDTree(raster_coords)
    gdf_tree = ss.KDTree(vector_coords)
    # get distance bands
    inner_band = distance_band(
        gdf_tree,
        raster_tree,
        inner_radius,
        binary=True,
        distance_metric=distance_metric,
    )
    outer_band = distance_band(
        gdf_tree,
        raster_tree,
        outer_radius,
        binary=True,
        distance_metric=distance_metric,
    )
    ring = outer_band - inner_band.multiply(outer_band)
    if normalize is True:
        ring = row_normalize(ring)
    # Lag
    ring_bands = pd.DataFrame(ring @ raster_bands)
    ring_bands.columns = map(
        lambda x: f"{x}_{inner_radius}_{outer_radius}", list(raster_columns)
    )
    if n_points is True:
        # Add number of points
        ring_bands["n_points"] = ring.getnnz(axis=1)
    # Join data
    final_data = pd.concat([vector_coords, ring_bands], axis=1)
    return final_data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
