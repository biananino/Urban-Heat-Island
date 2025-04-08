import pandas as pd
import scipy.spatial as ss
from weights import distance_band

# This file contains functions for computing spatial lag
# between 2 GeoDataFrames composed of points.


def count_points(gdf, input, radius, name):
    """
    Count the number of points from input GeoDataFrame within a specified radius
    of each point in the reference GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Reference GeoDataFrame containing points to count around.
    input : GeoDataFrame
        Input GeoDataFrame containing points to count.
    radius : float
        Search radius.
    name : str
        Base name for the output count column.

    Returns
    -------
    DataFrame
        A DataFrame with 'X', 'Y' columns from gdf and a new column containing
        the count of nearby points from input GeoDataFrame. The count column
        is named '{name}_{radius}'.
    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>>
    >>> # Create test data
    >>> ref_df=gpd.GeoDataFrame({
    ...     'X': [0, 2],
    ...     'Y': [0, 2],
    ...     'geometry': [Point(0,0), Point(2,2)]
    ... })
    >>> input_df=gpd.GeoDataFrame({
    ...     'geometry': [Point(1,1), Point(3,3)]
    ... })
    >>>
    >>> # Count points within radius 1.5
    >>> result=count_points(ref_df, input_df, 1.5, 'nearby')
    >>> list(result['nearby_1.5'])
    [1, 2]
    >>>
    >>> # Count points within radius 1.0
    >>> result=count_points(ref_df, input_df, 1.0, 'nearby')
    >>> list(result['nearby_1.0'])
    [0, 0]
    """
    # Preprocess data
    gdf_coords=list(zip(gdf.geometry.x, gdf.geometry.y))
    input_coords=list(zip(input.geometry.x, input.geometry.y))
    # Build tree
    gdf_tree=ss.KDTree(gdf_coords)
    input_tree=ss.KDTree(input_coords)
    # Get distance band
    dist=distance_band(gdf_tree, input_tree, radius, binary=True, distance_metric=2)
    count=pd.DataFrame(dist.sum(axis=1))
    count.columns=[f'{name}_{radius}']
    output=pd.concat([gdf[["X", "Y"]], count], axis=1)
    return output


if __name__ == "__main__":
    import doctest

    doctest.testmod()
