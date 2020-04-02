from typing import List, Union

import pandas as pd
import random

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.neighbors import NearestNeighbors
from vresutils.graph import voronoi_partition_pts
import networkx as nx
import numpy as np

from src.data.geographics.manager import return_points_in_shape


def plot_topology(buses: pd.DataFrame, lines: pd.DataFrame = None):
    """
    Plots a map with buses and lines.

    Parameters
    ----------
    buses: pd.DataFrame
        DataFrame with columns 'x', 'y' and 'region'
    lines: pd.DataFrame (default: None)
        DataFrame with columns 'bus0', 'bus1' whose values must be index of 'buses'.
        If None, do not display the lines.
    """

    # Fill the countries with one color
    def get_xy(shape):
        # Get a vector of latitude and longitude
        xs = [i for i, _ in shape.exterior.coords]
        ys = [j for _, j in shape.exterior.coords]
        return xs, ys

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plotting the buses
    for idx in buses.index:

        color = (random.random(), random.random(), random.random())

        # If buses are associated to regions, display the region
        if 'region' in buses.columns:
            region = buses.loc[idx].region
            if isinstance(region, MultiPolygon):
                for polygon in region:
                    x, y = get_xy(polygon)
                    ax.fill(x, y, c=color, alpha=0.3)
            elif isinstance(region, Polygon):
                x, y = get_xy(region)
                ax.fill(x, y, c=color, alpha=0.3)

        # Plot the bus position
        ax.scatter(buses.loc[idx].x, buses.loc[idx].y, c=[color], marker="s")

    # Plotting the lines
    if lines is not None:
        for idx in lines.index:

            bus0 = lines.loc[idx].bus0
            bus1 = lines.loc[idx].bus1
            if bus0 not in buses.index or bus1 not in buses.index:
                print(f"Warning: not showing line {idx} because missing bus {bus0} or {bus1}")
                continue

            color = 'r' if 'carrier' in lines.columns and lines.loc[idx].carrier == "DC" else 'b'
            plt.plot([buses.loc[bus0].x, buses.loc[bus1].x], [buses.loc[bus0].y, buses.loc[bus1].y], c=color)


def voronoi_special(shape: Union[Polygon, MultiPolygon], centroids: List[List[float]], resolution: float = 0.5):
    """This function applies a special Voronoi partition of a non-convex polygon based on
    an approximation of the geodesic distance to a set of points which define the centroids
    of each partition.

    Parameters
    ----------
    shape: Union[Polygon, MultiPolygon]
        Non-convex shape
    centroids: List[List[float]], shape: Nx2
        List of coordinates
    resolution: float (default: 0.5)
        The smaller this value the more precise the geodesic approximation
    Returns
    -------
    List of N Polygons
    """

    # Get all the points in the shape at a certain resolution
    points = return_points_in_shape(shape, resolution)

    # Build a network from these points where each points correspond to a node
    #   and each points is connected to its adjacent points
    adjacency_matrix = np.zeros((len(points), len(points)))
    for i, c_point in enumerate(points):
        adjacency_matrix[i, :] = \
            [1 if np.abs(c_point[0]-point[0]) <= resolution and np.abs(c_point[1]-point[1]) <= resolution else 0
             for point in points]
        adjacency_matrix[i, i] = 0.0
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Find the closest node in the graph corresponding to each centroid
    nbrs = NearestNeighbors(n_neighbors=1).fit(points)
    _, idxs = nbrs.kneighbors(centroids)
    centroids_nodes_indexes = [idx[0] for idx in idxs]

    # For each point, find the closest centroid using shortest path in the graph
    # (i.e approximation of the geodesic distance)
    points_closest_centroid_index = np.zeros((len(points), ))
    points_closest_centroid_length = np.ones((len(points), ))*1000
    for index in centroids_nodes_indexes:
        shortest_paths_length = nx.shortest_path_length(graph, source=index)
        for i in range(len(points)):
            if i in shortest_paths_length and shortest_paths_length[i] < points_closest_centroid_length[i]:
                points_closest_centroid_index[i] = index
                points_closest_centroid_length[i] = shortest_paths_length[i]

    # Compute the classic voronoi partitions of the shape using all points and then join the region
    # corresponding to the same centroid
    voronoi_partitions = voronoi_partition_pts(points, shape)

    return [cascaded_union(voronoi_partitions[points_closest_centroid_index == index])
            for index in centroids_nodes_indexes]
