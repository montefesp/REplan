from typing import List, Union

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union

from sklearn.neighbors import NearestNeighbors
from vresutils.graph import voronoi_partition_pts
import networkx as nx
import numpy as np

from src.data.geographics import get_points_in_shape


def voronoi_special(shape: Union[Polygon, MultiPolygon], centroids: List[List[float]], resolution: float = 0.5):
    """
    Apply a special Voronoi partition of a non-convex polygon based on an approximation of the
    geodesic distance to a set of points which define the centroids of each partition.

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
    points = get_points_in_shape(shape, resolution)

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
