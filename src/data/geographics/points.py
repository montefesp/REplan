from typing import List, Tuple, Union

import pandas as pd
import numpy as np

from itertools import product

import shapely
import shapely.prepared
from shapely.ops import unary_union
from shapely.errors import TopologicalError
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, GeometryCollection

from src.data.geographics.shapes import get_shapes

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: need to check if it is not removing to much points
def match_points_to_regions(points: List[Tuple[float, float]], shapes_ds: pd.Series,
                            keep_outside: bool = True) -> pd.Series:
    """
    Match a set of points to regions by identifying in which region each point falls.

    If keep_outside is True, points that don't fall in any shape but which are close enough to one shape are kept,
    otherwise those points are dumped.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points to assign to regions
    shapes_ds : pd.Series
        Dataframe storing geometries of NUTS regions.
    keep_outside: bool (default: True)
        Whether to keep points that fall outside of shapes

    Returns
    -------
    points_region_ds : pd.Series
        Series giving for each point the associated region or NA
    """

    points_region_ds = pd.Series(index=pd.MultiIndex.from_tuples(points)).sort_index()
    points = MultiPoint(points)

    for index, subregion in shapes_ds.items():

        try:
            points_in_region = points.intersection(subregion)
        except shapely.errors.TopologicalError:
            print(f"Warning: Problem with shape {index}")
            continue
        if points_in_region.is_empty:
            continue
        elif isinstance(points_in_region, Point):
            points = points.difference(points_in_region)
            points_in_region = (points_in_region.x, points_in_region.y)
        elif len(points_in_region) == 0:
            continue
        else:
            points = points.difference(points_in_region)
            points_in_region = [(point.x, point.y) for point in points_in_region]
        points_region_ds.loc[points_in_region] = index

        if points.is_empty:
            return points_region_ds

    logger.debug(f"Warning: Some points ({points}) are not contained in any shape.")

    if not keep_outside:
        return points_region_ds

    min_distance = 1.
    logger.debug(f"These points will be assigned to closest one if distance is less than {min_distance}.")
    if isinstance(points, Point):
        points = [points]

    not_added_points = []
    for point in points:
        distances = [point.distance(shapes_ds.loc[subregion]) for subregion in shapes_ds.index]
        if min(distances) < min_distance:
            closest_index = np.argmin(distances)
            points_region_ds.loc[(point.x, point.y)] = shapes_ds.index.values[closest_index]
        else:
            not_added_points += [point]
    if len(not_added_points) != 0:
        logger.info(f"Warning: These points were not assigned to any shape: "
                    f"{[(point.x, point.y) for point in not_added_points]}.")

    return points_region_ds


def match_points_to_countries(points: List[Tuple[float, float]], countries: List[str]) -> pd.Series:
    """
    Return in which country's region (onshore + offshore) each points falls into.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points defined as tuples (longitude, latitude)
    countries: List[str]
        List of ISO codes of countries

    Returns
    -------
    pd.Series
        Series giving for each point the associated region or NA if the point didn't fall into any region
    """

    shapes = get_shapes(countries, which='onshore_offshore', save_file_str='countries')

    union_shapes = pd.Series(index=countries)
    for country in countries:
        union_shapes[country] = unary_union(shapes.loc[country, 'geometry'])

    # Assign points to each country based on region
    return match_points_to_regions(points, union_shapes)


def get_points_in_shape(shape: Union[Polygon, MultiPolygon], resolution: float,
                        points: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """
    Return list of coordinates (lon, lat) located inside a geographical shape at a certain resolution.

    Parameters
    ----------
    shape: Polygon or MultiPolygon
        Geographical shape made of points (lon, lat)
    resolution: float
        Longitudinal and latitudinal spatial resolution of points
    points: List[(float, float)]
        Points from which to start

    Returns
    -------
    points: List[(float, float)]

    """
    # Generate all points at the given resolution inside a rectangle whose bounds are the
    #  based on the outermost points of the shape.
    if points is None:
        minx, miny, maxx, maxy = shape.bounds
        minx = round(minx / resolution) * resolution
        maxx = round(maxx / resolution) * resolution
        miny = round(miny / resolution) * resolution
        maxy = round(maxy / resolution) * resolution
        xs = np.linspace(minx, maxx, num=int((maxx - minx) / resolution) + 1)
        ys = np.linspace(miny, maxy, num=int((maxy - miny) / resolution) + 1)
        points = list(product(xs, ys))
    points = MultiPoint(points)
    points = [(point.x, point.y) for point in points.intersection(shape)]

    return points


def divide_shape_with_voronoi(shape: Union[Polygon, MultiPolygon], resolution: float) \
        -> (List[Tuple[float, float]], List[Union[Polygon, MultiPolygon]]):
    """Divide a geographical shape by applying voronoi partition."""

    from vresutils.graph import voronoi_partition_pts

    points = get_points_in_shape(shape, resolution)
    grid_cells = voronoi_partition_pts(points, shape)

    # Keep only Polygons and MultiPolygons
    for i, shape in enumerate(grid_cells):
        if isinstance(shape, GeometryCollection):
            geos = [geo for geo in shape if isinstance(geo, Polygon) or isinstance(geo, MultiPolygon)]
            grid_cells[i] = unary_union(geos)

    return points, grid_cells
