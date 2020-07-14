from typing import List, Tuple, Union, Dict

import pandas as pd
import numpy as np

from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from vresutils.graph import voronoi_partition_pts

from src.data.geographics import get_points_in_shape, display_polygons
from src.data.technologies import get_config_dict

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def plot_grid_cells(grid_cells_ds: pd.Series, show=False):
    """Plot grid cells (points and regions)"""

    land_50m = cf.NaturalEarthFeature('physical', 'land', '50m',
                                      edgecolor='darkgrey',
                                      facecolor=cf.COLORS['land_alt1'])

    axes = []
    for tech in set(grid_cells_ds.index.get_level_values(0)):
        tech_grid_cells_ds = grid_cells_ds.loc[tech]
        ax = display_polygons(tech_grid_cells_ds.values, fill=False, show=False)
        points = list(tech_grid_cells_ds.index)
        xs, ys = zip(*points)
        ax.add_feature(land_50m, linewidth=0.5)
        ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5)
        ax.scatter(xs, ys, transform=ccrs.PlateCarree(), c='k', zorder=10)
        axes += [ax]

    if show:
        plt.show()
    return axes


def create_grid_cells(shape: Union[Polygon, MultiPolygon], resolution: float) \
        -> (List[Tuple[float, float]], List[Union[Polygon, MultiPolygon]]):
    """Divide a geographical shape by applying voronoi partition."""

    points = get_points_in_shape(shape, resolution)
    if len(points) == 0:
        logger.warning("WARNING: No points at given resolution falls into shape.")
        return points, []
    grid_cells = voronoi_partition_pts(points, shape)

    # Keep only Polygons and MultiPolygons
    for i, shape in enumerate(grid_cells):
        if isinstance(shape, GeometryCollection):
            geos = [geo for geo in shape if isinstance(geo, Polygon) or isinstance(geo, MultiPolygon)]
            grid_cells[i] = unary_union(geos)

    return points, grid_cells


def get_grid_cells(technologies: List[str], resolution: float,
                   onshore_shape: Union[Polygon, MultiPolygon] = None,
                   offshore_shape: Union[Polygon, MultiPolygon] = None) -> pd.Series:
    """
    Divide shapes in grid cell for a list of technologies.

    Parameters
    ----------
    technologies: List[str]
        List of technologies for which we want to generate grid cells.
    resolution: float
        Spatial resolution at which the grid cells must be defined.
    onshore_shape: Union[Polygon, MultiPolygon] (default: None)
        Onshore geographical scope.
    offshore_shape: Union[Polygon, MultiPolygon] (default: None)
        Offshore geographical scope.

    Returns
    -------
    pd.Series
        Series indicating for each technology and each grid cell defined for this technology the associated
        grid cell shape.

    """

    assert len(technologies) != 0, 'Error: Empty list of technologies.'

    # Determine if tech are onshore- or offshore-based
    tech_config = get_config_dict(technologies, ["onshore"])

    # Check the right shapes have been passed
    for tech in technologies:
        is_onshore = tech_config[tech]["onshore"]
        shape = onshore_shape if is_onshore else offshore_shape
        assert shape is not None, f"Error: Missing {'onshore' if is_onshore else 'offshore'} " \
                                  f"shape for technology {tech}"

    # Divide onshore and offshore shapes at a given resolution
    onshore_points, onshore_grid_cells_shapes = None, None
    # TODO: why is taking ages on onshore shape of EU compared to offshore?
    if onshore_shape is not None:
        onshore_points, onshore_grid_cells_shapes = create_grid_cells(onshore_shape, resolution)
    offshore_points, offshore_grid_cells_shapes = None, None
    if offshore_shape is not None:
        offshore_points, offshore_grid_cells_shapes = create_grid_cells(offshore_shape, resolution)

    # Collect onshore and offshore grid cells for each technology
    tech_point_tuples = []
    grid_cells_shapes = np.array([])
    for i, tech in enumerate(technologies):
        is_onshore = tech_config[tech]["onshore"]
        points = onshore_points if is_onshore else offshore_points
        tech_grid_cell_shapes = onshore_grid_cells_shapes if is_onshore else offshore_grid_cells_shapes
        grid_cells_shapes = np.append(grid_cells_shapes, tech_grid_cell_shapes)
        tech_point_tuples += [(tech, point) for point in points]

    grid_cells = pd.Series(grid_cells_shapes, index=pd.MultiIndex.from_tuples(tech_point_tuples))
    return grid_cells.sort_index()
