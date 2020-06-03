from typing import List, Union, Any

from random import random
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd

from shapely.geometry import MultiPolygon, Polygon


def display_polygons(polygons_list: List[Union[Polygon, MultiPolygon]], fill=True, show=True) -> Any:
    """
    Display in different colours a set of polygons or multipolygons.

    Parameters
    ----------
    polygons_list: List[Union[Polygon, MultiPolygon]]
        List of shapely polygons or multipolygons
    fill: bool (default: True)
        Whether to fill in shapes or not
    show: bool (default: True)
        Whether to use plt.show or to return plotting axis
    """

    assert isinstance(polygons_list, list) or isinstance(polygons_list, np.ndarray) \
        or isinstance(polygons_list, gpd.array.GeometryArray), \
        f'The argument must be a list of polygons or multipolygons, got {type(polygons_list)}'

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    for polygons in polygons_list:
        c = (random(), random(), random())
        if isinstance(polygons, Polygon):
            polygons = [polygons]
        for poly in polygons:
            longitudes = [i[0] for i in poly.exterior.coords]
            latitudes = [i[1] for i in poly.exterior.coords]
            if not fill:
                ax.plot(longitudes, latitudes, transform=ccrs.PlateCarree(), color='k')
            else:
                ax.fill(longitudes, latitudes, transform=ccrs.PlateCarree(), color=c)
                # Remove interior
                interior_polys = list(poly.interiors)
                for i_poly in interior_polys:
                    longitudes = [i[0] for i in i_poly.coords]
                    latitudes = [i[1] for i in i_poly.coords]
                    ax.fill(longitudes, latitudes, transform=ccrs.PlateCarree(), color='white')

    if show:
        plt.show()
    else:
        return ax
