from typing import Tuple

import geopy.distance
import pandas as pd

from pyggrid.data import data_path


def get_nuts_area() -> pd.DataFrame:
    """Return for each NUTS region (2013 and 2016 version) its size in km2"""

    area_fn = f"{data_path}geographics/source/eurostat/reg_area3.xls"
    return pd.read_excel(area_fn, header=9, index_col=0)[:2193]


def get_area_per_site(point: Tuple[float, float], spatial_resolution: float) -> float:
    """Compute cell area for potential siting locations based on reanalysis grids with different resolutions.

     Parameters
     ----------
     point: Tuple[float, float]
         Point coordinates in (lon, lat) format.
     resolution: float
         Spatial resolution of data.

     Returns
     -------
     site_area: float
         Area of grid cell.
     """

    point_limit_south = (point[0], point[1] - spatial_resolution / 2.)
    point_limit_north = (point[0], point[1] + spatial_resolution / 2.)
    point_limit_west = (point[0] - spatial_resolution / 2., point[1])
    point_limit_east = (point[0] + spatial_resolution / 2., point[1])

    dist_latitude = geopy.distance.distance(geopy.distance.lonlat(*point_limit_south),
                                            geopy.distance.lonlat(*point_limit_north)).km
    dist_longitude = geopy.distance.distance(geopy.distance.lonlat(*point_limit_west),
                                             geopy.distance.lonlat(*point_limit_east)).km
    site_area = dist_latitude * dist_longitude

    return site_area
