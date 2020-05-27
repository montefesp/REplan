from os.path import join, dirname, abspath
from typing import List, Dict, Union
from ast import literal_eval

import numpy as np
import pandas as pd

from osgeo import ogr, osr
import geokit as gk
import glaes as gl

import multiprocessing as mp
import progressbar as pgb

import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon

from src.data.geographics import get_shapes

def init_land_availability_globals(filters: Dict) -> None:
    """
    Initialize global variables to use in land availability computation

    Parameters
    ----------
    filters: Dict
        Dictionary containing a set of values describing the filters to apply to obtain land availability.

    """

    # global in each process of the multiprocessing.Pool
    global gebco_, clc_, natura_, spatial_ref_, filters_, onshore_shape_

    filters_ = filters

    spatial_ref_ = osr.SpatialReference()
    spatial_ref_.ImportFromEPSG(4326)

    land_data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/")

    # Natura dataset (protected areas in Europe)
    if "natura" in filters:
        natura_ = gk.raster.loadRaster(f"{land_data_dir}generated/natura2000.tif")

    # GEBCO dataset (altitude and depth)
    if 'depth_thresholds' in filters or 'altitude_threshold' in filters:
        gebco_fn = f"{land_data_dir}source/GEBCO/GEBCO_2019/gebco_2019_n75.0_s30.0_w-20.0_e40.0.tif"
        gebco_ = gk.raster.loadRaster(gebco_fn)

    # Corine dataset (land use)
    if 'clc' in filters:
        clc_ = gk.raster.loadRaster(f"{land_data_dir}source/CLC2018/CLC2018_CLC2018_V2018_20.tif")
        clc_.SetProjection(gk.srs.loadSRS(3035).ExportToWkt())

    #if 'distances_to_shore' in filters:
    #    onshore_shape = unary_union(get_shapes(get_subregions("EU"), 'onshore', save=True)["geometry"].values)
    #    onshore_shape_wkt = shapely.wkt.dumps(onshore_shape)
    #    onshore_shape_ = ogr.CreateGeometryFromWkt(onshore_shape_wkt, spatial_ref_)


def compute_land_availability(shape: Union[Polygon, MultiPolygon]) -> float:
    """
    Compute land availability in (km2) of a geographical shape.

    Parameters
    ----------
    shape: Union[Polygon, MultiPolygon]
        Geographical shape
    Returns
    -------
    float
        Available area in (km2)
    Notes
    -----
    spatial_ref, filters, natura, clc and gebco must have been previously initialized as global variables

    """

    import matplotlib.pyplot as plt

    poly_wkt = shapely.wkt.dumps(shape)
    poly = ogr.CreateGeometryFromWkt(poly_wkt, spatial_ref_)

    # Compute rooftop area using COPERNICUS
    if filters_.get("copernicus", 0):
        path_cop = join(dirname(abspath(__file__)),
                        f"../../../data/land_data/source/COPERNICUS/ESM_class50_100m/ESM_class50_100m.tif")
        ec = gl.ExclusionCalculator(poly, pixelRes=1000, initialValue=path_cop)

        return ec.areaAvailable/1e6

    ec = gl.ExclusionCalculator(poly, pixelRes=1000)

    # GLAES priors
    if 'glaes_priors' in filters_:
        priors_filters = filters_["glaes_priors"]
        for prior_name in priors_filters.keys():
            prior_value = priors_filters[prior_name]
            if isinstance(prior_value, str):
                prior_value = literal_eval(prior_value)
            ec.excludePrior(prior_name, value=prior_value)

    # Exclude protected areas
    if 'natura' in filters_:
        ec.excludeRasterType(natura_, value=filters_["natura"])

    # Depth and altitude filters
    if 'depth_thresholds' in filters_:
        depth_thresholds = filters_["depth_thresholds"]
        # Keep points between two depth thresholds
        ec.excludeRasterType(gebco_, (depth_thresholds["low"], depth_thresholds["high"]), invert=True)
    elif 'altitude_threshold' in filters_ and filters_['altitude_threshold'] > 0.:
        ec.excludeRasterType(gebco_, (filters_['altitude_threshold'], None))

    # Corine filters
    if 'clc' in filters_:
        clc_filters = filters_["clc"]

        if "remove_codes" in clc_filters and "remove_distance" in clc_filters and clc_filters["remove_distance"] > 0.:
            # for remove_code in clc_filters["remove_codes"]:
            #    ec.excludeRasterType(clc_, value=remove_code, buffer=clc_filters["remove_distance"])
            ec.excludeRasterType(clc_, value=clc_filters["remove_codes"], buffer=clc_filters["remove_distance"])

        if "keep_codes" in clc_filters:
            # Invert True indicates code that need to be kept
            # TODO: doing a loop or not doesn't give the same result...
            # for keep_code in clc_filters["keep_codes"]:
            #    ec.excludeRasterType(clc_, value=keep_code, mode='include')
            ec.excludeRasterType(clc_, value=clc_filters["keep_codes"], mode='include')

    #if 'distances_to_shore' in filters_:
    #    distance_filters = filters_['distances_to_shore']
    #    if 'min' in distance_filters:
    #        ec.excludeVectorType(onshore_shape_, buffer=distance_filters['min'])

    # ec.draw()
    # plt.show()

    return ec.areaAvailable/1e6


def get_land_availability_for_shapes_mp(shapes: List[Union[Polygon, MultiPolygon]],
                                        filters: Dict, processes: int = None) -> np.array:
    """Return land availability in a list of geographical region for a given technology using multiprocessing"""

    with mp.Pool(initializer=init_land_availability_globals, initargs=(filters, ),
                 maxtasksperchild=20, processes=processes) as pool:

        widgets = [
            pgb.widgets.Percentage(),
            ' ', pgb.widgets.SimpleProgress(format='(%s)' % pgb.widgets.SimpleProgress.DEFAULT_FORMAT),
            ' ', pgb.widgets.Bar(),
            ' ', pgb.widgets.Timer(),
            ' ', pgb.widgets.ETA()
        ]
        progressbar = pgb.ProgressBar(prefix='Compute GIS potentials: ', widgets=widgets, max_value=len(shapes))
        available_areas = list(progressbar(pool.imap(compute_land_availability, shapes)))

    return np.array(available_areas)


def get_land_availability_for_shapes_non_mp(shapes: List[Union[Polygon, MultiPolygon]], filters: Dict) -> np.array:
    """Return land availability in a list of geographical region for a given technology NOT using multiprocessing"""

    init_land_availability_globals(filters)
    available_areas = np.zeros((len(shapes), ))
    for i, shape in enumerate(shapes):
        available_areas[i] = compute_land_availability(shape)
    return available_areas


def get_land_availability_for_shapes(shapes: List[Union[Polygon, MultiPolygon]], filters: Dict,
                                     processes: int = None) -> np.array:
    """
    Return land availability in a list of geographical region for a given technology.

    Parameters
    ----------
    shapes: List[Union[Polygon, MultiPolygon]]
        List of geographical regions
    filters: Dict
        Dictionary containing a set of values describing the filters to apply to obtain land availability.
    processes: int
        Number of parallel processes

    Returns
    -------
    np.array
        Land availability (in km) for each shape

    """

    assert len(shapes) != 0, "Error: List of shapes is empty."

    if processes == 1:
        return get_land_availability_for_shapes_non_mp(shapes, filters)
    else:
        return get_land_availability_for_shapes_mp(shapes, filters, processes)


def get_capacity_potential_for_shapes(shapes: List[Union[Polygon, MultiPolygon]], filters: Dict,
                                      power_density: float, processes: int = None) -> np.array:
    """
    Return capacity potentials (GW) in a series of geographical shapes.

    Parameters
    ----------
    shapes: List[Union[Polygon, MultiPolygon]]
        List of geographical regions
    filters: Dict
        Dictionary containing a set of values describing the filters to apply to obtain land availability.
    power_density: float
        Power density in MW/km2
    processes: int (default: None)
        Number of parallel processes

    Returns
    -------
    np.array
        Array of capacity potentials (GW)
    """
    return get_land_availability_for_shapes(shapes, filters, processes) * power_density / 1e3


def get_capacity_potential_per_country(countries: List[str], is_onshore: float, filters: Dict,
                                       power_density: float, processes: int = None):
    """
    Return capacity potentials (GW) in a series of countries.

    Parameters
    ----------
    countries: List[str]
        List of ISO codes.
    is_onshore: bool
        Whether the technology is onshore located.
    filters: Dict
        Dictionary containing a set of values describing the filters to apply to obtain land availability.
    power_density: float
        Power density in MW/km2
    processes: int (default: None)
        Number of parallel processes

    Returns
    -------
    pd.Series
        Series containing the capacity potentials (GW) for each code.

    """
    which = 'onshore' if is_onshore else 'offshore'
    shapes = get_shapes(countries, which=which, save=True)["geometry"]
    land_availability = get_land_availability_for_shapes(shapes, filters, processes)

    return pd.Series(land_availability*power_density/1e3, index=countries)
