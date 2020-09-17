from os.path import join, dirname, abspath, isfile
from os import listdir
from typing import List, Dict, Union, Any
from ast import literal_eval
import yaml

import numpy as np
import pandas as pd

from osgeo import ogr, osr
import geokit as gk
import glaes as gl

import multiprocessing as mp
import progressbar as pgb

import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon

from pyggrid.data.geographics import get_shapes


def get_glaes_prior_defaults(config: List[str], priors: List[str] = None) -> Dict[str, Any]:
    """
    Returns defaults thresholds values for a list of glaes priors.

    Parameters
    ----------
    config: List[str]
        List of strings determining the default configuration
    priors: List[str] (default: None)
        List of priors for which we want to have default thresholds.
        If None, return all the priors thresholds in the chosen configuration.

    Returns
    -------
    Dict[str, Any]

    """

    # Get the main configuration file
    assert len(config) != 0
    exclusions_fn = join(dirname(abspath(__file__)), f"../../../../../../data/generation/vres/potentials/source/"
                                                     f"GLAES/exclusions/{config[0]}.yml")
    assert isfile(exclusions_fn), f"Error: No exclusion configuration named {config[0]}. File {exclusions_fn} not found"
    prior_threshold_dict = yaml.load(open(exclusions_fn, 'r'), Loader=yaml.FullLoader)

    # Get the right sub-configuration in this file
    for sub_config in config[1:]:
        assert sub_config in prior_threshold_dict, f"Error: Did not find sub-configuration " \
                                                   f"{sub_config} of configuration {config[0]}"
        prior_threshold_dict = prior_threshold_dict[sub_config]

    # Filter on chosen priors
    if priors is None:
        return prior_threshold_dict
    prior_threshold_dict_final = {}
    for prior in priors:
        assert prior in prior_threshold_dict, f"Error: Default thresholds for prior {prior} is " \
                                              f"not available in configuration {config}"
        prior_threshold_dict_final[prior] = prior_threshold_dict[prior]

    return prior_threshold_dict_final


def init_land_availability_globals(filters: Dict) -> None:
    """
    Initialize global variables to use in land availability computation

    Parameters
    ----------
    filters: Dict
        Dictionary containing a set of values describing the filters to apply to obtain land availability.

    """

    # global in each process of the multiprocessing.Pool
    global gebcos_, clc_, natura_, spatial_ref_, filters_, cargo_, tanker_, cables_, pipelines_

    filters_ = filters

    spatial_ref_ = osr.SpatialReference()
    spatial_ref_.ImportFromEPSG(4326)

    data_dir = join(dirname(abspath(__file__)), "../../../../../../data/generation/vres/potentials/")

    # Natura dataset (protected areas in Europe)
    if "natura" in filters:
        natura_ = gk.raster.loadRaster(f"{data_dir}generated/GLAES/natura2000.tif")

    # GEBCO dataset (altitude and depth)
    if 'depth_thresholds' in filters or 'altitude_threshold' in filters:
        gebco_dir = f"{data_dir}source/GEBCO/"
        gebco_fns = [f"{gebco_dir}{fn}" for fn in listdir(gebco_dir) if fn.endswith(".tif")]
        gebcos_ = [gk.raster.loadRaster(fn) for fn in gebco_fns]

    # Corine dataset (land use)
    if 'clc' in filters:
        clc_ = gk.raster.loadRaster(f"{data_dir}source/CLC2018/CLC2018_CLC2018_V2018_20.tif")
        clc_.SetProjection(gk.srs.loadSRS(3035).ExportToWkt())

    if 'shipping' in filters:
        cargo_fn = f"{data_dir}source/EMODnet/HA_Routes_Density_2019/" \
                      f"wid6-cargo-all_europe-yearly-20190101000000_20191231235959-tdm-grid.tif"
        cargo_ = gk.raster.loadRaster(cargo_fn)
        tanker_fn = f"{data_dir}source/EMODnet/HA_Routes_Density_2019/" \
                    f"wid6-tanker-all_europe-yearly-20190101000000_20191231235959-tdm-grid.tif"
        tanker_ = gk.raster.loadRaster(tanker_fn)

    if 'cables' in filters:
        cables_fn = f"{data_dir}source/EMODnet/HA_TC_Cables_Schematic_20170801/Cables_schematic_20170801.shp"
        cables_ = gk.vector.loadVector(cables_fn)

    if 'pipelines' in filters:
        pipelines_fn = f"{data_dir}source/EMODnet/HA_Pipelines_20191220/EMODnet_HA_Pipelines_20191220.shp"
        pipelines_ = gk.vector.loadVector(pipelines_fn)


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

    # import matplotlib.pyplot as plt

    poly_wkt = shapely.wkt.dumps(shape)
    poly = ogr.CreateGeometryFromWkt(poly_wkt, spatial_ref_)

    # Compute rooftop area using ESM (European Settlement Map)
    if filters_.get("esm", 0):
        path_cop = join(dirname(abspath(__file__)), f"../../../../../../data/generation/vres/potentials/"
                                                    f"source/ESM/ESM_class50_100m/ESM_class50_100m.tif")
        ec = gl.ExclusionCalculator(poly, pixelRes=1000, initialValue=path_cop)

        return ec.areaAvailable/1e6

    ec = gl.ExclusionCalculator(poly, pixelRes=1000)
    ec.draw()

    # GLAES priors
    if 'glaes_prior_defaults' in filters_:
        prior_defaults_params = filters_['glaes_prior_defaults']
        prior_config = prior_defaults_params["config"]
        priors = prior_defaults_params["priors"] if 'priors' in prior_defaults_params else None
        filters_["glaes_priors"] = get_glaes_prior_defaults(prior_config, priors)

    if 'glaes_priors' in filters_:
        priors_filters = filters_["glaes_priors"]
        for prior_name in priors_filters.keys():
            prior_value = priors_filters[prior_name]
            if isinstance(prior_value, str):
                prior_value = literal_eval(prior_value)
            # Can receive a tuple or a list of tuples
            if not isinstance(prior_value, list):
                prior_value = [prior_value]
            for value in prior_value:
                ec.excludePrior(prior_name, value=value)

    # Exclude protected areas
    if 'natura' in filters_:
        ec.excludeRasterType(natura_, value=filters_["natura"])

    # Depth and altitude filters
    if 'depth_thresholds' in filters_:
        depth_thresholds = filters_["depth_thresholds"]
        # Keep points between two depth thresholds
        for gebco in gebcos_:
            ec.excludeRasterType(gebco, (-1e4, depth_thresholds["low"]))
            ec.excludeRasterType(gebco, (depth_thresholds["high"], 0))
    elif 'altitude_threshold' in filters_ and filters_['altitude_threshold'] > 0.:
        for gebco in gebcos_:
            ec.excludeRasterType(gebco, (filters_['altitude_threshold'], 1e6))

    # Corine filters
    if 'clc' in filters_:
        clc_filters = filters_["clc"]

        if "remove_codes" in clc_filters and "remove_distance" in clc_filters and clc_filters["remove_distance"] > 0.:
            # for remove_code in clc_filters["remove_codes"]:
            #    ec.excludeRasterType(clc_, value=remove_code, buffer=clc_filters["remove_distance"])
            ec.excludeRasterType(clc_, value=clc_filters["remove_codes"], buffer=clc_filters["remove_distance"])

        if "keep_codes" in clc_filters:
            # Invert True indicates code that need to be kept
            # for keep_code in clc_filters["keep_codes"]:
            #    ec.excludeRasterType(clc_, value=keep_code, mode='include')
            ec.excludeRasterType(clc_, value=clc_filters["keep_codes"], mode='include')

    if 'shipping' in filters_:
        value = filters_['shipping']
        value = literal_eval(value) if isinstance(value, str) else value
        ec.excludeRasterType(cargo_, value=value)
        ec.excludeRasterType(tanker_, value=value)

    if 'cables' in filters_:
        ec.excludeVectorType(cables_, buffer=filters_['cables'])

    if 'pipelines' in filters_:
        ec.excludeVectorType(pipelines_, buffer=filters_['pipelines'])

    # ec.draw()
    # plt.show()
    # plt.savefig("test.png")

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
        Land availability (in km2) for each shape

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
    available_area = get_land_availability_for_shapes(shapes, filters, processes)
    print(available_area)
    return available_area * power_density / 1e3


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

    return pd.Series(land_availability*power_density/1e3, index=shapes.index)


if __name__ == '__main__':
    from pyggrid.data.geographics import get_shapes
    from pyggrid.data.technologies import get_config_values
    filters_ = get_config_values("wind_onshore_national", ["filters"])
    print(filters_)
    # filters_ = {"depth_thresholds": {"high": -1., "low": -999.}}
    full_gl_shape = get_shapes(["FI"], "onshore")["geometry"][0]
    trunc_gl_shape = full_gl_shape.intersection(Polygon([(0., 50.), (0., 66.5), (40., 66.5), (40., 50.)]))
    from pyggrid.data.geographics.plot import display_polygons
    display_polygons([trunc_gl_shape])
    print(get_capacity_potential_for_shapes([trunc_gl_shape], filters_, 5), 1)
