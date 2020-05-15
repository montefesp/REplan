from os.path import join, dirname, abspath
from typing import List, Tuple, Dict, Any, Union
from copy import copy

import xarray as xr
import numpy as np
import pandas as pd
import xarray.ufuncs as xu
import dask.array as da
import geopandas as gpd
import scipy.spatial
import geopy.distance

# TODO: will probably need to use original versions and not PyPSAs
from osgeo import ogr, osr
import geokit as gk
import glaes as gl

import multiprocessing as mp
import progressbar as pgb

import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon

from src.data.vres_profiles import read_resource_database
from src.data.geographics import create_grid_cells


def filter_onshore_offshore_points(onshore: bool, points: List[Tuple[float, float]],
                                   spatial_resolution: float) -> List[Tuple[float, float]]:
    """
    Filter coordinates to leave only onshore and offshore coordinates depending on technology.

    Parameters
    ----------
    onshore: bool
        If True, keep only points that are offshore, else keep points offshore
    points : List[Tuple[float, float]]
        List of points to filter
    spatial_resolution : float
        Spatial resolution of coordinates

    Returns
    -------
    List[Tuple[float, float]]
        Points filtered via land/water mask.
    """

    path_land_data = join(dirname(abspath(__file__)),
                          f"../../../data/land_data/source/ERA5/ERA5_land_sea_mask_20181231_{spatial_resolution}.nc")
    dataset = xr.open_dataset(path_land_data)
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    array_watermask = dataset['lsm']

    if onshore:
        mask_watermask = array_watermask.where(array_watermask.data >= 0.3)
    else:
        mask_watermask = array_watermask.where(array_watermask.data < 0.3)

    points_in_mask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

    return list(set(points).intersection(set(points_in_mask)))


def read_filter_database(filename: str, coords: List[Tuple[float, float]] = None) -> xr.Dataset:
    """
    Open a file containing filtering information.

    Parameters
    ----------
    filename: str
        Name of the file containing the filtering information
    coords: List[Tuple[float, float]] (default: None)
        List of points for which we want the filtering information

    Returns
    -------
    dataset: xarray.Dataset
    """

    dataset = xr.open_dataset(filename)
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])

    # Changing longitude from 0-360 to -180-180
    dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    if coords is not None:
        dataset = dataset.sel(locations=coords)

    return dataset


def filter_points_by_layer(filter_name: str, points: List[Tuple[float, float]], spatial_resolution: float,
                           tech_dict: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Compute locations to remove from the initial set following various
    land-, vres_profiles-, population-based criteria.

    Parameters
    ----------
    filter_name: str
        Name of the filter to be applied
    points : List[Tuple[float, float]]
        List of points.
    spatial_resolution : float
        Spatial resolution of the points.
    tech_dict : Dict[str, Any]
        Dict object containing technical parameters and constraints of a given technology.

    Returns
    -------
    points : List[Tuple[float, float]]
        List of filtered points.

    """
    if filter_name == 'protected_areas':

        protected_areas_selection = tech_dict['protected_areas_selection']
        threshold_distance = tech_dict['protected_areas_distance_threshold']

        path_land_data = join(dirname(abspath(__file__)),
                              '../../../data/land_data/source/WDPA/WDPA_Apr2020-shapefile-points.shp')
        dataset = gpd.read_file(path_land_data)

        # Retrieve the geopandas Point objects and their coordinates
        dataset = dataset[dataset['IUCN_CAT'].isin(protected_areas_selection)]
        protected_points = dataset.geometry.apply(lambda p: (round(p[0].x, 2), round(p[0].y, 2))).values

        # Compute closest protected point for each coordinate
        protected_points = np.array([[p[0], p[1]] for p in protected_points])
        points = np.array([[p[0], p[1]] for p in points])
        closest_points = \
            protected_points[np.argmin(scipy.spatial.distance.cdist(protected_points, points, 'euclidean'), axis=0)]

        # Remove coordinates that are too close to protected areas
        points_to_remove = []
        for coord1, coord2 in zip(points, closest_points):
            if geopy.distance.geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).km < threshold_distance:
                points_to_remove.append(tuple(coord1))

        points = list(set(points) - set(points_to_remove))

    elif filter_name == 'resource_quality':

        #TODO: kinda slow (again a problem of xarray in-memory computations, as for hydro runoff)
        database = read_resource_database(spatial_resolution)
        database = database.sel(locations=sorted(points))

        if tech_dict['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 + database.v100 ** 2)
        elif tech_dict['resource'] == 'pv':
            array_resource = database.ssrd / 3600.
        else:
            raise ValueError("Error: Resource must be wind or pv")

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < tech_dict['resource_threshold'], 0)
        coords_mask_resource = mask_resource[da.nonzero(mask_resource)].locations.values.tolist()
        points = list(set(points).difference(set(coords_mask_resource)))

    elif filter_name == 'orography':

        dataset_name = join(dirname(abspath(__file__)),
                            f"../../../data/land_data/source/ERA5/"
                            f"ERA5_orography_characteristics_20181231_{spatial_resolution}.nc")
        dataset = read_filter_database(dataset_name, points)

        altitude_threshold = tech_dict['altitude_threshold']
        slope_threshold = tech_dict['terrain_slope_threshold']

        array_altitude = dataset['z'] / 9.80665
        array_slope = dataset['slor']

        mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
        points_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()

        mask_slope = array_slope.where(array_slope.data > slope_threshold)
        points_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

        points_mask_orography = set(points_mask_altitude).union(set(points_mask_slope))
        points = list(set(points).difference(points_mask_orography))

    elif filter_name == 'forestry':

        dataset_name = join(dirname(abspath(__file__)),
                            f"../../../data/land_data/source/ERA5/"
                            f"ERA5_surface_characteristics_20181231_{spatial_resolution}.nc")
        dataset = read_filter_database(dataset_name, points)

        forestry_threshold = tech_dict['forestry_ratio_threshold']

        array_forestry = dataset['cvh']

        mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
        points_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_forestry)))

    elif filter_name == 'water_mask':

        dataset_name = join(dirname(abspath(__file__)),
                            f"../../../data/land_data/source/ERA5/ERA5_land_sea_mask_20181231_{spatial_resolution}.nc")
        dataset = read_filter_database(dataset_name, points)

        array_watermask = dataset['lsm']

        mask_watermask = array_watermask.where(array_watermask.data < 0.9)
        points_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_watermask)))

    elif filter_name == 'bathymetry':

        dataset_name = join(dirname(abspath(__file__)),
                            f"../../../data/land_data/source/ERA5/ERA5_land_sea_mask_20181231_{spatial_resolution}.nc")
        dataset = read_filter_database(dataset_name, points)

        depth_threshold_low = tech_dict['depth_thresholds']['low']
        depth_threshold_high = tech_dict['depth_thresholds']['high']

        array_watermask = dataset['lsm']
        # Careful with this one because max depth is 999.
        array_bathymetry = dataset['wmb'].fillna(0.)

        # TODO: can we not remove some of the parentheses?
        mask_offshore = array_bathymetry.where(((array_bathymetry.data < depth_threshold_low) |
                                                (array_bathymetry.data > depth_threshold_high)) |
                                               (array_watermask.data > 0.1))
        points_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_offshore)))

    # TODO: check how we organize this file within the structure
    elif filter_name == 'population_density':

        # TODO: can we not load this with population_density.manager?
        degree_resolution = "30_min" if spatial_resolution == 0.5 else "1_deg"
        path_population_data = \
            join(dirname(abspath(__file__)),
                 f"../../../data/population_density/source/"
                 f"gpw_v4_population_density_adjusted_rev11_{degree_resolution}.nc")
        dataset = xr.open_dataset(path_population_data)

        varname = [item for item in dataset.data_vars][0]
        dataset = dataset.rename({varname: 'data'})
        # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
        data_pop = dataset.sel(raster=5)

        array_pop_density = data_pop['data'].interp(longitude=np.arange(-180, 180, float(spatial_resolution)),
                                                    latitude=np.arange(-89, 91, float(spatial_resolution))[::-1],
                                                    method='nearest').fillna(0.)
        array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude'))

        population_density_threshold_low = tech_dict['population_density_threshold_low']
        population_density_threshold_high = tech_dict['population_density_threshold_high']

        mask_population = array_pop_density.where((array_pop_density.data < population_density_threshold_low) |
                                                  (array_pop_density.data > population_density_threshold_high))
        points_mask_population = mask_population[mask_population.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_population)))

    else:

        raise ValueError(f"Layer {filter_name} is not available.")

    return points


def filter_points(technologies: List[str], tech_config: Dict[str, Any], init_points: List[Tuple[float, float]],
                  spatial_resolution: float, filtering_layers: Dict[str, bool]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Filter the set of potential deployment locations for each region and available technology.

    Parameters
    ----------
    init_points : List[Tuple(float, float)]
        List of points to filter
    tech_config: Dict[str, Any]
        Gives for every technology, a set of configuration parameters and their associated values
    spatial_resolution : float
        Spatial resolution at which the points are defined.
    technologies : List[str]
        List of technologies for which we want to filter points.
    filtering_layers: Dict[str, bool]
        Dictionary indicating if a given filtering layers needs to be applied. If the layer name is present as key and
        associated to a True boolean, then the corresponding is applied.

        List of possible filter names:

        resource_quality:
            If taken into account, discard points whose average vres_profiles quality over
            the available time horizon are below a threshold defined in the config_tech.yaml file.
        population_density:
            If taken into account, discard points whose population density is below a
            threshold defined in the config_tech.yaml file for each available technology.
        protected_areas:
            If taken into account, discard points who are closer to protected areas (defined in config_tech.yaml)
            in their vicinity than a distance threshold defined in the config_tech.yaml file.
        orography:
            If taken into account, discard points whose altitude and terrain slope
            are above thresholds defined in the config_tech.yaml file for each individual technology.
        forestry:
            If taken into account, discard points whose forest cover share
            is above a threshold defined in the config_tech.yaml file.
        water_mask:
            If taken into account, discard points whose water coverage share
            is above a threshold defined in the config_tech.yaml file.
        bathymetry (valid for offshore technologies):
            If taken into account, discard points whose water depth is above a threshold defined
            in the config_tech.yaml file for offshore and floating wind, respectively.

    Returns
    -------
    tech_points_dict : Dict[str, List[Tuple(float, float)]]
        Dict object giving for each technology the list of filtered points.

    """
    tech_points_dict = dict.fromkeys(technologies)

    for tech in technologies:

        tech_dict = tech_config[tech]

        points = copy(init_points)
        for key in filtering_layers:

            if len(points) == 0:
                break

            # Apply the filter if it is set to true
            if filtering_layers[key]:

                # Some filter should not apply to some technologies
                if key == 'bathymetry' and tech_dict['deployment'] in ['onshore', 'utility', 'residential']:
                    continue
                if key in ['orography', 'population_density', 'protected_areas', 'forestry', 'water_mask'] \
                        and tech_dict['deployment'] in ['offshore', 'floating']:
                    continue
                points = filter_points_by_layer(key, points, spatial_resolution, tech_dict)

        tech_points_dict[tech] = points

    return tech_points_dict


def init_land_availability_globals(tech_config: Dict):
    """
    Initialize global variables to use in land availability computation

    Parameters
    ----------
    tech_config: Dict
        Dictionary containing a set of values describing the configuration of the technology for which we want
        to obtain land availability.
    """

    # global in each process of the multiprocessing.Pool
    global gebco_, clc_, natura_, spatial_ref_, tech_config_

    tech_config_ = tech_config

    spatial_ref_ = osr.SpatialReference()
    spatial_ref_.ImportFromEPSG(4326)

    land_data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/")

    # Natura dataset (protected areas in Europe)
    natura_ = gk.raster.loadRaster(f"{land_data_dir}generated/natura2000.tif")

    # GEBCO dataset (altitude and depth)
    gebco_fn = f"{land_data_dir}source/GEBCO/GEBCO_2019/gebco_2019_n75.0_s30.0_w-20.0_e40.0.tif"
    gebco_ = gk.raster.loadRaster(gebco_fn)

    # Corine dataset (land use)
    if 'clc' in tech_config:
        clc_ = gk.raster.loadRaster(f"{land_data_dir}source/CLC2018/CLC2018_CLC2018_V2018_20.tif")
        clc_.SetProjection(gk.srs.loadSRS(3035).ExportToWkt())


def compute_land_availability(shape: Union[Polygon, MultiPolygon]):
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
    spatial_ref, tech_config, natura, clc and gebco must have been previously initialized as global variables

    """

    poly_wkt = shapely.wkt.dumps(shape)
    poly = ogr.CreateGeometryFromWkt(poly_wkt, spatial_ref_)
    ec = gl.ExclusionCalculator(poly, pixelRes=1000)

    import matplotlib.pyplot as plt

    # Exclude protected areas
    ec.excludeRasterType(natura_, value=1)

    # Depth and altitude filters
    if 'depth_thresholds' in tech_config_:
        depth_thresholds = tech_config_["depth_thresholds"]
        ec.excludeRasterType(gebco_, (-depth_thresholds["high"], -depth_thresholds["low"]), invert=True)
    elif 'altitude_threshold' in tech_config_ and tech_config_['altitude_threshold'] > 0.:
        ec.excludeRasterType(gebco_, (tech_config_['altitude_threshold'], None))

    # Corine filters
    if 'clc' in tech_config_:
        clc_filters = tech_config_["clc"]

        if "remove_codes" in clc_filters and "remove_distance" in clc_filters and clc_filters["remove_distance"] > 0.:
            for remove_code in clc_filters["remove_codes"]:
                ec.excludeRasterType(clc_, value=remove_code, buffer=clc_filters["remove_distance"])
            #ec.excludeRasterType(clc_, value=clc_filters["remove_codes"], buffer=clc_filters["remove_distance"])

        if "keep_codes" in clc_filters:
            # Invert True indicates code that need to be kept
            for keep_code in clc_filters["keep_codes"]:
                ec.excludeRasterType(clc_, value=keep_code, mode='include')
            #ec.excludeRasterType(clc_, value=clc_filters["keep_codes"], mode='include')

    ec.draw()
    plt.show()

    # GLAES priors
    # TODO: this was just a test, change and parametrize
    if 0:
        ec.excludePrior("agriculture_proximity", value=(None, 0))
        ec.excludePrior("settlement_proximity", value=(None, 1000))
        ec.excludePrior("roads_main_proximity", value=(None, 200))

    # TODO: add distance from the shore filter
    # TODO: filter on vres_profiles quality
    # TODO: add slope?
    # TODO: population density?

    return ec.areaAvailable / 10e5


def get_land_availability_for_shapes_mp(shapes: List[Union[Polygon, MultiPolygon]], tech_config: Dict, processes=None):
    """Return land availability in a list of geographical region for a given technology using multiprocessing"""

    with mp.Pool(initializer=init_land_availability_globals, initargs=(tech_config, ),
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


def get_land_availability_for_shapes_non_mp(shapes: List[Union[Polygon, MultiPolygon]], tech_config: Dict):
    """Return land availability in a list of geographical region for a given technology NOT using multiprocessing"""

    init_land_availability_globals(tech_config)
    available_areas = np.zeros((len(shapes), ))
    for i, shape in enumerate(shapes):
        available_areas[i] = compute_land_availability(shape)
    return available_areas


def get_land_availability_for_shapes(shapes: List[Union[Polygon, MultiPolygon]], tech_config: Dict,
                                     processes: int = None):
    """
    Return land availability in a list of geographical region for a given technology.

    Parameters
    ----------
    shapes: List[Union[Polygon, MultiPolygon]]
        List of geographical regions
    tech_config: Dict
        Dictionary containing a set of values describing the configuration of the technology for which we want
        to obtain land availability.
    processes: int
        Number of parallel processes

    Returns
    -------
    np.array
        Land availability (in km) for each shape

    """
    if processes == 1:
        return get_land_availability_for_shapes_non_mp(shapes, tech_config)
    else:
        return get_land_availability_for_shapes_mp(shapes, tech_config, processes)


# TODO: maybe grid cells shouldn't be created here but send as argument
def get_land_availability_in_grid_cells(technologies: List[str], tech_config: Dict,
                                        shapes: List[Union[Polygon, MultiPolygon]], resolution: float,
                                        processes: int = None) -> pd.DataFrame:
    """
    Compute land availability (in km2) for each grid cell in given shapes for a list of technologies.

    Parameters
    ----------
    technologies: List[str]
        List of technologies for which we want to obtain land availability.
        Each technology must have an entry in 'tech_config'.
    tech_config: Dict
        Dictionary containing a set of values describing the configurations of technologies.
    shapes: List[Union[Polygon, MultiPolygon]]
        Must contain a geographical shape to be divided into grid cells for each value in 'technologies'
    resolution: float
        Spatial resolution at which the grid cells must be defined.
    processes: int
        Number of parallel processes to use

    Returns
    -------
    pd.DataFrame
        DataFrame indicating for each technology and each grid cell defined for this technology the associated
        grid cell shape and land availability (in km2)

    """

    assert len(shapes) == len(technologies), "Error: Number of shapes must be equal to number of technologies"
    for tech in technologies:
        assert tech in tech_config, f"Error: Configuration of technology {tech} was not provided in 'tech_config'"

    tech_point_tuples = []
    available_areas = np.array([])
    grid_cells_shapes = np.array([])
    for i, tech in enumerate(technologies):
        # Compute grid cells
        points, tech_grid_cells_shapes = create_grid_cells(shapes[i], resolution)
        grid_cells_shapes = np.append(grid_cells_shapes, tech_grid_cells_shapes)
        # Compute available land
        available_areas = np.append(available_areas,
                                    get_land_availability_for_shapes(tech_grid_cells_shapes, tech_config[tech],
                                                                     processes))
        tech_point_tuples += [(tech, point) for point in points]

    return pd.DataFrame({"Area": available_areas, "Shape": grid_cells_shapes},
                        index=pd.MultiIndex.from_tuples(sorted(tech_point_tuples)))


# TODO: finish this function when needed
def get_land_availability_for_countries(countries: List[str], tech: str, tech_config: Dict):
    """
    Returns

    Parameters
    ----------
    countries
    tech
    tech_config

    Returns
    -------

    """

    all_shapes = get_shapes(countries, which='onshore_offshore', save=True)
    shapes = all_shapes[~all_shapes['offshore']]["geometry"].values
    if tech in ["wind_offshore", "wind_floating"]:
        shapes = all_shapes[all_shapes['offshore']]["geometry"].values
    land_availability = get_land_availability_for_shapes_mp(shapes, tech_config, True)


if __name__ == '__main__':

    # Define polys for which we want to get available area
    from src.data.geographics import get_subregions, get_shapes
    from shapely.ops import unary_union
    import yaml
    # TODO: need to change
    config_fn = join(dirname(abspath(__file__)), f"../../../data/technologies/vres_tech_config.yml")
    tech_config_ = yaml.load(open(config_fn, "r"), Loader=yaml.FullLoader)
    tech_ = "wind_onshore"
    region_ = 'BE'
    subregions_ = get_subregions(region_)
    all_shapes = get_shapes(subregions_, which='onshore_offshore')
    print(all_shapes)
    if tech_ in ["wind_floating", "wind_offshore"]:
        union = unary_union(all_shapes[all_shapes['offshore']]["geometry"].values)
    else:
        union = unary_union(all_shapes[~all_shapes['offshore']]["geometry"].values)

    spatial_res = 0.5
    land_availability = get_land_availability_for_shapes([union], tech_config_[tech_], processes=4)
