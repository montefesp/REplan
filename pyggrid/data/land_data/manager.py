from typing import List, Tuple, Dict, Any
from copy import copy

import xarray as xr
import numpy as np
import xarray.ufuncs as xu
import dask.array as da
import geopandas as gpd
import scipy.spatial
import geopy.distance

from pyggrid.data.generation.vres.profiles import read_resource_database
from pyggrid.data.indicators.population import load_population_density_data

from pyggrid.data import data_path


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

    path_land_data = f"{data_path}land_data/source/ERA5/ERA5_land_sea_mask_20181231_{spatial_resolution}.nc"
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
    land-, profiles-, population-based criteria.

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
    filters_dict = tech_dict['filters']
    if filter_name == 'protected_areas':

        protected_areas_selection = filters_dict['protected_areas_selection']
        threshold_distance = filters_dict['protected_areas_distance_threshold']

        path_land_data = f"{data_path}land_data/source/WDPA/WDPA_Apr2020-shapefile-points.shp"
        dataset = gpd.read_file(path_land_data)

        # Retrieve the geopandas Point objects and their coordinates
        dataset = dataset[dataset['IUCN_CAT'].isin(protected_areas_selection)]
        protected_points = dataset.geometry.apply(lambda p: (round(p[0].x, 2), round(p[0].y, 2))).values

        # Compute closest protected point for each coordinate
        protected_points = np.array([[p[0], p[1]] for p in protected_points])
        points = np.array([[p[0], p[1]] for p in points])
        closest_points = \
            protected_points[np.argmin(scipy.spatial.distance.cdist(protected_points, points, 'euclidean'), axis=0)]

        # Remove points that are too close to protected areas
        points_to_remove = []
        for point1, point2 in zip(points, closest_points):
            if geopy.distance.geodesic((point1[1], point1[0]), (point2[1], point2[0])).km < threshold_distance:
                points_to_remove.append(tuple(point1))

        points = list(set(points) - set(points_to_remove))

    elif filter_name == 'resource_quality':

        database = read_resource_database(spatial_resolution)
        database = database.sel(locations=sorted(points))

        if tech_dict['plant'] == 'Wind':
            array_resource = xu.sqrt(database.u100 ** 2 + database.v100 ** 2)
        elif tech_dict['plant'] == 'PV':
            array_resource = database.ssrd / 3600.
        else:
            raise ValueError("Error: Resource must be wind or pv")

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < filters_dict['resource_threshold'], 0)
        coords_mask_resource = mask_resource[da.nonzero(mask_resource)].locations.values.tolist()
        points = list(set(points).difference(set(coords_mask_resource)))

    elif filter_name == 'orography':

        dataset_name = f"{data_path}land_data/source/ERA5/" \
                       f"ERA5_orography_characteristics_20181231_{spatial_resolution}.nc"
        dataset = read_filter_database(dataset_name, points)

        altitude_threshold = filters_dict['altitude_threshold']
        slope_threshold = filters_dict['terrain_slope_threshold']

        array_altitude = dataset['z'] / 9.80665
        array_slope = dataset['slor']

        mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
        points_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()

        mask_slope = array_slope.where(array_slope.data > slope_threshold)
        points_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

        points_mask_orography = set(points_mask_altitude).union(set(points_mask_slope))
        points = list(set(points).difference(points_mask_orography))

    elif filter_name == 'forestry':

        dataset_name = f"{data_path}land_data/source/ERA5/" \
                       f"ERA5_surface_characteristics_20181231_{spatial_resolution}.nc"
        dataset = read_filter_database(dataset_name, points)

        forestry_threshold = filters_dict['forestry_ratio_threshold']

        array_forestry = dataset['cvh']

        mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
        points_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_forestry)))

    elif filter_name == 'water_mask':

        dataset_name = f"{data_path}land_data/source/ERA5/ERA5_land_sea_mask_20181231_{spatial_resolution}.nc"
        dataset = read_filter_database(dataset_name, points)

        array_watermask = dataset['lsm']

        mask_watermask = array_watermask.where(array_watermask.data < 0.9)
        points_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_watermask)))

    elif filter_name == 'bathymetry':

        dataset_name = f"{data_path}land_data/source/ERA5/ERA5_land_sea_mask_20181231_{spatial_resolution}.nc"
        dataset = read_filter_database(dataset_name, points)

        depth_threshold_low = filters_dict['depth_thresholds']['low']
        depth_threshold_high = filters_dict['depth_thresholds']['high']

        array_watermask = dataset['lsm']
        # Careful with this one because max depth is 999.
        array_bathymetry = dataset['wmb'].fillna(0.)

        mask_offshore = array_bathymetry.where(((array_bathymetry.data < depth_threshold_low) |
                                                (array_bathymetry.data > depth_threshold_high)) |
                                               (array_watermask.data > 0.1))
        points_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

        points = list(set(points).difference(set(points_mask_offshore)))

    elif filter_name == 'population_density':

        array_pop_density = load_population_density_data(spatial_resolution)

        population_density_threshold_low = filters_dict['population_density_threshold_low']
        population_density_threshold_high = filters_dict['population_density_threshold_high']

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
            If taken into account, discard points whose average profiles quality over
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
                if key == 'bathymetry' and tech_dict['type'] in ['Onshore', 'Utility', 'Residential']:
                    continue
                if key in ['orography', 'population_density', 'protected_areas', 'forestry', 'water_mask'] \
                        and tech_dict['type'] in ['Offshore', 'Floating']:
                    continue
                points = filter_points_by_layer(key, points, spatial_resolution, tech_dict)

        tech_points_dict[tech] = points

    return tech_points_dict
