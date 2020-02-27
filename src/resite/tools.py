from src.data.geographics.manager import get_onshore_shapes, get_offshore_shapes, \
    match_points_to_region, return_ISO_codes_from_countries, get_subregions, display_polygons
from numpy import arange, interp, asarray, clip, array, max
import xarray as xr
import xarray.ufuncs as xu
from os.path import *
import os
import geopandas as gpd
import pandas as pd
from pandas import read_csv, read_excel, Series, notnull
import scipy.spatial
from ast import literal_eval
import windpowerlib.power_curves, windpowerlib.wind_speed
from copy import copy
import dask.array as da
import yaml
import geopy.distance
import numpy as np
from typing import *
from shapely.ops import cascaded_union
import glob


def get_tech_points_tuples(tech_points_dict: Dict[str, List[Tuple[float, float]]]) \
        -> List[Tuple[str, Tuple[float, float]]]:
    """
    Returns a list of tuples (tech, point) corresponding to the elements of a dictionary with tech as keys
    associated to list of points.
    """
    return [(tech, point) for tech, points in tech_points_dict.items() for point in points]


# TODO:
#  - need to more precise on description of function, and name it more specifically
#  - add the filtering on coordinates
def read_database(file_path, coords=None):
    """
    Reads resource database from .nc files.

    Parameters
    ----------
    file_path : str
        Relative path to resource data.

    Returns
    -------
    dataset: xarray.Dataset

    """
    # Read through all files, extract the first 2 characters (giving the
    # macro-region) and append in a list that will keep the unique elements.
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    areas = []
    datasets = []
    for item in files:
        areas.append(item[:2])
    areas_unique = list(set(areas))

    # For each available area use the open_mfdataset method to open
    # all associated datasets, while directly concatenating on time dimension
    # and also aggregating (longitude, latitude) into one single 'location'. As
    # well, data is read as float32 (memory concerns).
    for area in areas_unique:
        file_list = [file for file in glob.glob(file_path + '/*.nc') if area in file]
        ds = xr.open_mfdataset(file_list,
                               combine='by_coords',
                               chunks={'latitude': 20, 'longitude': 20})\
                        .stack(locations=('longitude', 'latitude')).astype(np.float32)
        datasets.append(ds)

    # Concatenate all regions on locations.
    dataset = xr.concat(datasets, dim='locations')
    # Removing duplicates potentially there from previous concat of multiple regions.
    _, index = np.unique(dataset['locations'], return_index=True)
    dataset = dataset.isel(locations=index)
    # dataset = dataset.sel(locations=~dataset.indexes['locations'].duplicated(keep='first'))
    # Sorting dataset on coordinates (mainly due to xarray peculiarities between concat and merge).
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    # Remove attributes from datasets. No particular use, looks cleaner.
    dataset.attrs = {}

    return dataset


# TODO: comment
def read_filter_database(filename, coords=None):

    dataset = xr.open_dataset(filename)
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])

    # Changing longitude from 0-360 to -180-180
    dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    if coords is not None:
        dataset = dataset.sel(locations=coords)

    return dataset


# TODO: Should we actually return the points to keep rather than the points to remove?
def filter_points_by_layer(filter_name: str, points: List[Tuple[float, float]], spatial_resolution: float,
                           tech_dict: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Compute locations to remove from the initial set following various
    land-, resource-, populatio-based criteria.

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
    points_to_remove : List[Tuple[float, float]]
        List of points to be removed from the initial set.

    """
    if filter_name == 'protected_areas':

        protected_areas_selection = tech_dict['protected_areas_selection']
        threshold_distance = tech_dict['protected_areas_distance_threshold']

        path_land_data = join(dirname(abspath(__file__)), '../../data/land_data/WDPA_Feb2019-shapefile-points.shp')
        dataset = gpd.read_file(path_land_data)

        # Retrieve the geopandas Point objects and their coordinates
        dataset = dataset[dataset['IUCN_CAT'].isin(protected_areas_selection)]
        protected_points = dataset.geometry.apply(lambda p: (round(p[0].x, 2), round(p[0].y, 2))).values

        # Compute closest protected point for each coordinae
        protected_points = np.array([[p[0], p[1]] for p in protected_points])
        points = np.array([[p[0], p[1]] for p in points])
        closest_points = \
            protected_points[np.argmin(scipy.spatial.distance.cdist(protected_points, points, 'euclidean'), axis=0)]

        # Remove coordinates that are too close to protected areas
        points_to_remove = []
        for coord1, coord2 in zip(points, closest_points):
            if geopy.distance.geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).km < threshold_distance:
                points_to_remove.append(tuple(coord1))

    elif filter_name == 'resource_quality':

        # TODO: still fucking slow, make no sense to be so slow
        # TODO: does it make sense to reload this dataset?
        path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(spatial_resolution))
        database = read_database(path_resource_data)
        database = database.sel(locations=sorted(points))
        # TODO: slice on time?

        if tech_dict['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 + database.v100 ** 2)
        elif tech_dict['resource'] == 'pv':
            array_resource = database.ssrd / 3600.
        else:
            raise ValueError("Error: Resource must be wind or pv")

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < tech_dict['resource_threshold'], 0)
        coords_mask_resource = mask_resource[da.nonzero(mask_resource)].locations.values.tolist()
        points_to_remove = list(set(points).intersection(set(coords_mask_resource)))

    elif filter_name == 'orography':

        dataset_name = join(dirname(abspath(__file__)),
                            '../../data/land_data/ERA5_orography_characteristics_20181231_' + str(spatial_resolution) + '.nc')
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
        points_to_remove = list(set(points).intersection(points_mask_orography))

    elif filter_name == 'forestry':

        dataset_name = join(dirname(abspath(__file__)),
                            '../../data/land_data/ERA5_surface_characteristics_20181231_'+str(spatial_resolution)+'.nc')
        dataset = read_filter_database(dataset_name, points)

        forestry_threshold = tech_dict['forestry_ratio_threshold']

        array_forestry = dataset['cvh']

        mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
        points_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

        points_to_remove = list(set(points).intersection(set(points_mask_forestry)))

    elif filter_name == 'water_mask':

        dataset_name = join(dirname(abspath(__file__)),
                            '../../data/land_data/ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc')
        dataset = read_filter_database(dataset_name, points)

        array_watermask = dataset['lsm']

        mask_watermask = array_watermask.where(array_watermask.data < 0.9)
        points_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

        points_to_remove = list(set(points).intersection(set(points_mask_watermask)))

    elif filter_name == 'bathymetry':

        dataset_name = join(dirname(abspath(__file__)),
                            '../../data/land_data/ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc')
        dataset = read_filter_database(dataset_name, points)

        depth_threshold_low = tech_dict['depth_threshold_low']
        depth_threshold_high = tech_dict['depth_threshold_high']

        array_watermask = dataset['lsm']
        # Careful with this one because max depth is 999.
        array_bathymetry = dataset['wmb'].fillna(0.)

        mask_offshore = array_bathymetry.where((
            (array_bathymetry.data < depth_threshold_low) | (array_bathymetry.data > depth_threshold_high)) | \
            (array_watermask.data > 0.1))
        points_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

        points_to_remove = list(set(points).intersection(set(points_mask_offshore)))

    elif filter_name == 'population_density':

        path_population_data = \
            join(dirname(abspath(__file__)),
                 '../../data/population_density/gpw_v4_population_density_rev11_' + str(spatial_resolution) + '.nc')
        dataset = xr.open_dataset(path_population_data)

        varname = [item for item in dataset.data_vars][0]
        dataset = dataset.rename({varname: 'data'})
        # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
        data_pop = dataset.sel(raster=5)

        array_pop_density = data_pop['data'].interp(longitude=arange(-180, 180, float(spatial_resolution)),
                                                    latitude=arange(-89, 91, float(spatial_resolution))[::-1],
                                                    method='nearest').fillna(0.)
        array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude'))

        population_density_threshold_low = tech_dict['population_density_threshold_low']
        population_density_threshold_high = tech_dict['population_density_threshold_high']

        mask_population = array_pop_density.where((array_pop_density.data < population_density_threshold_low) |
                                                  (array_pop_density.data > population_density_threshold_high))
        points_mask_population = mask_population[mask_population.notnull()].locations.values.tolist()

        points_to_remove = list(set(points).intersection(set(points_mask_population)))

    else:

        raise ValueError(' Layer {} is not available.'.format(str(filter_name)))

    return points_to_remove


def filter_points(technologies: List[str], init_points: List[Tuple[float, float]], spatial_resolution: float,
                  filtering_layers: Dict[str, bool]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns the set of potential deployment locations for each region and available technology.

    Parameters
    ----------
    init_points : List[Tuple(float, float)]
        List of points to filter
    spatial_resolution : float
        Spatial resolution at which the points are defined.
    technologies : List[str]
        List of technologies for which we want to filter points.
    filtering_layers: Dict[str, bool]
        Dictionary indicating if a given filtering layers needs to be applied. If the layer name is present as key and
        associated to a True boolean, then the corresponding is applied.

        List of possible filter names:

        resource_quality:
            If taken into account, discard points whose average resource quality over 
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
    # TODO: take care of the case where you get a empty list of coordinates?

    tech_config_path = join(dirname(abspath(__file__)), 'config_techs.yml')
    tech_config = yaml.safe_load(open(tech_config_path))
    tech_points_dict = dict.fromkeys(technologies)

    for tech in technologies:

        tech_dict = tech_config[tech]

        points = copy(init_points)
        for key in filtering_layers:

            # Apply the filter if it is set to true
            if filtering_layers[key]:

                # Some filter should not apply to some technologies
                if key == 'bathymetry' and tech_dict['deployment'] in ['onshore', 'utility', 'residential']:
                    continue
                if key in ['orography', 'population_density', 'protected_areas', 'forestry', 'water_mask'] \
                        and tech_dict['deployment'] in ['offshore', 'floating']:
                    continue
                points_to_remove = filter_points_by_layer(key, points, spatial_resolution, tech_dict)
                points = list(set(points) - set(points_to_remove))

        tech_points_dict[tech] = points

    return tech_points_dict


# TODO:
#  - replace using atlite?
#  - data. ?
#  - Might be nice to decrease precision of values
def compute_capacity_factors(tech_points_dict: Dict[str, List[Tuple[float, float]]],
                             spatial_res: float, timestamps: List[np.datetime64],
                             smooth_wind_power_curve: bool = True) -> pd.DataFrame:
    """
    Computes capacity factors for a list of points associated to a list of technologies.

    Parameters
    ----------
    tech_points_dict : Dict[str, List[Tuple[float, float]]]
        Dictionary associating to each tech a list of points.
    spatial_res: float
        Spatial resolution of coordinates
    timestamps: List[np.datetime64]
        Time stamps for which we want capacity factors
    smooth_wind_power_curve : boolean
        If "True", the transfer function of wind assets replicates the one of a wind farm,
        rather than one of a wind turbine.

    Returns
    -------
    cap_factor_df : pd.DataFrame
         DataFrame storing capacity factors for each technology and each point

    """
    tech_config_path = join(dirname(abspath(__file__)), 'config_techs.yml')
    tech_dict = yaml.safe_load(open(tech_config_path))

    path_to_transfer_function = join(dirname(abspath(__file__)), '../../data/transfer_functions/')
    data_converter_wind = read_csv(join(path_to_transfer_function, 'data_wind_turbines.csv'), sep=';', index_col=0)
    data_converter_pv = read_csv(join(path_to_transfer_function, 'data_pv_modules.csv'), sep=';', index_col=0)

    path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(spatial_res))
    dataset = read_database(path_resource_data).sel(time=timestamps)

    # Create output dataframe with mutliindex (tech, coords)
    tech_points_tuples = get_tech_points_tuples(tech_points_dict)
    cap_factor_df = pd.DataFrame(index=timestamps,
                                 columns=pd.MultiIndex.from_tuples(tech_points_tuples,
                                                                   names=['technologies', 'coordinates']))

    for tech in tech_points_dict.keys():

        resource = tech.split('_')[0]
        converter = tech_dict[tech]['converter']
        sub_dataset = dataset.sel(locations=sorted(tech_points_dict[tech]))

        if resource == 'wind':

            # TODO: should that be a parameter?
            wind_speed_height = 100.
            array_roughness = sub_dataset.fsr

            # Compute wind speed for the all the coordinates
            wind = xu.sqrt(sub_dataset.u100 ** 2 + sub_dataset.v100 ** 2)
            wind_log = windpowerlib.wind_speed.logarithmic_profile(
                wind.values, wind_speed_height,
                float(data_converter_wind.loc['Hub height [m]', converter]),
                array_roughness.values)
            wind_data = da.from_array(wind_log, chunks='auto', asarray=True)

            # Get the transfer function curve
            # literal_eval converts a string to an array (in this case)
            power_curve_array = literal_eval(data_converter_wind.loc['Power curve', converter])
            wind_speed_references = asarray([i[0] for i in power_curve_array])
            capacity_factor_references = asarray([i[1] for i in power_curve_array])
            capacity_factor_references_pu = capacity_factor_references / max(capacity_factor_references)

            # The transfer function of wind assets replicates the one of a wind farm rather than one of a wind turbine.
            if smooth_wind_power_curve:

                turbulence_intensity = wind.std(dim='time') / wind.mean(dim='time')

                capacity_factor_farm = windpowerlib.power_curves.smooth_power_curve(
                    Series(wind_speed_references), Series(capacity_factor_references_pu),
                    standard_deviation_method='turbulence_intensity',
                    turbulence_intensity=float(turbulence_intensity.min().values),
                    wind_speed_range=10.0)  # TODO: parametrize ?

                power_output = da.map_blocks(interp, wind_data,
                                             capacity_factor_farm['wind_speed'].values,
                                             capacity_factor_farm['value'].values).compute()
            else:

                power_output = da.map_blocks(interp, wind_data,
                                             wind_speed_references,
                                             capacity_factor_references_pu).compute()

        elif resource == 'pv':

            # Get irradiance in W from J
            irradiance = sub_dataset.ssrd / 3600.
            # Get temperature in C from K
            temperature = sub_dataset.t2m - 273.15

            # Homer equation here:
            # https://www.homerenergy.com/products/pro/docs/latest/how_homer_calculates_the_pv_array_power_output.html
            # https://enphase.com/sites/default/files/Enphase_PVWatts_Derate_Guide_ModSolar_06-2014.pdf
            power_output = (float(data_converter_pv.loc['f', converter]) *
                            (irradiance/float(data_converter_pv.loc['G_ref', converter])) *
                            (1. + float(data_converter_pv.loc['k_P [%/C]', converter])/100. *
                             (temperature - float(data_converter_pv.loc['t_ref', converter]))))

        else:
            raise ValueError(' The resource specified is not available yet.')

        # TODO: it is pretty strange because we get a list when resource = wind and an xarray when resource = pv
        #  Should we homogenize this?
        power_output = np.array(power_output)
        cap_factor_df[tech] = power_output

    # Decrease precision of capacity factors
    cap_factor_df = cap_factor_df.round(2)

    return cap_factor_df


# TODO:
#  - need to change this - use what I have done in my code or try at least
#  - data.res_potential
def update_potential_files(input_ds: pd.DataFrame, tech: str) -> pd.DataFrame:
    """
    Updates NUTS2 potentials with i) non-EU data and ii) re-indexed (2013 vs 2016) NUTS2 regions.

    Parameters
    ----------
    input_ds: pd.DataFrame
    tech : str

    Returns
    -------
    input_ds : pd.DataFrame
    """

    if tech in ['wind_onshore', 'pv_residential', 'pv_utility']:

        dict_regions_update = {'FR21': 'FRF2', 'FR22': 'FRE2', 'FR23': 'FRD1', 'FR24': 'FRB0', 'FR25': 'FRD2',
                               'FR26': 'FRC1', 'FR30': 'FRE1', 'FR41': 'FRF3', 'FR42': 'FRF1', 'FR43': 'FRC2',
                               'FR51': 'FRG0', 'FR52': 'FRH0', 'FR53': 'FRI3', 'FR61': 'FRI1', 'FR62': 'FRJ2',
                               'FR63': 'FRI2', 'FR71': 'FRK2', 'FR72': 'FRK1', 'FR81': 'FRJ1', 'FR82': 'FRL0',
                               'FR83': 'FRM0', 'PL11': 'PL71', 'PL12': 'PL9', 'PL31': 'PL81', 'PL32': 'PL82',
                               'PL33': 'PL72', 'PL34': 'PL84', 'UKM2': 'UKM7'}

        new_index = [dict_regions_update[x] if x in dict_regions_update else x for x in input_ds.index]
        input_ds.index = new_index

    if tech == 'wind_onshore':

        input_ds.loc['AL01'] = 2.
        input_ds.loc['AL02'] = 2.
        input_ds.loc['AL03'] = 2.
        input_ds.loc['BA'] = 3.
        input_ds.loc['ME00'] = 3.
        input_ds.loc['MK00'] = 5.
        input_ds.loc['RS11'] = 0.
        input_ds.loc['RS12'] = 10.
        input_ds.loc['RS21'] = 10.
        input_ds.loc['RS22'] = 10.
        input_ds.loc['CH01'] = 1.
        input_ds.loc['CH02'] = 1.
        input_ds.loc['CH03'] = 1.
        input_ds.loc['CH04'] = 1.
        input_ds.loc['CH05'] = 1.
        input_ds.loc['CH06'] = 1.
        input_ds.loc['CH07'] = 1.
        input_ds.loc['NO01'] = 3.
        input_ds.loc['NO02'] = 3.
        input_ds.loc['NO03'] = 3.
        input_ds.loc['NO04'] = 3.
        input_ds.loc['NO05'] = 3.
        input_ds.loc['NO06'] = 3.
        input_ds.loc['NO07'] = 3.
        input_ds.loc['IE04'] = input_ds.loc['IE01']
        input_ds.loc['IE05'] = input_ds.loc['IE02']
        input_ds.loc['IE06'] = input_ds.loc['IE02']
        input_ds.loc['LT01'] = input_ds.loc['LT00']
        input_ds.loc['LT02'] = input_ds.loc['LT00']
        input_ds.loc['UKM8'] = input_ds.loc['UKM3']
        input_ds.loc['UKM9'] = input_ds.loc['UKM3']
        input_ds.loc['PL92'] = input_ds.loc['PL9']
        input_ds.loc['PL91'] = 0.
        input_ds.loc['HU11'] = 0.
        input_ds.loc['HU12'] = input_ds.loc['HU10']
        input_ds.loc['UKI5'] = 0.
        input_ds.loc['UKI6'] = 0.
        input_ds.loc['UKI7'] = 0.

    elif tech == 'wind_offshore':

        input_ds.loc['EZAL'] = 2.
        input_ds.loc['EZBA'] = 0.
        input_ds.loc['EZME'] = 0.
        input_ds.loc['EZMK'] = 0.
        input_ds.loc['EZRS'] = 0.
        input_ds.loc['EZCH'] = 0.
        input_ds.loc['EZNO'] = 20.
        input_ds.loc['EZIE'] = 20.
        input_ds.loc['EZEL'] = input_ds.loc['EZGR']

    elif tech == 'wind_floating':

        input_ds.loc['EZAL'] = 2.
        input_ds.loc['EZBA'] = 0.
        input_ds.loc['EZME'] = 0.
        input_ds.loc['EZMK'] = 0.
        input_ds.loc['EZRS'] = 0.
        input_ds.loc['EZCH'] = 0.
        input_ds.loc['EZNO'] = 100.
        input_ds.loc['EZIE'] = 120.
        input_ds.loc['EZEL'] = input_ds.loc['EZGR']

    elif tech == 'pv_residential':

        input_ds.loc['AL01'] = 1.
        input_ds.loc['AL02'] = 1.
        input_ds.loc['AL03'] = 1.
        input_ds.loc['BA'] = 3.
        input_ds.loc['ME00'] = 1.
        input_ds.loc['MK00'] = 1.
        input_ds.loc['RS11'] = 5.
        input_ds.loc['RS12'] = 2.
        input_ds.loc['RS21'] = 2.
        input_ds.loc['RS22'] = 2.
        input_ds.loc['CH01'] = 6.
        input_ds.loc['CH02'] = 6.
        input_ds.loc['CH03'] = 6.
        input_ds.loc['CH04'] = 6.
        input_ds.loc['CH05'] = 6.
        input_ds.loc['CH06'] = 6.
        input_ds.loc['CH07'] = 6.
        input_ds.loc['NO01'] = 3.
        input_ds.loc['NO02'] = 0.
        input_ds.loc['NO03'] = 3.
        input_ds.loc['NO04'] = 3.
        input_ds.loc['NO05'] = 0.
        input_ds.loc['NO06'] = 0.
        input_ds.loc['NO07'] = 0.
        input_ds.loc['IE04'] = input_ds.loc['IE01']
        input_ds.loc['IE05'] = input_ds.loc['IE02']
        input_ds.loc['IE06'] = input_ds.loc['IE02']
        input_ds.loc['LT01'] = input_ds.loc['LT00']
        input_ds.loc['LT02'] = input_ds.loc['LT00']
        input_ds.loc['UKM8'] = input_ds.loc['UKM3']
        input_ds.loc['UKM9'] = input_ds.loc['UKM3']
        input_ds.loc['PL92'] = input_ds.loc['PL9']
        input_ds.loc['PL91'] = 5.
        input_ds.loc['HU11'] = input_ds.loc['HU10']
        input_ds.loc['HU12'] = input_ds.loc['HU10']
        input_ds.loc['UKI5'] = 1.
        input_ds.loc['UKI6'] = 1.
        input_ds.loc['UKI7'] = 1.

    elif tech == 'pv_utility':

        input_ds.loc['AL01'] = 1.
        input_ds.loc['AL02'] = 1.
        input_ds.loc['AL03'] = 1.
        input_ds.loc['BA'] = 3.
        input_ds.loc['ME00'] = 1.
        input_ds.loc['MK00'] = 1.
        input_ds.loc['RS11'] = 0.
        input_ds.loc['RS12'] = 2.
        input_ds.loc['RS21'] = 2.
        input_ds.loc['RS22'] = 1.
        input_ds.loc['CH01'] = 6.
        input_ds.loc['CH02'] = 6.
        input_ds.loc['CH03'] = 6.
        input_ds.loc['CH04'] = 6.
        input_ds.loc['CH05'] = 6.
        input_ds.loc['CH06'] = 6.
        input_ds.loc['CH07'] = 6.
        input_ds.loc['NO01'] = 3.
        input_ds.loc['NO02'] = 0.
        input_ds.loc['NO03'] = 3.
        input_ds.loc['NO04'] = 3.
        input_ds.loc['NO05'] = 0.
        input_ds.loc['NO06'] = 0.
        input_ds.loc['NO07'] = 0.
        input_ds.loc['IE04'] = input_ds.loc['IE01']
        input_ds.loc['IE05'] = input_ds.loc['IE02']
        input_ds.loc['IE06'] = input_ds.loc['IE02']
        input_ds.loc['LT01'] = input_ds.loc['LT00']
        input_ds.loc['LT02'] = input_ds.loc['LT00']
        input_ds.loc['UKM8'] = input_ds.loc['UKM3']
        input_ds.loc['UKM9'] = input_ds.loc['UKM3']
        input_ds.loc['PL92'] = input_ds.loc['PL9']
        input_ds.loc['PL91'] = 2.
        input_ds.loc['HU11'] = 0.
        input_ds.loc['HU12'] = 2.
        input_ds.loc['UKI5'] = 0.
        input_ds.loc['UKI6'] = 0.
        input_ds.loc['UKI7'] = 0.

    regions_to_remove = ['AD00', 'SM00', 'CY00', 'LI00', 'FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5', 'ES63', 'ES64', 'ES70',
                         'HU10', 'IE01', 'IE02', 'LT00', 'UKM3']

    input_ds = input_ds.drop(regions_to_remove, errors='ignore')

    return input_ds


# TODO:
#  - data.res_potential or  data.cap_potential
#  - merge with my code
#  - Need at least to add as argument a list of codes for which we want the capacity
def capacity_potential_from_enspresso(tech: str) -> pd.DataFrame:
    """
    Returning capacity potential per NUTS2 region for a given tech, based on the ENSPRESSO dataset.

    Parameters
    ----------
    tech : str
        Technology name among 'wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility' and 'pv_residential'

    Returns
    -------
    nuts2_capacity_potentials: pd.DataFrame
        Dict storing technical potential per NUTS2 region.
    """
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    path_potential_data = join(dirname(abspath(__file__)), '../../data/res_potential/source/ENSPRESO')
    if tech == 'wind_onshore':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        onshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Onshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide high restrictions')]

        nuts2_capacity_potentials_ds = onshore_wind.groupby(onshore_wind.index)['Value'].sum()

    elif tech == 'wind_offshore':

        offshore_categories = ['12nm zone, water depth 0-30m', '12nm zone, water depth 30-60m',
                               '12nm zone, water depth 60-100m Floating', 'Water depth 0-30m',
                               'Water depth 30-60m', 'Water depth 60-100m Floating']

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        offshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Offshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide low restrictions') &
            (cap_potential_file['Offshore categories'].isin(offshore_categories))]
        nuts2_capacity_potentials_ds = offshore_wind.groupby(offshore_wind.index)['Value'].sum()

    elif tech == 'wind_floating':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        offshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Offshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide low restrictions') &
            (cap_potential_file['Wind condition'] == 'CF > 25%') &
            (cap_potential_file['Offshore categories'] == 'Water depth 100-1000m Floating')]
        nuts2_capacity_potentials_ds = offshore_wind.groupby(offshore_wind.index)['Value'].sum()

    elif tech == 'pv_utility':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                        sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)
        nuts2_capacity_potentials_ds = cap_potential_file['PV - ground']

    elif tech == 'pv_residential':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                        sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)
        nuts2_capacity_potentials_ds = cap_potential_file['PV - roof/facades']

    # TODO: need to update this function
    return update_potential_files(nuts2_capacity_potentials_ds, tech)


# TODO:
#  - data.res_potential or  data.cap_potential
def get_capacity_potential(tech_points_dict: Dict[str, List[Tuple[float, float]]], spatial_resolution: float,
                           regions: List[str], existing_capacity_ds: pd.Series = None) -> pd.Series:
    """
    Computes the capacity that can potentially be deployed at

    Parameters
    ----------
    tech_points_dict : Dict[str, Dict[str, List[Tuple[float, float]]]
        Dictionary associating to each tech a list of points.
    spatial_resolution : float
        Spatial resolution of the points.
    regions: List[str]
        Codes of geographical regions in which the points are situated
    existing_capacity_ds: pd.Series (default: None)
        Data series given for each tuple of (tech, point) the existing capacity.

    Returns
    -------
    capacity_potential_df : pd.Series
        TODO comment
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in tech_points_dict.keys():
        assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    # Load population density dataset
    path_pop_data = join(dirname(abspath(__file__)), '../../data/population_density')
    dataset_population = \
        xr.open_dataset(join(path_pop_data, 'gpw_v4_population_density_rev11_' + str(spatial_resolution) + '.nc'))
    # Rename the only variable to data # TODO: is there not a cleaner way to do this?
    varname = [item for item in dataset_population.data_vars][0]
    dataset_population = dataset_population.rename({varname: 'data'})
    # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
    data_pop = dataset_population.sel(raster=5)

    # Compute population density at intermediate points
    array_pop_density = data_pop['data'].interp(longitude=arange(-180, 180, float(spatial_resolution)),
                                                latitude=arange(-89, 91, float(spatial_resolution))[::-1],
                                                method='linear').fillna(0.)
    array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude'))

    subregions = []
    for region in regions:
        subregions += get_subregions(region)

    tech_coords_tuples = get_tech_points_tuples(tech_points_dict)
    capacity_potential_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_coords_tuples))

    for tech in tech_points_dict.keys():

        # Get coordinates for which we want capacity
        coords = tech_points_dict[tech]

        # Compute potential for each NUTS2 or EEZ
        potential_per_subregion_df = capacity_potential_from_enspresso(tech)

        # Get NUTS2 and EEZ shapes
        # TODO: this is shit -> not generic enough, expl: would probably not work for us states
        #  would need to get this out of the loop
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../output/geographics/")
        if tech in ['wind_offshore', 'wind_floating']:
            onshore_shapes_union = \
                cascaded_union(get_onshore_shapes(subregions, filterremote=True, save_file=output_dir + ''.join(sorted(subregions)) + "_subregions_on.geojson")["geometry"].values)
            filter_shape_data = get_offshore_shapes(subregions, onshore_shape=onshore_shapes_union,
                                                    filterremote=True, save_file=output_dir + ''.join(sorted(subregions)) + "_subregions_off.geojson")
            filter_shape_data.index = ["EZ" + code if code != 'GB' else 'EZUK' for code in filter_shape_data.index]
        else:
            codes = [code for code in potential_per_subregion_df.index if code[:2] in subregions]
            filter_shape_data = get_onshore_shapes(codes, filterremote=True, save_file=output_dir + ''.join(sorted(subregions)) + "_nuts2_on.geojson")

        # Find the geographical region code associated to each coordinate
        # TODO: might be cleaner to use a pd.series
        coords_to_subregions_df = match_points_to_region(coords, filter_shape_data)

        if tech in ['wind_offshore', 'wind_floating']:

            # For offshore sites, divide the total potential of the region by the number of coordinates
            # associated to that region
            # TODO: change variable names
            region_freq_ds = coords_to_subregions_df.groupby(['subregion'])['subregion'].count()
            region_freq_df = pd.DataFrame(region_freq_ds.values, index=region_freq_ds.index, columns=['freq'])
            region_freq_df["cap_pot"] = potential_per_subregion_df[region_freq_df.index]
            coords_to_subregions_df = \
                coords_to_subregions_df.merge(region_freq_df,
                                              left_on='subregion', right_on='subregion', right_index=True)
            capacity_potential = coords_to_subregions_df["cap_pot"]/coords_to_subregions_df["freq"]
            capacity_potential_ds.loc[tech, capacity_potential.index] = capacity_potential.values

        elif tech in ['wind_onshore', 'pv_utility', 'pv_residential']:

            # TODO: change variable names
            coords_to_subregions_df['pop_dens'] = \
                 clip(array_pop_density.sel(locations=coords).values, a_min=1., a_max=None)
            if tech in ['wind_onshore', 'pv_utility']:
                coords_to_subregions_df['pop_dens'] = 1./coords_to_subregions_df['pop_dens']
            coords_to_subregions_df_sum = coords_to_subregions_df.groupby(['subregion']).sum()
            coords_to_subregions_df_sum["cap_pot"] = potential_per_subregion_df[coords_to_subregions_df_sum.index]
            coords_to_subregions_df_sum.columns = ['sum_per_subregion', 'cap_pot']
            coords_to_subregions_df_merge = \
                coords_to_subregions_df.merge(coords_to_subregions_df_sum,
                                              left_on='subregion', right_on='subregion', right_index=True)

            capacity_potential_per_coord = coords_to_subregions_df_merge['pop_dens'] * \
                coords_to_subregions_df_merge['cap_pot']/coords_to_subregions_df_merge['sum_per_subregion']
            capacity_potential_ds.loc[tech, capacity_potential_per_coord.index] = capacity_potential_per_coord.values

    # Update capacity potential with existing potential if present
    if existing_capacity_ds is not None:
        underestimated_capacity = existing_capacity_ds > capacity_potential_ds
        capacity_potential_ds[underestimated_capacity] = existing_capacity_ds[underestimated_capacity]

    return capacity_potential_ds


# TODO:
#  - data.existing_cap
def read_legacy_capacity_data(tech: str, regions: List[str], points: List[Tuple[float, float]]) \
        -> Dict[List[Tuple[float, float]], float]:
    """
    Reads dataset of existing RES units in the given area and associated to the closest points. Available for EU only.

    Parameters
    ----------
    tech: str
        Technology for which we want existing capacity
    regions: List[str]
        List of regions codes for which we want data
    points : List[Tuple[float, float]]
        Points to which existing capacity must be associated

    Returns
    -------
    point_capacity_dict : Dict[List[Tuple[float, float]], float]
        Dictionary storing existing capacities per node.
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility']
    assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    path_legacy_data = join(dirname(abspath(__file__)), '../../data/legacy')

    if tech in ["wind_onshore", "wind_offshore"]:

        data = read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127.xls'), sheet_name='Windfarms',
                          header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
        data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
        data = data[data['Status'] != 'Dismantled']
        data = data[data['ISO code'].isin(regions)]
        # Converting from kW to GW
        data['Total power'] *= 1e-6

        # Keep only onshore or offshore point depending on technology
        if tech == 'wind_onshore':
            capacity_threshold = 0.2  # TODO: ask David -> to avoid to many points, Parametrize
            data = data[data['Area'] != 'Offshore']

        else:  # wind_offhsore
            capacity_threshold = 0.5
            data = data[data['Area'] == 'Offshore']

        # Associate each location with legacy capacity to a point in points
        legacy_capacity_locs = array(list(zip(data['Longitude'], data['Latitude'])))
        points = np.array(points)
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(np.array(points), legacy_capacity_locs, 'euclidean'), axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['Total power'].agg('sum')

        point_capacity_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    else:

        data = read_excel(join(path_legacy_data, 'Solarfarms_Europe_20200208.xlsx'), sheet_name='ProjReg_rpt',
                          header=0, usecols=[0, 3, 4, 5, 8])
        data = data[notnull(data['Coords'])]
        data['Longitude'] = data['Coords'].str.split(',', 1).str[1]
        data['Latitude'] = data['Coords'].str.split(',', 1).str[0]

        # TODO: check if this possibility works -> problem with UK - GB
        # countries_codes_fn = join(dirname(abspath(__file__)),
        #                          "../../data/countries-codes.csv")
        # countries_code = pd.read_csv(countries_codes_fn, index_col="Name")
        # data['ISO code'] = countries_code[["Code"]].to_dict()["Code"]
        data['ISO code'] = data['Country'].map(return_ISO_codes_from_countries())
        data = data[data['ISO code'].isin(regions)]
        # Converting from MW to GW
        data['MWac'] *= 1e-3

        # Associate each location with legacy capacity to a point in points
        # TODO: make a function of this? use kneighbors?
        legacy_capacity_locs = array(list(zip(data['Longitude'], data['Latitude'])))
        points = np.array(points)
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(points, legacy_capacity_locs, 'euclidean'), axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')

        capacity_threshold = 0.05  # TODO: parametrize?
        point_capacity_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    return point_capacity_dict


# TODO:
#  - data.existing_cap
#  - update comments
def get_legacy_capacity(technologies: List[str], regions: List[str], points: List[Tuple[float, float]], spatial_resolution: float) \
        -> Dict[str, Dict[Tuple[float, float], float]]:
    """
    Returns, for each technology and for each point, the existing (legacy) capacity in GW.

    Parameters
    ----------
    technologies: List[str]
        List of technologies for which we want to obtain legacy capacity
    regions: List[str]
        Regions for which we want legacy capacity
    points : List[Tuple[float, float]]
        Points to which existing capacity must be associated
    spatial_resolution: float
        Spatial resolution of the points.

    Returns
    -------
    existing_capacity_dict : Dict[str, Dict[Tuple[float, float], float]]
        Dictionary giving for each technology a dictionary associating each location TODO: with non-zero capacity?
        with its legacy capacity for that technology.

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility']
    for tech in technologies:
        assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    existing_capacity_dict = dict.fromkeys(technologies)
    for tech in technologies:
        # Filter coordinates to obtain only the ones on land or offshore
        onshore = False if tech == 'wind_offshore' else True
        land_filtered_points = filter_onshore_offshore_points(onshore, points, spatial_resolution)
        # Get legacy capacity at points in land_filtered_coordinates where legacy capacity exists
        existing_capacity_dict[tech] = read_legacy_capacity_data(tech, regions, land_filtered_points)

    return existing_capacity_dict


# TODO:
#  - It's actually probably smarter than using shapes to differentiate between onshore and offshore
#  - I think we should actually change the tech argument to an onshore or offshore argument
#  - Or maybe not because then we have the problem of associating offshore points which are considered onshore
def filter_onshore_offshore_points(onshore: bool, points: List[Tuple[float, float]], spatial_resolution: float):
    """
    Filters coordinates to leave only onshore and offshore coordinates depending on technology

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
    coordinates : List[tuple(float, float)]
        Coordinates filtered via land/water mask.
    """

    path_land_data = join(dirname(abspath(__file__)),
                          '../../data/land_data/ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc')
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

