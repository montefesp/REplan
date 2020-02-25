from src.data.geographics.manager import return_region_shapefile, return_coordinates_from_shapefiles,\
    match_point_to_region, return_ISO_codes_from_countries, get_subregions_list
from src.resite.helpers import update_potential_files, collapse_dict_region_level, return_dict_keys, read_database, get_tech_coords_tuples
from numpy import arange, interp, sqrt, \
                  asarray, clip, array, sum, \
                  max, hstack, dtype, radians, arctan2, sin, cos
import xarray as xr
import xarray.ufuncs as xu
import dask.array as da
from os.path import *
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import pandas as pd
from pandas import read_csv, read_excel, Series, notnull
import scipy.spatial
from ast import literal_eval
import windpowerlib.power_curves, windpowerlib.wind_speed
from collections import Counter
from copy import deepcopy, copy
import dask.array as da
import yaml
import geopy.distance
from time import time
import numpy as np
from typing import *


# TODO:
#  - this function is pretty fat no? divide it?
def filter_locations_by_layer(tech_dict, coords, spatial_resolution, which, filename='dummy'):
    """
    Compute locations to remove from the initial set following various
    land-, resource-, populatio-based criteria.

    Parameters
    ----------
    tech_dict : dict
        Dict object containing technical parameters and constraints of a given technology.
    coords : list
        List of coordinates.
    spatial_resolution : float
        Spatial resolution of the resource data.
    which : str
        Filter to be applied.
    filename : str
        Name of the file; associated with the filter type.

    Returns
    -------
    coords_to_remove : list
        List of coordinates to be removed from the initial set.

    """
    if which == 'protected_areas':

        protected_areas_selection = tech_dict['protected_areas_selection']
        threshold_distance = tech_dict['protected_areas_distance_threshold']

        path_land_data = join(dirname(abspath(__file__)), '../../data/land_data/WDPA_Feb2019-shapefile-points.shp')
        dataset = gpd.read_file(path_land_data)

        # Retrieve the geopandas Point objects and their coordinates
        dataset = dataset[dataset['IUCN_CAT'].isin(protected_areas_selection)]
        protected_coords = dataset.geometry.apply(lambda p: (round(p[0].x, 2), round(p[0].y, 2))).values

        # Compute closest protected point for each coordinae
        protected_coords = np.array([[p[0], p[1]] for p in protected_coords])
        coords = np.array([[p[0], p[1]] for p in coords])
        closest_points = \
            protected_coords[np.argmin(scipy.spatial.distance.cdist(protected_coords, coords, 'euclidean'), axis=0)]

        # Remove coordinates that are too close to protected areas
        coords_to_remove = []
        for coord1, coord2 in zip(coords, closest_points):
            if geopy.distance.geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).km < threshold_distance:
                coords_to_remove.append(tuple(coord1))

    elif which == 'resource':

        # TODO: still fucking slow, make no sense to be so slow
        # TODO: does it make sense to reload this dataset?
        path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(spatial_resolution))
        database = read_database(path_resource_data)
        database = database.sel(locations=coords)
        # TODO: slice on time?

        if tech_dict['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 + database.v100 ** 2)
        elif tech_dict['resource'] == 'solar':
            array_resource = database.ssrd / 3600.
        else:
            print("Error: Resource must be wind or solar")
            exit(1)

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < tech_dict['resource_threshold'], 0)
        coords_mask_resource = mask_resource[da.nonzero(mask_resource)].locations.values.tolist()
        coords_to_remove = list(set(coords).intersection(set(coords_mask_resource)))

    elif which in ['orography', 'forestry', 'water_mask', 'bathymetry']:

        path_land_data = join(dirname(abspath(__file__)), '../../data/land_data')
        dataset = xr.open_dataset(join(path_land_data, filename))
        dataset = dataset.sortby([dataset.longitude, dataset.latitude])

        # Changing longitude from 0-360 to -180-180
        dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby('longitude')
        dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
        dataset = dataset.sel(locations=coords)

        if which == 'orography':

            altitude_threshold = tech_dict['altitude_threshold']
            slope_threshold = tech_dict['terrain_slope_threshold']

            array_altitude = dataset['z'] / 9.80665
            array_slope = dataset['slor']

            mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
            coords_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()

            mask_slope = array_slope.where(array_slope.data > slope_threshold)
            coords_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

            coords_mask_orography = set(coords_mask_altitude).union(set(coords_mask_slope))
            coords_to_remove = list(set(coords).intersection(coords_mask_orography))

        elif which == 'forestry':

            forestry_threshold = tech_dict['forestry_ratio_threshold']

            array_forestry = dataset['cvh']

            mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
            coords_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

            coords_to_remove = list(set(coords).intersection(set(coords_mask_forestry)))

        elif which == 'water_mask':
            array_watermask = dataset['lsm']

            mask_watermask = array_watermask.where(array_watermask.data < 0.9)
            coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

            coords_to_remove = list(set(coords).intersection(set(coords_mask_watermask)))

        elif which == 'bathymetry':

            depth_threshold_low = tech_dict['depth_threshold_low']
            depth_threshold_high = tech_dict['depth_threshold_high']

            array_watermask = dataset['lsm']
            # Careful with this one because max depth is 999.
            array_bathymetry = dataset['wmb'].fillna(0.)

            mask_offshore = array_bathymetry.where((
                (array_bathymetry.data < depth_threshold_low) | (array_bathymetry.data > depth_threshold_high)) | \
                (array_watermask.data > 0.1))
            coords_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

            coords_to_remove = list(set(coords).intersection(set(coords_mask_offshore)))

    elif which == 'population':

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
        coords_mask_population = mask_population[mask_population.notnull()].locations.values.tolist()

        coords_to_remove = list(set(coords).intersection(set(coords_mask_population)))

    else:

        raise ValueError(' Layer {} is not available.'.format(str(which)))

    return coords_to_remove


# TODO
#  - Ask david: Why is protected_areas_layer to False?
def filter_coordinates(all_coordinates, spatial_resolution, technologies, regions,
                       resource_quality_layer=True, population_density_layer=True,
                       protected_areas_layer=False, orography_layer=True, forestry_layer=True,
                       water_mask_layer=True, bathymetry_layer=True, legacy_layer=True):
    """
    Returns the set of potential deployment locations for each region and available technology.

    Parameters
    ----------
    all_coordinates : list
        List of coordinate pairs in the region of interest.
    spatial_resolution : float
        Spatial resolution of the resource data.
    technologies : list
        List of available technologies.
    regions : list
        List of regions.
    resource_quality_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the average resource
        quality over the available time horizon. Resource quality threshold defined
        in the config_tech.yaml file.
    population_density_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the population density.
        Population density threshold defined in the config_tech.yaml file per each
        available technology.
    protected_areas_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the existance of protected
        areas in their vicinity. Distance threshold, as well as classes of areas are
         defined in the config_tech.yaml file.
    orography_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on their altitude and terrain
        slope. Both thresholds defined in the config_tech.yaml file for each individual
        technology.
    forestry_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on its forest cover share.
        Forest share threshold above which technologies are not built are defined
        in the config_tech.yaml file.
    water_mask_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the water coverage share.
        Threshold defined in the config_tech.yaml file.
    bathymetry_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        (valid for offshore technologies) it discards points in coordinates_in_region
        based on the water depth. Associated threshold defined in the config_tech.yaml
        file for offshore and floating wind, respectively.
    legacy_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it adds points to the final set based on the existence of RES projects in the area,
        thus avoiding a greenfield approach.

    Returns
    -------
    output_dict : dict
        Dict object storing potential locations sets per region and technology.

    """
    # TODO: continue to improve this and filter_locations_by_layer
    # TODO: take care of the case where you get a empty list of coordinates?

    tech_config_path = join(dirname(abspath(__file__)), 'config_techs.yml')
    tech_config = yaml.safe_load(open(tech_config_path))
    output_dict = dict.fromkeys(technologies)

    for tech in technologies:
        print(tech)

        tech_dict = tech_config[tech]
        coordinates = copy(all_coordinates)
        start_coordinates = copy(all_coordinates)

        # TODO: I would rearrange this and filter_locations_by_layer
        # print("resource_quality_layer")
        if resource_quality_layer:

            coords_to_remove = filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                                         which='resource')
            coordinates = list(set(coordinates) - set(coords_to_remove))

        if tech_dict['deployment'] in ['onshore', 'utility', 'residential']:

            # print("orography_layer")
            if orography_layer:
                filename = 'ERA5_orography_characteristics_20181231_'+str(spatial_resolution)+'.nc'
                coords_to_remove = filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                                             which='orography', filename=filename)
                coordinates = list(set(coordinates) - set(coords_to_remove))

            # print("population_density_layer")
            if population_density_layer:
                coords_to_remove = filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                                             which='population')
                coordinates = list(set(coordinates) - set(coords_to_remove))

            # print("protected_areas_layer")
            if protected_areas_layer:
                coords_to_remove = filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                                             which='protected_areas')
                coordinates = list(set(coordinates) - set(coords_to_remove))

            # print("forestry_layer")
            if forestry_layer:
                filename = 'ERA5_surface_characteristics_20181231_'+str(spatial_resolution)+'.nc'
                coords_to_remove = \
                    filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                              which='forestry', filename=filename)
                coordinates = list(set(coordinates) - set(coords_to_remove))

            # print("water_mask_layer")
            if water_mask_layer:
                filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
                coords_to_remove = \
                    filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                              which='water_mask', filename=filename)
                coordinates = list(set(coordinates) - set(coords_to_remove))

        elif tech_dict['deployment'] in ['offshore', 'floating']:

            # print("bathymetry_layer")
            if bathymetry_layer:
                filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
                coords_to_remove = \
                    filter_locations_by_layer(tech_dict, coordinates, spatial_resolution,
                                              which='bathymetry', filename=filename)
                coordinates = list(set(coordinates) - set(coords_to_remove))

        """
        # TODO: Move this out of here
        # print("legacy_layer")
        if tech in ["wind_onshore", "wind_offshore", "solar_utility"] and legacy_layer:

            # TODO: need to look if there is not another way to organize this
            land_filtered_coordinates = filter_onshore_offshore_locations(start_coordinates,
                                                                          spatial_resolution, tech)
            legacy_dict = read_legacy_capacity_data(land_filtered_coordinates,
                                                    region_shapes['region_subdivisions'], tech)
            locations = list(legacy_dict.keys())
            coords_to_add_legacy = \
                retrieve_nodes_with_legacy_units(locations, region_shapes['region_shapefiles'], tech)

            coordinates = list(set(coordinates).union(set(coords_to_add_legacy)))
        """
        output_dict[tech] = coordinates

    return output_dict


# TODO:
#  - replace using atlite?
#  - data. ?
def compute_capacity_factors(tech_coordinates_dict: Dict[str, List[Tuple[float, float]]],
                             spatial_res: float, time_stamps: List[np.datetime64],
                             smooth_wind_power_curve: bool = True) -> pd.DataFrame:
    """
    Computes capacity factors for a list of points associated to a list of technologies.

    Parameters
    ----------
    tech_coordinates_dict : Dict[str, List[Tuple[float, float]]]
        Dictionary associating to a techs a list of coordinates.
    spatial_res: float
        Spatial resolution of coordinates
    time_stamps: List[np.datetime64]
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
    data_converter_solar = read_csv(join(path_to_transfer_function, 'data_solar_modules.csv'), sep=';', index_col=0)

    path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(spatial_res))
    dataset = read_database(path_resource_data).sel(time=time_stamps)

    # Create output dataframe with mutliindex (tech, coords)
    tech_coords_tuples = get_tech_coords_tuples(tech_coordinates_dict)
    cap_factor_df = pd.DataFrame(index=time_stamps,
                                 columns=pd.MultiIndex.from_tuples(tech_coords_tuples,
                                                                   names=['technologies', 'coordinates']))

    for tech in tech_coordinates_dict.keys():

        resource = tech.split('_')[0]
        converter = tech_dict[tech]['converter']
        sub_dataset = dataset.sel(locations=tech_coordinates_dict[tech])

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
            # TODO: what does literal_eval do? -> ask david
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

        elif resource == 'solar':

            # Get irradiance in W from J
            irradiance = sub_dataset.ssrd / 3600.
            # Get temperature in C from K
            temperature = sub_dataset.t2m - 273.15

            # Homer equation here:
            # https://www.homerenergy.com/products/pro/docs/latest/how_homer_calculates_the_pv_array_power_output.html
            # https://enphase.com/sites/default/files/Enphase_PVWatts_Derate_Guide_ModSolar_06-2014.pdf
            power_output = (float(data_converter_solar.loc['f', converter]) *
                            (irradiance/float(data_converter_solar.loc['G_ref', converter])) *
                            (1. + float(data_converter_solar.loc['k_P [%/C]', converter])/100. *
                             (temperature - float(data_converter_solar.loc['t_ref', converter]))))

        else:
            raise ValueError(' The resource specified is not available yet.')

        # TODO: it is pretty strange because we get a list when resource = wind and an xarray when resource = solar
        #  Should we homogenize this?
        power_output = np.array(power_output)
        # TODO: why this filtering under 0.01?
        power_output[power_output <= 0.01] = 0.
        cap_factor_df[tech] = power_output

    return cap_factor_df


# TODO:
#  - data.res_potential or  data.cap_potential
#  - merge with my code
#  - Need at least to add as argument a list of codes for which we want the capacity
def capacity_potential_from_enspresso(tech: str):
    """
    Returning capacity potential per NUTS2 region for a given tech, based on the ENSPRESSO dataset.

    Parameters
    ----------
    tech : str
        Technology name among 'wind_onshore', 'wind_offshore', 'wind_floating', 'solar_utility' and 'solar_residential'

    Returns
    -------
    nuts2_capacity_potentials: Dict[str, float]
        Dict storing technical potential per NUTS2 region.
    """
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'solar_utility', 'solar_residential']
    assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    path_potential_data = join(dirname(abspath(__file__)), '../../data/res_potential/source/ENSPRESO')
    if tech == 'wind_onshore':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        onshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Onshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide high restrictions')]
        nuts2_capacity_potentials = onshore_wind.groupby(onshore_wind.index)['Value'].sum().to_dict()

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
        nuts2_capacity_potentials = offshore_wind.groupby(offshore_wind.index)['Value'].sum().to_dict()

    elif tech == 'wind_floating':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        offshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Offshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide low restrictions') &
            (cap_potential_file['Wind condition'] == 'CF > 25%') &
            (cap_potential_file['Offshore categories'] == 'Water depth 100-1000m Floating')]
        nuts2_capacity_potentials = offshore_wind.groupby(offshore_wind.index)['Value'].sum().to_dict()

    elif tech == 'solar_utility':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                        sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)

        nuts2_capacity_potentials = cap_potential_file['PV - ground'].to_dict()

    elif tech == 'solar_residential':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                        sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)

        nuts2_capacity_potentials = cap_potential_file['PV - roof/facades'].to_dict()

    return update_potential_files(nuts2_capacity_potentials, tech)


# TODO:
#  - data.res_potential or  data.cap_potential
def get_capacity_potential(tech_coordinates_dict, regions, spatial_resolution, existing_capacity = None):
    """
    Assigning a technical potential to each individual node: TODO: what node? precise

    Parameters
    ----------
    input_dict : Dict[str, Dict[str, List[tuple(float, float)]]
        Dict object containing coordinates for regions and technologies
    spatial_resolution : float
        Spatial resolution of the study.

    Returns
    -------
    output_dict : Dict[str, Dict[str, xr.DataArray]]
        Dict object storing technical potential for each point per region and technology.

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'solar_utility', 'solar_residential']
    for tech in tech_coordinates_dict.keys():
        assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    # Reading shape files  # TODO: why not used the same function as before (return_region_shapefile)
    path_shapefile_data = join(dirname(abspath(__file__)), '../../data/shapefiles')
    shapefile_data_onshore = gpd.read_file(join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_2_incl_BA.geojson'))
    shapefile_data_offshore = gpd.read_file(join(path_shapefile_data, 'EEZ_RG_01M_2016_4326_LEVL_0.geojson'))

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

    subregions_list = []
    for region in regions:
        subregions_list += get_subregions_list(region)

    tech_coords_tuples = get_tech_coords_tuples(tech_coordinates_dict)
    capacity_potential_df = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_coords_tuples))
    for tech in tech_coordinates_dict.keys():

        # Get NUTS2 and EEZ shapes
        if tech in ['wind_offshore', 'wind_floating']:

            filter_shape_data = \
                shapefile_data_offshore[shapefile_data_offshore['ISO_ID'].str.contains('|'.join(subregions_list))]
            filter_shape_data.index = filter_shape_data['ISO_ID']
            filter_shape_data.index = 'EZ' + filter_shape_data.index

        else:

            filter_shape_data = shapefile_data_onshore[(shapefile_data_onshore['NUTS_ID'].str.contains(
                                                       '|'.join(subregions_list)))]
            filter_shape_data.index = filter_shape_data['NUTS_ID']

        # Compute potential for each NUTS2 or EEZ
        potential_per_subregion = capacity_potential_from_enspresso(tech)
        potential_per_subregion_df = pd.Series(list(potential_per_subregion.values()), index=list(potential_per_subregion.keys()))
        coords = tech_coordinates_dict[tech]
        coords_to_subregions = dict.fromkeys(coords)
        coords_to_subregions_df = pd.DataFrame(columns=['subregion', 'pop_dens'], index=coords) # pd.MultiIndex.from_tuples(coords))
        subregions_to_coords = {}

        # Find the geographical region code associated to each coordinate
        for coord in coords:
            region = match_point_to_region(coord, filter_shape_data, list(potential_per_subregion.keys()))
            coords_to_subregions_df.loc[coord, 'subregion'] = region
            coords_to_subregions[coord] = region
            if region in subregions_to_coords:
                subregions_to_coords[region] += [coord]
            else:
                subregions_to_coords[region] = [coord]

        if tech in ['wind_offshore', 'wind_floating']:

            # For offshore sites, divide the total potential of the region by the number of coordinates
            # associated to that region
            region_freq = dict(Counter(coords_to_subregions.values()).most_common())
            for coord in coords:
                coord_region = coords_to_subregions[coord]
                capacity_potential_df[tech, coord] = float(potential_per_subregion[coord_region]) / \
                    float(region_freq.get(coord_region))

        elif tech in ['wind_onshore', 'solar_utility', 'solar_residential']:

            coords_to_subregions_df['pop_dens'] = \
                 clip(array_pop_density.sel(locations=coords).values, a_min=1., a_max=None)
            if tech in ['wind_onshore', 'solar_utility']:
                coords_to_subregions_df['pop_dens'] = 1./coords_to_subregions_df['pop_dens']
            coords_to_subregions_df_sum = coords_to_subregions_df.groupby(['subregion']).sum()
            coords_to_subregions_df_sum["cap_pot"] = potential_per_subregion_df[coords_to_subregions_df_sum.index]
            print(coords_to_subregions_df_sum)
            coords_to_subregions_df_sum.columns = ['sum_per_subregion', 'cap_pot']
            coords_to_subregions_df_merge = \
                coords_to_subregions_df.merge(coords_to_subregions_df_sum,
                                              left_on='subregion', right_on='subregion', right_index=True)
            print(coords_to_subregions_df_merge['pop_dens']/coords_to_subregions_df_merge['sum_per_subregion'])

            # TODO: this seems pretty inefficient
            for coord in coords:

                # Get population density of the current coordinate
                incumbent_loc_pop_dens = float(array_pop_density.sel(locations=coord).values)
                # TODO: if incumbent_loc_pop_dens is 0, I have an error
                if incumbent_loc_pop_dens < 1.:
                    incumbent_loc_pop_dens = 1.

                # Get population densities of all coordinates associated to the current coordinate region
                loc_in_region = subregions_to_coords[coords_to_subregions[coord]]
                # TODO: why are we clipping?
                region_loc_pop_dens = clip(array_pop_density.sel(locations=loc_in_region).values, a_min=1., a_max=None)

                # Associate to the coordinate a portion of the regions potential
                if tech in ['wind_onshore', 'solar_utility']:
                    # Inversely proportional to population density
                    distribution_key = (1/incumbent_loc_pop_dens) / sum(1/region_loc_pop_dens)
                else:
                    # Proportional to population density
                    distribution_key = incumbent_loc_pop_dens / sum(region_loc_pop_dens)
                capacity_potential_df[tech, coord] = float(potential_per_subregion[coords_to_subregions[coord]]) \
                                                     * distribution_key
            print(capacity_potential_df)

    # Update capacity potential with existing potential if present
    if existing_capacity is not None:
        underestimated_capacity = existing_capacity > capacity_potential_df
        capacity_potential_df[underestimated_capacity] = existing_capacity[underestimated_capacity]

    return capacity_potential_df

# TODO:
#  - data.res_potential or  data.cap_potential
def capacity_potential_per_node(input_dict, spatial_resolution):
    """
    Assigning a technical potential to each individual node: TODO: what node? precise

    Parameters
    ----------
    input_dict : Dict[str, Dict[str, List[tuple(float, float)]]
        Dict object containing coordinates for regions and technologies
    spatial_resolution : float
        Spatial resolution of the study.

    Returns
    -------
    output_dict : Dict[str, Dict[str, xr.DataArray]]
        Dict object storing technical potential for each point per region and technology.

    """

    # Reading shape files # TODO: why not used the same function as before (return_region_shapefile)
    path_shapefile_data = join(dirname(abspath(__file__)), '../../data/shapefiles')
    shapefile_data_onshore = gpd.read_file(join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_2_incl_BA.geojson'))
    shapefile_data_offshore = gpd.read_file(join(path_shapefile_data, 'EEZ_RG_01M_2016_4326_LEVL_0.geojson'))

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

    output_dict = deepcopy(input_dict)
    key_list = return_dict_keys(input_dict)
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'solar_utility', 'solar_residential']
    for region, tech in key_list:

        assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

        subregions_list = get_subregions_list(region)

        # Get NUTS2 and EEZ shapes
        if tech in ['wind_offshore', 'wind_floating']:

            filter_shape_data = \
                shapefile_data_offshore[shapefile_data_offshore['ISO_ID'].str.contains('|'.join(subregions_list))]
            filter_shape_data.index = filter_shape_data['ISO_ID']
            filter_shape_data.index = 'EZ' + filter_shape_data.index

        else:

            filter_shape_data = shapefile_data_onshore[(shapefile_data_onshore['NUTS_ID'].str.contains(
                                                       '|'.join(subregions_list)))]
            filter_shape_data.index = filter_shape_data['NUTS_ID']

        # Compute potential for each NUTS2 or EEZ
        potential_per_tech = capacity_potential_from_enspresso(tech)

        coords = input_dict[region][tech]
        coords_subregions = dict.fromkeys(coords)
        coords_potentials = dict.fromkeys(coords)

        # Find the geographical region code associated to each coordinate
        for coord in coords:
            coords_subregions[coord] = match_point_to_region(coord, filter_shape_data, list(potential_per_tech.keys()))

        if tech in ['wind_offshore', 'wind_floating']:

            # For offshore sites, divide the total potential of the region by the number of coordinates
            # associated to that region
            region_freq = dict(Counter(coords_subregions.values()).most_common())
            for coord in coords:
                coord_region = coords_subregions[coord]
                coords_potentials[coord] = float(potential_per_tech[coord_region]) / \
                                    float(region_freq.get(coord_region))

        elif tech in ['wind_onshore', 'solar_utility', 'solar_residential']:

            # TODO: this seems pretty inefficient
            for coord in coords:

                # Get population density of the current coordinate
                incumbent_loc_pop_dens = float(array_pop_density.sel(locations=coord).values)

                # Get population densities of all coordinates associated to the current coordinate region
                loc_in_region = [key for key, value in coords_subregions.items() if value == coords_subregions[coord]]
                region_loc_pop_dens = clip(array_pop_density.sel(locations=loc_in_region).values, a_min=1., a_max=None)

                # Associate to the coordinate a portion of the regions potential
                if tech in ['wind_onshore', 'solar_utility']:
                    # Inversely proportional to population density
                    distribution_key = (1/incumbent_loc_pop_dens) / sum(1/region_loc_pop_dens)
                else:
                    # Proportional to population density
                    distribution_key = incumbent_loc_pop_dens / sum(region_loc_pop_dens)
                coords_potentials[coord] = float(potential_per_tech[coords_subregions[coord]]) * distribution_key

        # Converting the potential dicts into the right format
        dict_to_xarray = {'coords': {'locations': {'dims': 'locations',
                                                   'data': asarray(coords, dtype='f4,f4')}},
                          'dims': 'locations',
                          'data': asarray(list(coords_potentials.values()), dtype=float)}

        output_dict[region][tech] = xr.DataArray.from_dict(dict_to_xarray)

    return output_dict


# TODO:
#  - data.existing_cap
#  - update comment
def read_legacy_capacity_data(coordinates, region_subdivisions, tech):
    """
    Reads dataset of existing RES units in the given area. Available for EU only.

    Parameters
    ----------
    coordinates : list
    tech : str

    Returns
    -------
    output_dict : dict
        Dict object storing existing capacities per node for a given technology.
    """

    path_legacy_data = join(dirname(abspath(__file__)), '../../data/legacy')

    if tech in ["wind_onshore", "wind_offshore"]:  # (tech.split('_')[0] == 'wind')&(tech.split('_')[1] != 'floating'):

        data = read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127.xls'), sheet_name='Windfarms',
                          header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
        data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
        data = data[data['Status'] != 'Dismantled']
        data = data[data['ISO code'].isin(region_subdivisions)]

        # Keep only onshore or offshore point depending on technology
        if tech == 'wind_onshore':
            capacity_threshold = 0.2  # TODO: ask David -> to avoid to many points
            data = data[data['Area'] != 'Offshore']

        else:  # wind_offhsore
            capacity_threshold = 0.5
            data = data[data['Area'] == 'Offshore']

        asset_coordinates = array(list(zip(data['Longitude'], data['Latitude'])))

        coordinates = np.array(coordinates)
        node_list = \
            [(x[0], x[1]) for x in
             coordinates[np.argmin(scipy.spatial.distance.cdist(coordinates, asset_coordinates, 'euclidean'), axis=0)]]

        data['Node'] = node_list
        aggregate_capacity_per_node = data.groupby(['Node'])['Total power'].agg('sum')
        aggregate_capacity_per_node = aggregate_capacity_per_node * 1e-6

        output_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    elif tech == "solar_utility":  # (tech.split('_')[0] == 'solar') & (tech.split('_')[1] == 'utility'):

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

        data = data[data['ISO code'].isin(region_subdivisions)]

        asset_coordinates = array(list(zip(data['Longitude'], data['Latitude'])))

        coordinates = np.array(coordinates)
        node_list = \
            [(x[0], x[1]) for x in
             coordinates[np.argmin(scipy.spatial.distance.cdist(coordinates, asset_coordinates, 'euclidean'), axis=0)]]

        data['Node'] = node_list
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')
        aggregate_capacity_per_node = aggregate_capacity_per_node * 1e-3

        capacity_threshold = 0.05 # TODO: parametrize?
        output_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    else:
        raise ValueError(' Technology {} is not supposed to be used with this function.'.format(tech))

    return output_dict


# TODO:
#  - I suspect that we could get rid off this function by merging it with other ones
def retrieve_nodes_with_legacy_units(locations, offshore_onshore_shapes, tech):
    """
    Returns list of nodes where capacity exists.

    Parameters
    ----------
    input_dict : dict
        Dict object storing existing capacities per node for a given technology.
    offshore_onshore_shapes : Dict[str, Polygon or MultiPolygon]
        Onshore and offshore region shape
    tech : str
        Technology.

    Returns
    -------
    existing_locations_filtered : array
        Array populated with coordinate tuples where capacity exists, for a given region and technology.

    """

    # TODO: should directly pass the locations
    # TODO: should add as argument of this function if we want just onshore or offshore, need to change that

    if tech in ['wind_offshore', 'wind_floating']:
        region_shape = offshore_onshore_shapes['offshore']
    else:
        region_shape = offshore_onshore_shapes['onshore']

    if len(locations) > 1:
        locations = [(point.x, point.y) for point in MultiPoint(locations).intersection(region_shape)]
    else:
        locations = [(locations[0][0], locations[0][1])] \
            if Point(locations[0][0], locations[0][1]).within(region_shape) else []

    return locations

# TODO:
#  - data.existing_cap
#  - update comments
def get_legacy_capacity(start_coordinates, regions, technologies, spatial_resolution):
    """
    Returns, for each point, the technical potential share covered by existing (legacy) capacity.

    Parameters
    ----------
    input_dict : dict
        Dict object used to retrieve the (region, tech) keys.

    Returns
    -------
    output_dict : dict
        Dict object storing information on the relative shares of the technical potential of
        each technology and in each region, covered by existing capacity.

    """

    ex_cap_dict = dict.fromkeys(technologies)

    # Get shape of region and list of subregions
    region_subdivisions = []
    for region in regions:
        region_subdivisions += get_subregions_list(region)

    for tech in technologies:

        # Filter coordinates to obtain only the ones on land or offshore
        land_filtered_coordinates = filter_onshore_offshore_locations(start_coordinates, spatial_resolution, tech)
        if tech in ["wind_onshore", "wind_offshore", "solar_utility"]:
            # Get legacy capacity at points in land_filtered_coordinates where legacy capacity exists
            #  and save it in a dict
            ex_cap_dict[tech] = read_legacy_capacity_data(land_filtered_coordinates, region_subdivisions, tech)

    return ex_cap_dict

# TODO:
#  - data.existing_cap
#  - update comments
def retrieve_capacity_share_legacy_units(max_cap_potential, init_coordinate_dict, all_coordinates, spatial_resolution):
    """
    Returns, for each point, the technical potential share covered by existing (legacy) capacity.

    Parameters
    ----------
    input_dict : dict
        Dict object used to retrieve the (region, tech) keys.

    Returns
    -------
    output_dict : dict
        Dict object storing information on the relative shares of the technical potential of
        each technology and in each region, covered by existing capacity.

    """
    key_list = return_dict_keys(max_cap_potential)

    output_dict = deepcopy(max_cap_potential)

    for region, tech in key_list:

        # Get shape of region and list of subregions
        # TODO: as this only depend on region, maybe we should do a double loop
        region_shapefile_data = return_region_shapefile(region)
        # TODO: change return coordinates from shapefiles so that it doesn't take all_coordinates as argument
        start_coordinates = return_coordinates_from_shapefiles(all_coordinates, region_shapefile_data['region_shapefiles'])
        region_subdivisions = region_shapefile_data['region_subdivisions']

        # Filter coordinates to obtain only the ones on land or offshore
        land_filtered_coordinates = filter_onshore_offshore_locations(start_coordinates, spatial_resolution, tech)

        init_coords = init_coordinate_dict[region][tech]
        dict_developed_potential = dict.fromkeys(init_coords, 0.)
        if tech in ["wind_onshore", "wind_offshore", "solar_utility"]:

            # Get legacy capacity at points in land_filtered_coordinates where legacy capacity exists
            #  and save it in a dict
            legacy_dict = read_legacy_capacity_data(land_filtered_coordinates, region_subdivisions, tech)

            for coord in init_coords:

                if coord in legacy_dict.keys():
                    dict_developed_potential[coord] = legacy_dict[coord]

        dict_to_xarray = {'coords': {'locations': {'dims': 'locations',
                                                   'data': asarray(init_coords,
                                                                   dtype='f4,f4')}},
                          'dims': 'locations',
                          'data': asarray(list(dict_developed_potential.values()), dtype=float)}

        developed_potential_array = xr.DataArray.from_dict(dict_to_xarray)
        # Compute the percentage of capacity potential the existing capacity corresponds to
        potential_array_as_share = developed_potential_array.__truediv__(max_cap_potential[region][tech])
        output_dict[region][tech] = potential_array_as_share.where(xu.isfinite(potential_array_as_share), other=0.)

    return output_dict


# TODO:
#  - It's actually probably smarter than using shapes to differentiate between onshore and offshore
#  - I think we should actually change the tech argument to an onshore or offshore argument
def filter_onshore_offshore_locations(coordinates, spatial_resolution: float, tech: str):
    """
    Filters coordinates to leave only onshore and offshore coordinates depending on technology

    Parameters
    ----------
    coordinates : List[tuple(float, float)]
    spatial_resolution : float
        Spatial resolution of coordinates # TODO: do we need to pass this as an argument?
    tech : str
        Technology in ['wind_onshore', 'wind_offshore', 'wind_floating', 'solar_utility', 'solar_residential']

    Returns
    -------
    coordinates : List[tuple(float, float)]
        Coordinates filtered via land/water mask.
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'solar_utility', 'solar_residential']
    assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    path_land_data = '../../data/land_data/ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
    dataset = xr.open_dataset(path_land_data)
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    array_watermask = dataset['lsm']

    if tech in ['wind_onshore', 'solar_utility', 'solar_residential']:

        mask_watermask = array_watermask.where(array_watermask.data >= 0.3)

    elif tech in ['wind_offshore', 'wind_floating']:

        mask_watermask = array_watermask.where(array_watermask.data < 0.3)

    else:
        raise ValueError(' This technology does not exist.')

    coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

    return list(set(coordinates).intersection(set(coords_mask_watermask)))


# TODO:
#  - would it not be possible to merge this with previous computations?
#  - data.existing_cap ?
def update_potential_per_node(potential_per_node_dict, deployment_shares_dict):
    """
    Updates the technical potential per node, for nodes where existing capacity
    goes beyond the calculated potential via the ENSPRESSO dataset.

    Parameters
    ----------
    potential_per_node_dict : dict
        Dict object storing (for each region and technology) the per node potential,
        as computed from the ENSPRESSO dataset.
    deployment_shares_dict : dict
        Dict object storing (for each region and technology) the shares of technical potential
        covered by existing capacities.

    Returns
    -------
    potential_per_node_dict: Dict[str, Dict[str, xr.DataArray]
        Updated potential per node for each region and each tech
    , Dict[str, xr.DataArray]
        Updated deployment shares per node for each tech
    """

    key_list = return_dict_keys(potential_per_node_dict)
    for region, tech in key_list:

        deployment_shares = deployment_shares_dict[region][tech]
        potential_per_node = potential_per_node_dict[region][tech]

        # For 'nodes' where existing capacity is greater than max potential, increase max potential to reach
        # existing capacity
        # TODO: it is a bit strange to already have a share here rather than the installed capacity no?
        for p in deployment_shares[deployment_shares > 1.].locations.values:

            potential_per_node.loc[dict(locations=p)].values *= deployment_shares.sel(locations=p).values

        # Set to 1. deployment share for nodes where potential is now equal to existing cap
        deployment_shares_dict[region][tech] = deployment_shares.clip(max=1.0)

    # TODO: does it make sense to do this collapsing here?
    return potential_per_node_dict, collapse_dict_region_level(deployment_shares_dict)
