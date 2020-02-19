from src.helpers import get_onshore_shapes, get_offshore_shapes, \
                        chunk_split, update_potential_files, collapse_dict_region_level, \
                        match_point_to_region, read_inputs, return_ISO_codes_from_countries, return_dict_keys
from numpy import arange, interp, float32, datetime64, sqrt, \
                  asarray, clip, array, sum, \
                  max, hstack, dtype, unique, radians, arctan2, sin, cos
import xarray as xr
import xarray.ufuncs as xu
from glob import glob
import dask.array as da
from os import listdir
from os.path import join, isfile
from geopandas import read_file
from shapely import prepared
from shapely.geometry import Point
from shapely.ops import unary_union
from pandas import read_csv, read_excel, Series, notnull, to_datetime
from scipy.spatial import distance
from ast import literal_eval
from windpowerlib import power_curves, wind_speed
from collections import Counter
from copy import deepcopy


# TODO:
#  - resite.dataset
#  - need to more precise on description of function, and name it more specifically
def read_database(file_path):
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
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
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
        file_list = [file for file in glob(file_path + '/*.nc') if area in file]
        ds = xr.open_mfdataset(file_list,
                               combine='by_coords',
                               chunks={'latitude': 20, 'longitude': 20})\
                        .stack(locations=('longitude', 'latitude')).astype(float32)
        datasets.append(ds)

    # Concatenate all regions on locations.
    dataset = xr.concat(datasets, dim='locations')
    # Removing duplicates potentially there from previous concat of multiple regions.
    _, index = unique(dataset['locations'], return_index=True)
    dataset = dataset.isel(locations=index)
    # dataset = dataset.sel(locations=~dataset.indexes['locations'].duplicated(keep='first'))
    # Sorting dataset on coordinates (mainly due to xarray peculiarities between concat and merge).
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    # Remove attributes from datasets. No particular use, looks cleaner.
    dataset.attrs = {}

    return dataset


# TODO:
#  - why shapefile? just shape?
#  - data.geographics
def return_region_shapefile(region, path_shapefile_data):
    """
    Returns shapefile associated with the region(s) of interest.

    Parameters
    ----------
    region : str

    path_shapefile_data : str

    Returns
    -------
    output_dict : dict
        Dict object containing i) region subdivisions (if the case) and
        ii) associated onshore and offshore shapes.

    """

    # Load countries/regions shapes
    onshore_shapes_all = read_file(join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson'))

    if region == 'EU':
        region_subdivisions = ['AT', 'BE', 'DE', 'DK', 'ES', 'BA',
                            'FR', 'UK', 'IE', 'IT', 'LU',
                            'NL', 'NO', 'PT', 'SE', 'CH', 'CZ',
                            'AL', 'BG', 'EE', 'LV', 'ME',
                            'FI', 'EL', 'HR', 'HU', 'LT',
                            'MK', 'PL', 'RO', 'RS', 'SI', 'SK']
    elif region == 'NA':
        region_subdivisions = ['DZ', 'EG', 'MA', 'LY', 'TN']
    elif region == 'ME':
        region_subdivisions = ['AE', 'BH', 'CY', 'IR', 'IQ', 'IL', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'SY', 'YE']
    elif region == 'US_S':
        region_subdivisions = ['US-TX']
    elif region == 'US_W':
        region_subdivisions = ['US-AZ', 'US-CA', 'US-CO', 'US-MT', 'US-WY', 'US-NM',
                               'US-UT', 'US-ID', 'US-WA', 'US-NV', 'US-OR']
    elif region == 'US_E':
        region_subdivisions = ['US-ND', 'US-SD', 'US-NE', 'US-KS', 'US-OK', 'US-MN',
                               'US-IA', 'US-MO', 'US-AR', 'US-LA', 'US-MS', 'US-AL', 'US-TN',
                               'US-IL', 'US-WI', 'US-MI', 'US-IN', 'US-OH', 'US-KY', 'US-GA', 'US-FL',
                               'US-PA', 'US-SC', 'US-NC', 'US-VA', 'US-WV',
                               'US-MD', 'US-DE', 'US-NJ', 'US-NY', 'US-CT', 'US-RI',
                               'US-MA', 'US-VT', 'US-ME', 'US-NH']
    elif region in onshore_shapes_all['CNTR_CODE'].values:
        region_subdivisions = [region]
    else:
        raise ValueError(' Unknown region ', region)

    onshore_shapes_selected = get_onshore_shapes(region_subdivisions, path_shapefile_data)
    offshore_shapes_selected = get_offshore_shapes(region_subdivisions, onshore_shapes_selected, path_shapefile_data)

    onshore = hstack((onshore_shapes_selected["geometry"].values))
    offshore = hstack((offshore_shapes_selected["geometry"].values))

    onshore_union = unary_union(onshore)
    offshore_union = unary_union(offshore)

    onshore_prepared = prepared.prep(onshore_union)
    offshore_prepared = prepared.prep(offshore_union)

    output_dict = {'region_subdivisions': region_subdivisions,
                   'region_shapefiles': {'onshore': onshore_prepared,
                                         'offshore': offshore_prepared}}

    return output_dict


# TODO:
#  - can probably improve running time using shapely multipoint intersection
#  - data.geographics
def return_coordinates_from_shapefiles(resource_dataset, shapefiles_region):
    """
    Returning coordinate (lon, lat) pairs falling into the region(s) of interest.

    Parameters
    ----------
    resource_dataset : xarray.Dataset
        Resource dataset.
    shapefiles_region : dict
        Dict object containing the onshore and offshore shapefiles.

    Returns
    -------
    coordinates_in_region : list
        List of coordinate pairs in the region of interest.

    """
    start_coordinates = list(zip(resource_dataset.longitude.values, resource_dataset.latitude.values))

    coordinates_in_region_onshore = array(start_coordinates, dtype('float,float'))[
                                [shapefiles_region['onshore'].contains(Point(p)) for p in start_coordinates]].tolist()

    coordinates_in_region_offshore = array(start_coordinates, dtype('float,float'))[
        [shapefiles_region['offshore'].contains(Point(p)) for p in start_coordinates]].tolist()

    coordinates_in_region = list(set(coordinates_in_region_onshore).union(set(coordinates_in_region_offshore)))

    return coordinates_in_region


# TODO:
#  - this function is pretty fat no? divide it?
#  - data.geographics
def filter_locations_by_layer(tech_dict,
                              start_coordinates, spatial_resolution,
                              path_land_data, path_resource_data, path_population_data,
                              which='dummy', filename='dummy'):
    """
    Filters (removes) locations from the initial set following various
    land-, resource-, populatio-based criteria.

    Parameters
    ----------
    tech_dict : dict
        Dict object containing technical parameters and constraints of a given technology.
    start_coordinates : list
        List of initial (starting) coordinates.
    spatial_resolution : float
        Spatial resolution of the resource data.
    path_land_data : str
        Relative path to land data.
    path_resource_data : str
        Relative path to resource data.
    path_population_data : str
        Relative path to population density data.
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

        coords_to_remove = []

        R = 6371.

        dataset = read_file(join(path_land_data, filename))

        lons = []
        lats = []

        # Retrieve the geopandas Point objects and their coordinates
        for item in protected_areas_selection:
            for index, row in dataset.iterrows():
                if (row['IUCN_CAT'] == item):
                    lons.append(round(row.geometry[0].x, 2))
                    lats.append(round(row.geometry[0].y, 2))

        protected_coords = list(zip(lons, lats))

        # Compute distance between reference coordinates and Points
        for i in start_coordinates:
            for j in protected_coords:
                lat1 = radians(i[1])
                lon1 = radians(i[0])
                lat2 = radians(j[1])
                lon2 = radians(j[0])

                dlon = lon2 - lon1
                dlat = lat2 - lat1

                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * arctan2(sqrt(a), sqrt(1 - a))

                distance = R * c

                if distance < threshold_distance:
                    coords_to_remove.append(i)

    elif which == 'resource':

        database = read_database(path_resource_data)

        if tech_dict['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 +
                                    database.v100 ** 2)
        elif tech_dict['resource'] == 'solar':
            array_resource = database.ssrd / 3600.

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < tech_dict['resource_threshold'])
        coords_mask_resource = mask_resource[mask_resource.notnull()].locations.values.tolist()
        coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_resource)))

    elif which in ['orography', 'forestry', 'water_mask', 'bathymetry']:

        dataset = xr.open_dataset(join(path_land_data, filename))
        dataset = dataset.sortby([dataset.longitude, dataset.latitude])

        dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                     + 180) % 360) - 180)).sortby('longitude')
        dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))

        if which == 'orography':

            altitude_threshold = tech_dict['altitude_threshold']
            slope_threshold = tech_dict['terrain_slope_threshold']

            array_altitude = dataset['z'] / 9.80665
            array_slope = dataset['slor']

            mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
            mask_slope = array_slope.where(array_slope.data > slope_threshold)

            coords_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()
            coords_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

            coords_mask_orography = list(set(coords_mask_altitude).union(set(coords_mask_slope)))
            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_orography)))

        elif which == 'forestry':

            forestry_threshold = tech_dict['forestry_ratio_threshold']

            array_forestry = dataset['cvh']

            mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
            coords_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_forestry)))

        elif which == 'water_mask':

            array_watermask = dataset['lsm']

            mask_watermask = array_watermask.where(array_watermask.data < 0.9)
            coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_watermask)))

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

            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_offshore)))

    elif which == 'population':

        dataset = xr.open_dataset(join(path_population_data, filename))

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

        coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_population)))

    else:

        raise ValueError(' Layer {} is not available.'.format(str(which)))

    return coords_to_remove


# TODO
#  - need to update comment, rename just 'filter_coordinates'?
#  - data.geographics
def return_filtered_coordinates(dataset, spatial_resolution, technologies, regions,
                                path_land_data, path_resource_data, path_legacy_data, path_shapefile_data,
                                path_population_data, resource_quality_layer=True, population_density_layer=True,
                                protected_areas_layer=False, orography_layer=True, forestry_layer=True,
                                water_mask_layer=True, bathymetry_layer=True, legacy_layer=True):
    """
    Returns the set of potential deployment locations for each region and available technology.

    Parameters
    ----------
    coordinates_in_region : list
        List of coordinate pairs in the region of interest.
    spatial_resolution : float
        Spatial resolution of the resource data.
    technologies : list
        List of available technologies.
    regions : list
        List of regions.
    path_land_data : str

    path_resource_data : str

    path_legacy_data : str
        Relative path to existing capacities (for wind and solar PV) data.
    path_shapefile_data : str

    path_population_data : str

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
    tech_config = read_inputs('../config_techs.yml')
    output_dict = {region: {tech: None for tech in technologies} for region in regions}

    for region in regions:

        for tech in technologies:

            tech_dict = tech_config[tech]
            region_shapefile_data = return_region_shapefile(region, path_shapefile_data)
            start_coordinates = return_coordinates_from_shapefiles(dataset, region_shapefile_data['region_shapefiles'])

            # TODO: I would rearrange this and filter_locations_by_layer
            if resource_quality_layer:

                coords_to_remove_resource = \
                    filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                path_land_data, path_resource_data, path_population_data,
                                                which='resource')
            else:
                coords_to_remove_resource = []

            if tech_dict['deployment'] in ['onshore', 'utility', 'residential']:

                if population_density_layer:
                    filename = 'gpw_v4_population_density_rev11_'+str(spatial_resolution)+'.nc'
                    coords_to_remove_population = \
                        filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                    path_land_data, path_resource_data, path_population_data,
                                                    which='population', filename=filename)

                else:
                    coords_to_remove_population = []

                if protected_areas_layer:
                    filename = 'WDPA_Feb2019-shapefile-points.shp'
                    coords_to_remove_protected_areas = \
                        filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                    path_land_data, path_resource_data, path_population_data,
                                                    which='protected_areas', filename=filename)
                else:
                    coords_to_remove_protected_areas = []

                if orography_layer:
                    filename = 'ERA5_orography_characteristics_20181231_'+str(spatial_resolution)+'.nc'
                    coords_to_remove_orography = \
                        filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                    path_land_data, path_resource_data, path_population_data,
                                                    which='orography', filename=filename)
                else:
                    coords_to_remove_orography = []


                if forestry_layer:
                    filename = 'ERA5_surface_characteristics_20181231_'+str(spatial_resolution)+'.nc'
                    coords_to_remove_forestry = \
                        filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                    path_land_data, path_resource_data, path_population_data,
                                                    which='forestry', filename=filename)
                else:
                    coords_to_remove_forestry = []

                if water_mask_layer:
                    filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
                    coords_to_remove_water = \
                        filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                    path_land_data, path_resource_data, path_population_data,
                                                    which='water_mask', filename=filename)
                else:
                    coords_to_remove_water = []

                # TODO: probably be to do this cleaner
                list_coords_to_remove = [coords_to_remove_resource,
                                         coords_to_remove_population,
                                         coords_to_remove_protected_areas,
                                         coords_to_remove_orography,
                                         coords_to_remove_forestry,
                                         coords_to_remove_water]
                coords_to_remove = set().union(*list_coords_to_remove)
                # Set difference between "global" coordinates and the sets computed in this function.
                updated_coordinates = set(start_coordinates) - coords_to_remove

            elif tech_dict['deployment'] in ['offshore', 'floating']:

                if bathymetry_layer:
                    filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
                    coords_to_remove_bathymetry = \
                        filter_locations_by_layer(tech_dict, start_coordinates, spatial_resolution,
                                                    path_land_data, path_resource_data, path_population_data,
                                                    which='bathymetry', filename=filename)
                else:
                    coords_to_remove_bathymetry = []

                list_coords_to_remove = [coords_to_remove_resource,
                                            coords_to_remove_bathymetry]
                coords_to_remove = set().union(*list_coords_to_remove)
                updated_coordinates = set(start_coordinates) - coords_to_remove

            if legacy_layer:

                land_filtered_coordinates = filter_onshore_offshore_locations(start_coordinates,
                                                                              spatial_resolution, tech)
                legacy_dict = read_legacy_capacity_data(land_filtered_coordinates,
                                                        region_shapefile_data['region_subdivisions'],
                                                        tech, path_legacy_data)
                coords_to_add_legacy = retrieve_nodes_with_legacy_units(legacy_dict, region, tech, path_shapefile_data)

                final_coordinates = set(updated_coordinates).union(set(coords_to_add_legacy))

            else:

                final_coordinates = updated_coordinates

            output_dict[region][tech] = list(final_coordinates)

    for key, value in output_dict.items():

        output_dict[key] = {k: v for k, v in output_dict[key].items() if len(v) > 0}

    return output_dict


# TODO:
#  - i would change the name: truncate_dataset?
#  - resite.dataset
def selected_data(dataset, input_dict, time_slice):
    """
    Slices xarray.Dataset based on relevant i) time horizon and ii) location sets.

    Parameters
    ----------
    dataset : xarray.Dataset
        Complete resource dataset.
    input_dict : dict
        Dict object storing location sets per region and technology.
    time_slice : list
        List containing start and end timestamps for the study.

    Returns
    -------
    output_dict : dict
        Dict object storing sliced data per region and technology.

    """
    key_list = return_dict_keys(input_dict)

    output_dict = deepcopy(input_dict)

    datetime_start = datetime64(time_slice[0])
    datetime_end = datetime64(time_slice[1])

    # This is a test which raised some problems on Linux distros, where for some
    # unknown reason the dataset is not sorted on the time dimension.
    if (datetime_start < dataset.time.values[0]) or \
            (datetime_end > dataset.time.values[-1]):
        raise ValueError(' At least one of the time indices exceeds the available data.')

    for region, tech in key_list:

        dataset_temp = []

        for chunk in chunk_split(input_dict[region][tech], n=50):

            dataset_region = dataset.sel(locations=chunk,
                                            time=slice(datetime_start, datetime_end))
            dataset_temp.append(dataset_region)

        output_dict[region][tech] = xr.concat(dataset_temp, dim='locations')

    return output_dict


# TODO:
#  - change name : compute_capacity_factors?
#  - replace using atlite?
#  - data. ?
def return_output(input_dict, path_to_transfer_function, smooth_wind_power_curve=True):
    """
    Applies transfer function to raw resource data.

    Parameters
    ----------
    input_dict : dict
        Dict object storing raw resource data.
    path_to_transfer_function : str
        Relative path to transfer function data.
    smooth_wind_power_curve : boolean
        If "True", the transfer function of wind assets replicates the one of a wind farm,
        rather than one of a wind turbine.


    Returns
    -------
    output_dict : dict
        Dict object storing capacity factors per region and technology.

    """
    key_list = return_dict_keys(input_dict)

    output_dict = deepcopy(input_dict)
    tech_dict = read_inputs('../config_techs.yml')

    data_converter_wind = read_csv(join(path_to_transfer_function, 'data_wind_turbines.csv'), sep=';', index_col=0)
    data_converter_solar = read_csv(join(path_to_transfer_function, 'data_solar_modules.csv'), sep=';', index_col=0)

    for region, tech in key_list:

        resource = tech.split('_')[0]
        converter = tech_dict[tech]['converter']

        if resource == 'wind':

            ###

            wind_speed_height = 100.
            array_roughness = input_dict[region][tech].fsr

            # Compute the resultant of the two wind components.
            wind = xu.sqrt(input_dict[region][tech].u100 ** 2 +
                           input_dict[region][tech].v100 ** 2)

            ti = wind.std(dim='time')/wind.mean(dim='time')

            wind_log = wind_speed.logarithmic_profile(wind.values,
                                                      wind_speed_height,
                                                      float(data_converter_wind.loc['Hub height [m]', converter]),
                                                      array_roughness.values)
            wind_data = da.from_array(wind_log, chunks='auto', asarray=True)

            coordinates = input_dict[region][tech].u100.coords
            dimensions = input_dict[region][tech].u100.dims

            power_curve_array = literal_eval(data_converter_wind.loc['Power curve', converter])

            wind_speed_references = asarray([i[0] for i in power_curve_array])
            capacity_factor_references = asarray([i[1] for i in power_curve_array])
            capacity_factor_references_pu = capacity_factor_references / max(capacity_factor_references)

            if smooth_wind_power_curve:
                # Windpowerlib function here:
                # https://windpowerlib.readthedocs.io/en/latest/temp/windpowerlib.power_curves.smooth_power_curve.html
                capacity_factor_farm = \
                    power_curves.smooth_power_curve(Series(wind_speed_references),
                                                    Series(capacity_factor_references_pu),
                                                    standard_deviation_method='turbulence_intensity',
                                                    turbulence_intensity=float(ti.min().values),
                                                    wind_speed_range=10.0)

                power_output = da.map_blocks(interp, wind_data,
                                             capacity_factor_farm['wind_speed'].values,
                                             capacity_factor_farm['value'].values).compute()

            else:

                power_output = da.map_blocks(interp, wind_data,
                                             wind_speed_references,
                                             capacity_factor_references_pu).compute()

        elif resource == 'solar':

            # Get irradiance in W from J
            irradiance = input_dict[region][tech].ssrd / 3600.
            # Get temperature in C from K
            temperature = input_dict[region][tech].t2m - 273.15

            coordinates = input_dict[region][tech].ssrd.coords
            dimensions = input_dict[region][tech].ssrd.dims

            # Homer equation here:
            # https://www.homerenergy.com/products/pro/docs/latest/how_homer_calculates_the_pv_array_power_output.html
            # https://enphase.com/sites/default/files/Enphase_PVWatts_Derate_Guide_ModSolar_06-2014.pdf
            power_output = (float(data_converter_solar.loc['f', converter]) *
                            (irradiance/float(data_converter_solar.loc['G_ref', converter])) *
                            (1. + float(data_converter_solar.loc['k_P [%/C]', converter])/100. *
                             (temperature - float(data_converter_solar.loc['t_ref', converter]))))

        else:
            raise ValueError(' The resource specified is not available yet.')

        output_array = xr.DataArray(power_output, coords=coordinates, dims=dimensions)
        output_array = output_array.where(output_array > 0.01, other=0.0)

        output_dict[region][tech] = output_array

    return output_dict


# TODO:
#   - data.load
#   - need to complete comments
def retrieve_load_data(regions, time_slice, path_load_data, path_shapefile_data):
    """
    Returns load time series for given regions and time horizon.

    Parameters
    ----------

    Returns
    -------

    """
    output_dict = dict.fromkeys(regions)

    load_data = read_csv(join(path_load_data, 'load_opsd_2015_2018.csv'), index_col=0, sep=';')
    load_data.index = to_datetime(load_data.index)

    for region in regions:

        region_unit_list = return_region_shapefile(region, path_shapefile_data)['region_subdivisions']

        load_data_sliced = load_data.loc[time_slice[0]:time_slice[1]][region_unit_list]

        output_dict[region] = load_data_sliced.sum(axis=1).values * (1e-3)

    return output_dict


# TODO:
#  - data.res_potential or  data.cap_potential
#  - merge with my code
def capacity_potential_from_enspresso(tech, path_potential_data):
    """
    Returning capacity potential per NUTS2 region, based on the ENSPRESSO dataset.

    Parameters
    ----------
    technologies : list
        List of available technologies.
    path_potential_data : str
        Relative path to technical potential data.

    Returns
    -------
    output_dict : dict
        Dict storing technical potential per NUTS2 region and technology.
    """

    if tech == 'wind_onshore':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        onshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Onshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide high restrictions')]
        output_dict = onshore_wind.groupby(onshore_wind.index)['Value'].sum().to_dict()

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
        output_dict = offshore_wind.groupby(offshore_wind.index)['Value'].sum().to_dict()

    elif tech == 'wind_floating':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                        sheet_name='Wind Potential EU28 Full', index_col=1)

        offshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Offshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide low restrictions') &
            (cap_potential_file['Wind condition'] == 'CF > 25%') &
            (cap_potential_file['Offshore categories'] == 'Water depth 100-1000m Floating')]
        output_dict = offshore_wind.groupby(offshore_wind.index)['Value'].sum().to_dict()

    elif tech == 'solar_utility':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                        sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)

        output_dict = cap_potential_file['PV - ground'].to_dict()

    elif tech == 'solar_residential':

        cap_potential_file = read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                        sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)

        output_dict = cap_potential_file['PV - roof/facades'].to_dict()

    output_dict_updated = update_potential_files(output_dict, tech)

    return output_dict_updated


# TODO:
#  - data.res_potential or  data.cap_potential
def capacity_potential_per_node(input_dict, spatial_resolution,
                                path_potential_data, path_shapefile_data, path_population_data):
    """
    Assigning a technical potential to each individual node.

    Parameters
    ----------
    input_dict : dict
        Dict object containing coordinates.
    coordinates_in_region : list
        List containing coordinates in region of interest.
    spatial_resolution : float
        Spatial resolution of the study.
    path_potential_data : str
    path_shapefile_data : str
    path_legacy_data : str
    legacy_layer : boolean

    Returns
    -------
    output_dict : dict
        Dict object storing technical potential for each point per region and technology.

    """
    key_list = return_dict_keys(input_dict)

    shapefile_data_onshore = read_file(join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_2_incl_BA.geojson'))
    shapefile_data_offshore = read_file(join(path_shapefile_data, 'EEZ_RG_01M_2016_4326_LEVL_0.geojson'))

    dataset_population = xr.open_dataset(join(path_population_data,
                                              'gpw_v4_population_density_rev11_'+str(spatial_resolution)+'.nc'))

    varname = [item for item in dataset_population.data_vars][0]
    dataset_population = dataset_population.rename({varname: 'data'})
    # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
    data_pop = dataset_population.sel(raster=5)

    array_pop_density = data_pop['data'].interp(longitude=arange(-180, 180, float(spatial_resolution)),
                                                latitude=arange(-89, 91, float(spatial_resolution))[::-1],
                                                method='linear').fillna(0.)
    array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude'))

    output_dict = deepcopy(input_dict)

    for region, tech in key_list:

        region_subdivisions = return_region_shapefile(region, path_shapefile_data)['region_subdivisions']

        if tech in ['wind_offshore', 'wind_floating']:

            filter_shape_data = shapefile_data_offshore[shapefile_data_offshore['ISO_ID'].str.contains(
                                                                                    '|'.join(region_subdivisions))]
            filter_shape_data.index = filter_shape_data['ISO_ID']
            filter_shape_data.index = 'EZ'+filter_shape_data.index

        else:

            filter_shape_data = shapefile_data_onshore[(shapefile_data_onshore['NUTS_ID'].str.contains(
                                                                                    '|'.join(region_subdivisions)))]
            filter_shape_data.index = filter_shape_data['NUTS_ID']

        potential_per_tech = capacity_potential_from_enspresso(tech, path_potential_data)

        dict_subregion = {coord: None for coord in input_dict[region][tech]}
        dict_potential = deepcopy(dict_subregion)

        for c in input_dict[region][tech]:

            dict_subregion[c] = match_point_to_region(c, filter_shape_data, potential_per_tech)

        if tech in ['wind_offshore', 'wind_floating']:

            region_freq = dict(Counter(dict_subregion.values()).most_common())

            for c in input_dict[region][tech]:
                dict_potential[c] = float(potential_per_tech[dict_subregion[c]]) / \
                                    float(region_freq.get(dict_subregion[c]))

        elif tech in ['wind_onshore', 'solar_utility']:

            for c in input_dict[region][tech]:

                incumbent_loc_pop_dens = float(array_pop_density.sel(locations=c).values)
                loc_in_region = [key for key, value in dict_subregion.items() if value == dict_subregion[c]]
                region_loc_pop_dens = clip(array_pop_density.sel(locations=loc_in_region).values, a_min=1., a_max=None)

                distribution_key = (1/incumbent_loc_pop_dens) / sum(1/region_loc_pop_dens)

                dict_potential[c] = float(potential_per_tech[dict_subregion[c]]) * distribution_key

        else:

            for c in input_dict[region][tech]:

                incumbent_loc_pop_dens = float(array_pop_density.sel(locations=c).values)
                loc_in_region = [key for key, value in dict_subregion.items() if value == dict_subregion[c]]
                region_loc_pop_dens = clip(array_pop_density.sel(locations=loc_in_region).values, a_min=1., a_max=None)

                distribution_key = incumbent_loc_pop_dens / sum(region_loc_pop_dens)

                dict_potential[c] = float(potential_per_tech[dict_subregion[c]]) * distribution_key

        dict_to_xarray = {'coords': {'locations': {'dims': 'locations',
                                                   'data': asarray(list(dict_potential.keys()),
                                                                   dtype='f4,f4')}},
                          'dims': 'locations',
                          'data': asarray(list(dict_potential.values()), dtype=float)}

        output_dict[region][tech] = xr.DataArray.from_dict(dict_to_xarray)

    return output_dict


# TODO:
#  - data.existing_cap
#  - update comment
def read_legacy_capacity_data(start_coordinates, region_subdivisions, tech, path_legacy_data):
    """
    Reads dataset of existing RES units in the given area. Available for EU only.

    Parameters
    ----------
    start_coordinates : list
    tech : str
    path_legacy_data : str

    Returns
    -------
    output_dict : dict
        Dict object storing existing capacities per node for a given technology.
    """
    if (tech.split('_')[0] == 'wind') & (tech.split('_')[1] != 'floating'):

        data = read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127.xls'), sheet_name='Windfarms',
                               header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1])

        data = data[~data['Latitude'].isin(['#ND'])]
        data = data[~data['Longitude'].isin(['#ND'])]
        data = data[~data['Total power'].isin(['#ND'])]
        data = data[data['Status'] != 'Dismantled']
        data = data[data['ISO code'].isin(region_subdivisions)]

        if tech == 'wind_onshore':

            capacity_threshold = 0.2
            data_filtered = data[data['Area'] != 'Offshore'].copy()

        elif tech == 'wind_offshore':

            capacity_threshold = 0.5
            data_filtered = data[data['Area'] == 'Offshore'].copy()

        asset_coordinates = array(list(zip(data_filtered['Longitude'],
                                           data_filtered['Latitude'])))

        node_list = []
        for c in asset_coordinates:
            node_list.append(tuple(start_coordinates[distance.cdist(start_coordinates, [c], 'euclidean').argmin()]))

        data_filtered['Node'] = node_list
        aggregate_capacity_per_node = data_filtered.groupby(['Node'])['Total power'].agg('sum')
        aggregate_capacity_per_node = aggregate_capacity_per_node * (1e-6)

        output_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    elif (tech.split('_')[0] == 'solar') & (tech.split('_')[1] == 'utility'):

        data = read_excel(join(path_legacy_data, 'Solarfarms_Europe_20200208.xlsx'), sheet_name='ProjReg_rpt',
                          header=0, usecols=[0, 3, 4, 5, 8])

        data = data[notnull(data['Coords'])]
        data['Longitude'] = data['Coords'].str.split(',', 1).str[1]
        data['Latitude'] = data['Coords'].str.split(',', 1).str[0]
        data['ISO code'] = data['Country'].map(return_ISO_codes_from_countries())

        data = data[data['ISO code'].isin(region_subdivisions)]

        capacity_threshold = 0.05

        asset_coordinates = array(list(zip(data['Longitude'],
                                           data['Latitude'])))

        node_list = []
        for c in asset_coordinates:
            node_list.append(tuple(start_coordinates[distance.cdist(start_coordinates, [c], 'euclidean').argmin()]))

        data['Node'] = node_list
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')
        aggregate_capacity_per_node = aggregate_capacity_per_node * (1e-3)

        output_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    else:

        output_dict = None

    return output_dict


# TODO:
#  - data.existing_cap
def retrieve_nodes_with_legacy_units(input_dict, region, tech, path_shapefile_data):
    """
    Returns list of nodes where capacity exists.

    Parameters
    ----------
    input_dict : dict
        Dict object storing existing capacities per node for a given technology.
    region : str
        Region.
    tech : str
        Technology.
    path_shapefile_data : str

    Returns
    -------
    existing_locations_filtered : array
        Array populated with coordinate tuples where capacity exists, for a given region and technology.

    """

    if input_dict is None:

        existing_locations_filtered = []

    else:

        existing_locations = list(input_dict.keys())

        if tech in ['wind_offshore', 'wind_floating']:

            region_shapefile = return_region_shapefile(region, path_shapefile_data)['region_shapefiles']['offshore']

        else:

            region_shapefile = return_region_shapefile(region, path_shapefile_data)['region_shapefiles']['onshore']

        existing_locations_filtered = array(existing_locations, dtype('float,float'))[
                                            [region_shapefile.contains(Point(p)) for p in existing_locations]].tolist()

    return existing_locations_filtered


# TODO:
#  - data.existing_cap
#  - update comments
def retrieve_capacity_share_legacy_units(input_dict, init_coordinate_dict, dataset,
                                         spatial_resolution, path_legacy_data, path_shapefile_data):
    """
    Returns, for each point, the technical potential share covered by existing (legacy) capacity.

    Parameters
    ----------
    input_dict : dict
        Dict object used to retrieve the (region, tech) keys.
    path_legacy_data : str

    Returns
    -------
    output_dict : dict
        Dict object storing information on the relative shares of the technical potential of
        each technology and in each region, covered by existing capacity.

    """
    key_list = return_dict_keys(input_dict)

    output_dict = deepcopy(input_dict)

    for region, tech in key_list:

        region_shapefile_data = return_region_shapefile(region, path_shapefile_data)

        start_coordinates = return_coordinates_from_shapefiles(dataset, region_shapefile_data['region_shapefiles'])
        region_subdivisions = region_shapefile_data['region_subdivisions']

        land_filtered_coordinates = filter_onshore_offshore_locations(start_coordinates, spatial_resolution, tech)
        legacy_dict = read_legacy_capacity_data(land_filtered_coordinates, region_subdivisions, tech, path_legacy_data)

        if legacy_dict is None:

            dict_developed_potential = {coord: 0. for coord in init_coordinate_dict[region][tech]}

        else:

            dict_developed_potential = {coord: None for coord in init_coordinate_dict[region][tech]}

            for item in dict_developed_potential.keys():

                if item in legacy_dict.keys():

                    dict_developed_potential[item] = legacy_dict[item]

                else:

                    dict_developed_potential[item] = 0.

        dict_to_xarray = {'coords': {'locations': {'dims': 'locations',
                                                   'data': asarray(list(dict_developed_potential.keys()),
                                                                   dtype='f4,f4')}},
                          'dims': 'locations',
                          'data': asarray(list(dict_developed_potential.values()), dtype=float)}

        developed_potential_array = xr.DataArray.from_dict(dict_to_xarray)
        potential_array_as_share = developed_potential_array.__truediv__(input_dict[region][tech])

        output_dict[region][tech] = potential_array_as_share.where(xu.isfinite(potential_array_as_share), other=0.)

    return output_dict


# TODO:
#  - don't know what to do with this
#  - data.geographics
def filter_onshore_offshore_locations(coordinates_in_region, spatial_resolution, tech):
    """
    Filters on- and offshore coordinates.

    Parameters
    ----------
    coordinates_in_region : list
    spatial_resolution : float
    tech : str

    Returns
    -------
    updated_coordinates : list
        Coordinates filtered via land/water mask.
    """
    # TODO: why do this in three lines?
    filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
    path_land_data = '../input_data/land_data'

    dataset = xr.open_dataset(join(path_land_data, filename))
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])

    dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                 + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    array_watermask = dataset['lsm']

    if tech in ['wind_onshore', 'solar_utility', 'solar_residential']:

        mask_watermask = array_watermask.where(array_watermask.data >= 0.3)

    elif tech in ['wind_offshore', 'wind_floating']:

        mask_watermask = array_watermask.where(array_watermask.data < 0.3)

    else:
        raise ValueError(' This technology does not exist.')

    coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

    updated_coordinates = list(set(coordinates_in_region).intersection(set(coords_mask_watermask)))

    return updated_coordinates


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

    """
    key_list = return_dict_keys(potential_per_node_dict)

    updated_potential_per_node_dict = deepcopy(potential_per_node_dict)
    updated_deployment_shares_per_node_dict = deepcopy(potential_per_node_dict)

    for region, tech in key_list:

        potential_per_node = potential_per_node_dict[region][tech].copy()
        deployment_shares = deployment_shares_dict[region][tech]

        for p in deployment_shares[deployment_shares > 1.].locations.values:

            potential_per_node.loc[dict(locations=p)].values *= deployment_shares.sel(locations=p).values

        updated_potential_per_node_dict[region][tech] = potential_per_node
        updated_deployment_shares_per_node_dict[region][tech] = deployment_shares_dict[region][tech].clip(max=1.0)

    output_dict = {'updated_potential': updated_potential_per_node_dict,
                   'updated_legacy_shares': collapse_dict_region_level(updated_deployment_shares_per_node_dict)}

    return output_dict


# TODO:
#  - resite.dataset
#  - comments - I don't understand what this does
def retrieve_location_dict(input_dict, site_dict):

    output_dict = {key: {} for key in input_dict['technologies']}

    potential_dict = collapse_dict_region_level(input_dict['technical_potential_dict'])

    for tech in site_dict.keys():
        for idx, y_object in site_dict[tech].items():
            if y_object is not None:
                if y_object.value > 0.:

                    loc = input_dict['starting_deployment_dict'][tech].isel(locations=idx).locations.values.flatten()[0]
                    cap = y_object.value * potential_dict[tech].isel(locations=idx).values

                    output_dict[tech][loc] = cap

    output_dict = {k: v for k, v in output_dict.items() if len(v) > 0}

    return output_dict
