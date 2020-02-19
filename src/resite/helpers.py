import os
import glob
import numpy as np
from shapely.geometry import Point
from xarray import concat
from copy import deepcopy
from pandas import DataFrame
import xarray as xr


# TODO: unused
def xarray_to_ndarray(input_dict):
    """
    Converts dict of xarray objects to ndarray to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    ndarray : ndarray

    """
    key_list = []
    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    array_list = ()

    for region, tech in key_list:

        array_list = (*array_list, input_dict[region][tech].values)

    ndarray = np.hstack(array_list)

    return ndarray


# TODO:
#  - resite.dataset
def xarray_to_dict(input_dict, levels):
    """
    Converts dict of xarray objects to dict of ndarrays to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    levels : int
        Depth of (nested) dict. Available values: 1 or 2.

    Returns
    -------
    output_dict : dict

    """
    output_dict = deepcopy(input_dict)

    if levels == 2:

        key_list = return_dict_keys(input_dict)

        for region, tech in key_list:

            output_dict[region][tech] = input_dict[region][tech].values

    else:

        key_list = input_dict.keys()

        for tech in key_list:

            output_dict[tech] = input_dict[tech].values

    return output_dict


# TODO:
#  - resite.dataset
#  - not very clear what this function is meant to
def retrieve_dict_max_length_item(input_dict):
    """
    Retrieve size of largest dict value.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    max_len : int

    """
    key_list = []

    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    max_len = 0

    for region, tech in key_list:

        incumbent_len = len(input_dict[region][tech].locations)

        if incumbent_len > max_len:

            max_len = incumbent_len

    return max_len


# TODO:
#  - resite.dataset
def dict_to_xarray(input_dict):
    """
    Converts dict of xarray objects to xarray DataArray to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    dataset : xr.DataArray

    """

    key_list = return_dict_keys(input_dict)

    array_list = []

    for region, tech in key_list:

        array_list.append(input_dict[region][tech])

    dataset = concat(array_list, dim='locations')

    return dataset


# TODO:
#  - resite.dataset
#  - (need to have a more specific name)
def collapse_dict_region_level(input_dict):
    """
    Converts nested dict (dict[region][tech]) to single-level (dict[tech]).

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    output_dict : dict

    """
    output_dict = {}

    technologies = list(set().union(*input_dict.values()))

    for item in technologies:
        l = []
        for region in input_dict:
            for tech in input_dict[region]:
                if tech == item:
                    l.append(input_dict[region][tech])
        output_dict[item] = concat(l, dim='locations')

    return output_dict


# TODO:
#  - resite.dataset
#  - more specific name or create some sort of class with this kind of dict
def return_dict_keys(input_dict):
    """
    Returns (region, tech) keys of nested dict.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    key_list : list

    """

    key_list = []
    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    return key_list


# TODO: unused
def concatenate_dict_keys(input_dict):
    """
    Converts nested dict (dict[region][tech]) keys into tuples (dict[(region, tech)]).

    Parameters
    ----------

    Returns
    -------
    output_dict : dict

    """

    output_dict = {}

    key_list = return_dict_keys(input_dict)

    for region, tech in key_list:

        output_dict[(region, tech)] = input_dict[region][tech]

    return output_dict


# TODO:
#  - resite.dataset
def retrieve_coordinates_from_dict(input_dict):
    """
    Retrieves coordinate list for each (region, tech) tuple key. Requires dict values to be xarray.DataArray objects!

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    output_dict : dict

    """
    output_dict = {}

    for concat_key in input_dict.keys():

        output_dict[concat_key] = list(input_dict[concat_key].locations.values)

    return output_dict


# TODO:
#  - resite.dataset
def retrieve_tech_coordinates_tuples(input_dict):
    """
    Retrieves list of all (tech, loc) tuples.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    l : list

    """
    l = []

    for key, value in input_dict.items():
        for idx, val in enumerate(value):
            l.append((key, idx))

    return l


# TODO: unused
def retrieve_incidence_matrix(input_dict):
    """
    Computes the (region, tech) vs. (lon, lat) incidence matrix.

    Parameters
    ----------
    input_dict : dict containing xarray.Dataset objects indexed by (region, tech) tuples.

    Returns
    -------
    incidence_matrix : DataFrame

    """
    coord_list = []
    for concat_key in input_dict.keys():
        coord_list.extend(list(input_dict[concat_key].locations.values))

    idx = list(set(coord_list))
    cols = list(input_dict.keys())

    incidence_matrix = DataFrame(0, index=idx, columns=cols)

    for concat_key in input_dict.keys():

        coordinates = input_dict[concat_key].locations.values

        for c in coordinates:

            incidence_matrix.loc[c, concat_key] = 1

    return incidence_matrix


# TODO: unused
def retrieve_location_indices_per_tech(input_dict):
    """
    Retrieves integer indices of locations associated with each technology.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    output_dict : dict

    """
    key_list = []

    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    output_dict = deepcopy(input_dict)

    for region, tech in key_list:

        output_dict[region][tech] = np.arange(len(input_dict[region][tech].locations))

    return output_dict


# TODO: unused
def get_partition_index(input_dict, deployment_vector, capacity_split='per_tech'):
    """
    Returns start and end indices for each (region, technology) tuple. Required in case the problem
    is defined with partitioning constraints.

    Parameters
    ----------
    input_dict : dict
        Dict object storing coordinates per region and tech.
    deployment_vector : list
        List containing the deployment requirements (un-partitioned or not).
    capacity_split : str
        Capacity splitting rule. To choose between "per_tech" and "per_country".

    Returns
    -------
    index_list : list
        List of indices associated with each (region, technology) tuple.

    """
    key_list = return_dict_keys(input_dict)

    init_index_dict = deepcopy(input_dict)

    regions = list(set([i[0] for i in key_list]))
    technologies = list(set([i[1] for i in key_list]))

    start_index = 0
    for region, tech in key_list:
        init_index_dict[region][tech] = list(np.arange(start_index, start_index + len(input_dict[region][tech])))
        start_index = start_index + len(input_dict[region][tech])

    if capacity_split == 'per_country':

        if len(deployment_vector) == len(regions):

            index_dict = dict.fromkeys(regions)
            for region in regions:
                index_list_per_region = []
                tech_list_in_region = [i[1] for i in key_list if i[0] == region]
                for tech in tech_list_in_region:
                    index_list_per_region.extend(init_index_dict[region][tech])
                index_dict[region] = [i+1 for i in index_list_per_region]

        else:

            raise ValueError(' Number of regions ({}) does not match number of deployment constraints ({}).'.format
                                            (len(regions), len(deployment_vector)))

    elif capacity_split == 'per_tech':

        if len(deployment_vector) == len(technologies):

            index_dict = dict.fromkeys(technologies)
            for tech in technologies:
                index_list_per_tech = []
                region_list_with_tech = [i[0] for i in key_list if i[1] == tech]
                for region in region_list_with_tech:
                    index_list_per_tech.extend(init_index_dict[region][tech])
                index_dict[tech] = [i+1 for i in index_list_per_tech]

        else:

            raise ValueError(' Number of technologies ({}) does not match number of deployment constraints ({}).'.format
                             (len(technologies), len(deployment_vector)))

    index_list = []
    for key, value in index_dict.items():
        index_list.append([i+1 for i in value])

    return init_index_dict


# TODO:
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


# TODO:
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


def chunk_split(l, n):
    """
    Splits large lists in smaller chunks. Done to avoid xarray warnings when slicing large datasets.

    Parameters
    ----------
    l : list
    n : chunk size
    """
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


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

    datetime_start = np.datetime64(time_slice[0])
    datetime_end = np.datetime64(time_slice[1])

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
#  - don't really know where this should go
def compute_generation_potential(capacity_factor_dict, potential_dict):
    """
    Computes generation potential (GWh) to be passed to the optimisation problem.

    Parameters
    ----------
    capacity_factor_dict : dict containing capacity factor time series.

    potential_dict : dict containing technical potential figures per location.

    Returns
    -------
    output_dict : dict

    """
    # TODO: this is a pretty heavy operation just to copy the structure no?
    output_dict = deepcopy(capacity_factor_dict)

    for region in capacity_factor_dict:
        for tech in capacity_factor_dict[region]:
            output_dict[region][tech] = capacity_factor_dict[region][tech] * potential_dict[region][tech]

    return output_dict


# TODO:
#  - need to change this - use what I have done in my code or try at least
#  - data.res_potential
def update_potential_files(input_dict, tech):
    """
    Updates NUTS2 potentials with i) non-EU data and ii) re-indexed (2013 vs 2016) NUTS2 regions.

    Parameters
    ----------
    input_dict : dict
        Dict object containing technical potential per NUTS2 region and technology.
    tech : str
        Technology.

    Returns
    -------
    input_dict : dict
        Updated potential dict.

    """
    regions_to_remove = ['AD00', 'SM00', 'CY00', 'LI00', 'FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5', 'ES63', 'ES64', 'ES70']

    for key in regions_to_remove:
        input_dict.pop(key, None)

    if tech in ['wind_onshore', 'solar_residential', 'solar_utility']:

        dict_regions_update = {'FR21': 'FRF2', 'FR22': 'FRE2', 'FR23': 'FRD1', 'FR24': 'FRB0', 'FR25': 'FRD2',
                               'FR26': 'FRC1', 'FR30': 'FRE1', 'FR41': 'FRF3', 'FR42': 'FRF1', 'FR43': 'FRC2',
                               'FR51': 'FRG0', 'FR52': 'FRH0', 'FR53': 'FRI3', 'FR61': 'FRI1', 'FR62': 'FRJ2',
                               'FR63': 'FRI2', 'FR71': 'FRK2', 'FR72': 'FRK1', 'FR81': 'FRJ1', 'FR82': 'FRL0',
                               'FR83': 'FRM0'}

        for key in dict_regions_update.keys():
            input_dict[dict_regions_update[key]] = input_dict.pop(key, dict_regions_update[key])

    if tech == 'wind_onshore':

        input_dict['AL01'] = 2.
        input_dict['AL02'] = 2.
        input_dict['AL03'] = 2.
        input_dict['BA00'] = 3.
        input_dict['ME00'] = 3.
        input_dict['MK00'] = 5.
        input_dict['RS11'] = 0.
        input_dict['RS12'] = 10.
        input_dict['RS21'] = 10.
        input_dict['RS22'] = 10.
        input_dict['CH01'] = 1.
        input_dict['CH02'] = 1.
        input_dict['CH03'] = 1.
        input_dict['CH04'] = 1.
        input_dict['CH05'] = 1.
        input_dict['CH06'] = 1.
        input_dict['CH07'] = 1.
        input_dict['NO01'] = 3.
        input_dict['NO02'] = 3.
        input_dict['NO03'] = 3.
        input_dict['NO04'] = 3.
        input_dict['NO05'] = 3.
        input_dict['NO06'] = 3.
        input_dict['NO07'] = 3.
        input_dict['IE04'] = input_dict['IE01']
        input_dict['IE05'] = input_dict['IE02']
        input_dict['IE06'] = input_dict['IE02']
        input_dict['LT01'] = input_dict['LT00']
        input_dict['LT02'] = input_dict['LT00']
        input_dict['UKM7'] = input_dict['UKM2']
        input_dict['UKM8'] = input_dict['UKM5']
        input_dict['UKM9'] = input_dict['UKM3']
        input_dict['PL71'] = input_dict['PL11']
        input_dict['PL72'] = input_dict['PL33']
        input_dict['PL81'] = input_dict['PL31']
        input_dict['PL82'] = input_dict['PL32']
        input_dict['PL84'] = input_dict['PL34']
        input_dict['PL92'] = input_dict['PL12']
        input_dict['PL91'] = 0.
        input_dict['HU11'] = 0.
        input_dict['HU12'] = input_dict['HU10']
        input_dict['UKI5'] = 0.
        input_dict['UKI6'] = 0.
        input_dict['UKI7'] = 0.

    elif tech == 'wind_offshore':

        input_dict['EZAL'] = 2.
        input_dict['EZBA'] = 0.
        input_dict['EZME'] = 0.
        input_dict['EZMK'] = 0.
        input_dict['EZRS'] = 0.
        input_dict['EZCH'] = 0.
        input_dict['EZNO'] = 20.
        input_dict['EZIE'] = 20.
        input_dict['EZEL'] = input_dict['EZGR']

    elif tech == 'wind_floating':

        input_dict['EZAL'] = 2.
        input_dict['EZBA'] = 0.
        input_dict['EZME'] = 0.
        input_dict['EZMK'] = 0.
        input_dict['EZRS'] = 0.
        input_dict['EZCH'] = 0.
        input_dict['EZNO'] = 100.
        input_dict['EZIE'] = 120.
        input_dict['EZEL'] = input_dict['EZGR']

    elif tech == 'solar_residential':

        input_dict['AL01'] = 1.
        input_dict['AL02'] = 1.
        input_dict['AL03'] = 1.
        input_dict['BA00'] = 3.
        input_dict['ME00'] = 1.
        input_dict['MK00'] = 1.
        input_dict['RS11'] = 5.
        input_dict['RS12'] = 2.
        input_dict['RS21'] = 2.
        input_dict['RS22'] = 2.
        input_dict['CH01'] = 6.
        input_dict['CH02'] = 6.
        input_dict['CH03'] = 6.
        input_dict['CH04'] = 6.
        input_dict['CH05'] = 6.
        input_dict['CH06'] = 6.
        input_dict['CH07'] = 6.
        input_dict['NO01'] = 3.
        input_dict['NO02'] = 0.
        input_dict['NO03'] = 3.
        input_dict['NO04'] = 3.
        input_dict['NO05'] = 0.
        input_dict['NO06'] = 0.
        input_dict['NO07'] = 0.
        input_dict['IE04'] = input_dict['IE01']
        input_dict['IE05'] = input_dict['IE02']
        input_dict['IE06'] = input_dict['IE02']
        input_dict['LT01'] = input_dict['LT00']
        input_dict['LT02'] = input_dict['LT00']
        input_dict['UKM7'] = input_dict['UKM2']
        input_dict['UKM8'] = input_dict['UKM5']
        input_dict['UKM9'] = input_dict['UKM3']
        input_dict['PL71'] = input_dict['PL11']
        input_dict['PL72'] = input_dict['PL33']
        input_dict['PL81'] = input_dict['PL31']
        input_dict['PL82'] = input_dict['PL32']
        input_dict['PL84'] = input_dict['PL34']
        input_dict['PL92'] = input_dict['PL12']
        input_dict['PL91'] = 5.
        input_dict['HU11'] = input_dict['HU10']
        input_dict['HU12'] = input_dict['HU10']
        input_dict['UKI5'] = 1.
        input_dict['UKI6'] = 1.
        input_dict['UKI7'] = 1.

    elif tech == 'solar_utility':

        input_dict['AL01'] = 1.
        input_dict['AL02'] = 1.
        input_dict['AL03'] = 1.
        input_dict['BA00'] = 3.
        input_dict['ME00'] = 1.
        input_dict['MK00'] = 1.
        input_dict['RS11'] = 0.
        input_dict['RS12'] = 2.
        input_dict['RS21'] = 2.
        input_dict['RS22'] = 1.
        input_dict['CH01'] = 6.
        input_dict['CH02'] = 6.
        input_dict['CH03'] = 6.
        input_dict['CH04'] = 6.
        input_dict['CH05'] = 6.
        input_dict['CH06'] = 6.
        input_dict['CH07'] = 6.
        input_dict['NO01'] = 3.
        input_dict['NO02'] = 0.
        input_dict['NO03'] = 3.
        input_dict['NO04'] = 3.
        input_dict['NO05'] = 0.
        input_dict['NO06'] = 0.
        input_dict['NO07'] = 0.
        input_dict['IE04'] = input_dict['IE01']
        input_dict['IE05'] = input_dict['IE02']
        input_dict['IE06'] = input_dict['IE02']
        input_dict['LT01'] = input_dict['LT00']
        input_dict['LT02'] = input_dict['LT00']
        input_dict['UKM7'] = input_dict['UKM2']
        input_dict['UKM8'] = input_dict['UKM5']
        input_dict['UKM9'] = input_dict['UKM3']
        input_dict['PL71'] = input_dict['PL11']
        input_dict['PL72'] = input_dict['PL33']
        input_dict['PL81'] = input_dict['PL31']
        input_dict['PL82'] = input_dict['PL32']
        input_dict['PL84'] = input_dict['PL34']
        input_dict['PL92'] = input_dict['PL12']
        input_dict['PL91'] = 2.
        input_dict['HU11'] = 0.
        input_dict['HU12'] = 2.
        input_dict['UKI5'] = 0.
        input_dict['UKI6'] = 0.
        input_dict['UKI7'] = 0.

    return input_dict


# TODO:
#  - data.geographics?
def match_point_to_region(point, shape_data, indicator_data):
    """
    Assings a given coordinate tuple (lon, lat) to a NUTS (or any other) region.

    Parameters
    ----------
    point : tuple
        Coordinate in (lon, lat) form.
    shape_data : GeoDataFrame
        Dataframe storing geometries of NUTS regions.
    indicator_data : dict
        Dict object storing technical potential of NUTS regions.

    Returns
    -------
    incumbent_region : str
        Region in which point "p" falls.
    """
    dist = {}

    p = Point(point)

    incumbent_region = None

    for subregion in list(indicator_data.keys()):

        if subregion in shape_data.index:

            if p.within(shape_data.loc[subregion, 'geometry']):

                incumbent_region = subregion

            dist[subregion] = p.distance(shape_data.loc[subregion, 'geometry'])

    if incumbent_region is None:

        print(p, min(dist, key=dist.get))

        incumbent_region = min(dist, key=dist.get)

        # else:
        #
        #     pass
    #
    # if incumbent_region == None:
    #
    #     warnings.warn(' Point {} does not fall in any of the pre-defined regions.'.format(point))

    return incumbent_region


# TODO: ok i have a file for this
def return_ISO_codes_from_countries():

    dict_ISO = {'Albania': 'AL', 'Armenia': 'AR', 'Belarus': 'BL', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
                'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Estonia': 'EE', 'Latvia': 'LV', 'Lithuania': 'LT',
                'Denmark': 'DK', 'France': 'FR', 'Germany': 'DE', 'Greece': 'EL', 'Hungary': 'HU', 'Ireland': 'IE',
                'Italy': 'IT', 'Macedonia': 'MK', 'Malta': 'MT', 'Norway': 'NO', 'Iceland': 'IS', 'Finland': 'FI',
                'Montenegro': 'MN', 'Netherlands': 'NL', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
                'Slovak Republic': 'SK', 'Spain': 'ES', 'Sweden': 'SE',
                'Switzerland': 'CH', 'Turkey': 'TR', 'Ukraine': 'UA', 'United Kingdom': 'UK'}

    return dict_ISO
