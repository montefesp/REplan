import os
import glob
import numpy as np
from copy import deepcopy
from pandas import DataFrame
import xarray as xr


# TODO:
#  - resite.dataset
#  - Why is this called xarray_to_dict? xarray_to_ndarray?
def xarray_to_dict(input_dict, levels):
    """
    Converts dict of xarray objects to dict of ndarrays to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    levels : int # TODO: isn't it a cleaner way to access the depth of a dictionary?
        Depth of (nested) dict. Available values: 1 or 2.

    Returns
    -------
    output_dict : dict

    """
    output_dict = deepcopy(input_dict)

    if levels == 2:
        for region, tech in return_dict_keys(input_dict):
            output_dict[region][tech] = input_dict[region][tech].values

    else:
        for tech in input_dict.keys():
            output_dict[tech] = input_dict[tech].values

    return output_dict


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

    array_list = []

    for region, tech in return_dict_keys(input_dict):

        array_list.append(input_dict[region][tech])

    dataset = xr.concat(array_list, dim='locations')

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
        output_dict[item] = xr.concat(l, dim='locations')

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

    d = {}
    for key in input_dict.keys():
        array = input_dict[key]
        for idx, val in enumerate(array.locations):
            d[(key, idx)] = val

    return d


# TODO:
#  - Need to change this function
#  - we should pass only the parts of the input dict that are actually used
def retrieve_location_dict(input_dict, instance, technologies):

    output_dict = {key: {} for key in technologies}
    potential_dict = collapse_dict_region_level(input_dict['capacity_potential'])

    dict_ = retrieve_tech_coordinates_tuples(input_dict['existing_cap_percentage'])
    for tuple_key in dict_:
        if instance.y[tuple_key].value > 0.:
            tech = tuple_key[0]
            loc = dict_[tuple_key]
            cap = instance.y[tuple_key].value * potential_dict[tech].sel(locations=loc).values

            output_dict[tech][loc.values.item()] = cap

    output_dict = {k: v for k, v in output_dict.items() if len(v) > 0}

    return output_dict


# TODO:
#  - need to more precise on description of function, and name it more specifically
#  - add the filtering on coordinates
def read_database(file_path, coordinates=None):
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


# TODO:
#  - don't really know where this should go
#  - I would just remove this function and make the computation directly where it is used (i.e. models.py)
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

