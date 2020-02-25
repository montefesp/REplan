import os
import glob
import numpy as np
from copy import deepcopy
from pandas import DataFrame
import xarray as xr


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


def get_tech_coords_tuples(tech_coord_dict):

    tech_coords_tuples = []
    for tech, tech_coords in tech_coord_dict.items():
        for coord in tech_coords:
            tech_coords_tuples.append((tech, coord))

    return tech_coords_tuples


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
#  - Need to change this function
#  - we should pass only the parts of the input dict that are actually used
def retrieve_location_dict(input_dict, instance, technologies):

    output_dict = {key: {} for key in technologies}
    potential_dict = input_dict['capacity_potential_df']

    tech_coordinates_list = list(input_dict['existing_cap_percentage_df'].index)
    # TODO: it's a bit shitty to have to do that but it's bugging otherwise
    tech_coordinates_list = [(tech, coord[0], coord[1]) for tech, coord in tech_coordinates_list]

    for tuple_key in tech_coordinates_list:
        print(tuple_key)
        if instance.y[tuple_key].value > 0.:
            tech = tuple_key[0]
            loc = (tuple_key[1], tuple_key[2])
            print(tech)
            print(loc)
            print(potential_dict)
            cap = instance.y[tuple_key].value * potential_dict[tech, loc].values

            output_dict[tech][loc] = cap

    output_dict = {k: v for k, v in output_dict.items() if len(v) > 0}
    print(output_dict)

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
#  - need to change this - use what I have done in my code or try at least
#  - data.res_potential
def update_potential_files(input_ds, tech):
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

    input_ds = input_ds.drop(regions_to_remove, errors='ignore')

    if tech in ['wind_onshore', 'solar_residential', 'solar_utility']:

        dict_regions_update = {'FR21': 'FRF2', 'FR22': 'FRE2', 'FR23': 'FRD1', 'FR24': 'FRB0', 'FR25': 'FRD2',
                               'FR26': 'FRC1', 'FR30': 'FRE1', 'FR41': 'FRF3', 'FR42': 'FRF1', 'FR43': 'FRC2',
                               'FR51': 'FRG0', 'FR52': 'FRH0', 'FR53': 'FRI3', 'FR61': 'FRI1', 'FR62': 'FRJ2',
                               'FR63': 'FRI2', 'FR71': 'FRK2', 'FR72': 'FRK1', 'FR81': 'FRJ1', 'FR82': 'FRL0',
                               'FR83': 'FRM0'}

        new_index = [dict_regions_update[x] if x in dict_regions_update else x for x in input_ds.index]
        input_ds.index = new_index

    if tech == 'wind_onshore':

        input_ds['AL01'] = 2.
        input_ds['AL02'] = 2.
        input_ds['AL03'] = 2.
        input_ds['BA00'] = 3.
        input_ds['ME00'] = 3.
        input_ds['MK00'] = 5.
        input_ds['RS11'] = 0.
        input_ds['RS12'] = 10.
        input_ds['RS21'] = 10.
        input_ds['RS22'] = 10.
        input_ds['CH01'] = 1.
        input_ds['CH02'] = 1.
        input_ds['CH03'] = 1.
        input_ds['CH04'] = 1.
        input_ds['CH05'] = 1.
        input_ds['CH06'] = 1.
        input_ds['CH07'] = 1.
        input_ds['NO01'] = 3.
        input_ds['NO02'] = 3.
        input_ds['NO03'] = 3.
        input_ds['NO04'] = 3.
        input_ds['NO05'] = 3.
        input_ds['NO06'] = 3.
        input_ds['NO07'] = 3.
        input_ds['IE04'] = input_ds['IE01']
        input_ds['IE05'] = input_ds['IE02']
        input_ds['IE06'] = input_ds['IE02']
        input_ds['LT01'] = input_ds['LT00']
        input_ds['LT02'] = input_ds['LT00']
        input_ds['UKM7'] = input_ds['UKM2']
        input_ds['UKM8'] = input_ds['UKM5']
        input_ds['UKM9'] = input_ds['UKM3']
        input_ds['PL71'] = input_ds['PL11']
        input_ds['PL72'] = input_ds['PL33']
        input_ds['PL81'] = input_ds['PL31']
        input_ds['PL82'] = input_ds['PL32']
        input_ds['PL84'] = input_ds['PL34']
        input_ds['PL92'] = input_ds['PL12']
        input_ds['PL91'] = 0.
        input_ds['HU11'] = 0.
        input_ds['HU12'] = input_ds['HU10']
        input_ds['UKI5'] = 0.
        input_ds['UKI6'] = 0.
        input_ds['UKI7'] = 0.

    elif tech == 'wind_offshore':

        input_ds['EZAL'] = 2.
        input_ds['EZBA'] = 0.
        input_ds['EZME'] = 0.
        input_ds['EZMK'] = 0.
        input_ds['EZRS'] = 0.
        input_ds['EZCH'] = 0.
        input_ds['EZNO'] = 20.
        input_ds['EZIE'] = 20.
        input_ds['EZEL'] = input_ds['EZGR']

    elif tech == 'wind_floating':

        input_ds['EZAL'] = 2.
        input_ds['EZBA'] = 0.
        input_ds['EZME'] = 0.
        input_ds['EZMK'] = 0.
        input_ds['EZRS'] = 0.
        input_ds['EZCH'] = 0.
        input_ds['EZNO'] = 100.
        input_ds['EZIE'] = 120.
        input_ds['EZEL'] = input_ds['EZGR']

    elif tech == 'solar_residential':

        input_ds['AL01'] = 1.
        input_ds['AL02'] = 1.
        input_ds['AL03'] = 1.
        input_ds['BA00'] = 3.
        input_ds['ME00'] = 1.
        input_ds['MK00'] = 1.
        input_ds['RS11'] = 5.
        input_ds['RS12'] = 2.
        input_ds['RS21'] = 2.
        input_ds['RS22'] = 2.
        input_ds['CH01'] = 6.
        input_ds['CH02'] = 6.
        input_ds['CH03'] = 6.
        input_ds['CH04'] = 6.
        input_ds['CH05'] = 6.
        input_ds['CH06'] = 6.
        input_ds['CH07'] = 6.
        input_ds['NO01'] = 3.
        input_ds['NO02'] = 0.
        input_ds['NO03'] = 3.
        input_ds['NO04'] = 3.
        input_ds['NO05'] = 0.
        input_ds['NO06'] = 0.
        input_ds['NO07'] = 0.
        input_ds['IE04'] = input_ds['IE01']
        input_ds['IE05'] = input_ds['IE02']
        input_ds['IE06'] = input_ds['IE02']
        input_ds['LT01'] = input_ds['LT00']
        input_ds['LT02'] = input_ds['LT00']
        input_ds['UKM7'] = input_ds['UKM2']
        input_ds['UKM8'] = input_ds['UKM5']
        input_ds['UKM9'] = input_ds['UKM3']
        input_ds['PL71'] = input_ds['PL11']
        input_ds['PL72'] = input_ds['PL33']
        input_ds['PL81'] = input_ds['PL31']
        input_ds['PL82'] = input_ds['PL32']
        input_ds['PL84'] = input_ds['PL34']
        input_ds['PL92'] = input_ds['PL12']
        input_ds['PL91'] = 5.
        input_ds['HU11'] = input_ds['HU10']
        input_ds['HU12'] = input_ds['HU10']
        input_ds['UKI5'] = 1.
        input_ds['UKI6'] = 1.
        input_ds['UKI7'] = 1.

    elif tech == 'solar_utility':

        input_ds['AL01'] = 1.
        input_ds['AL02'] = 1.
        input_ds['AL03'] = 1.
        input_ds['BA00'] = 3.
        input_ds['ME00'] = 1.
        input_ds['MK00'] = 1.
        input_ds['RS11'] = 0.
        input_ds['RS12'] = 2.
        input_ds['RS21'] = 2.
        input_ds['RS22'] = 1.
        input_ds['CH01'] = 6.
        input_ds['CH02'] = 6.
        input_ds['CH03'] = 6.
        input_ds['CH04'] = 6.
        input_ds['CH05'] = 6.
        input_ds['CH06'] = 6.
        input_ds['CH07'] = 6.
        input_ds['NO01'] = 3.
        input_ds['NO02'] = 0.
        input_ds['NO03'] = 3.
        input_ds['NO04'] = 3.
        input_ds['NO05'] = 0.
        input_ds['NO06'] = 0.
        input_ds['NO07'] = 0.
        input_ds['IE04'] = input_ds['IE01']
        input_ds['IE05'] = input_ds['IE02']
        input_ds['IE06'] = input_ds['IE02']
        input_ds['LT01'] = input_ds['LT00']
        input_ds['LT02'] = input_ds['LT00']
        input_ds['UKM7'] = input_ds['UKM2']
        input_ds['UKM8'] = input_ds['UKM5']
        input_ds['UKM9'] = input_ds['UKM3']
        input_ds['PL71'] = input_ds['PL11']
        input_ds['PL72'] = input_ds['PL33']
        input_ds['PL81'] = input_ds['PL31']
        input_ds['PL82'] = input_ds['PL32']
        input_ds['PL84'] = input_ds['PL34']
        input_ds['PL92'] = input_ds['PL12']
        input_ds['PL91'] = 2.
        input_ds['HU11'] = 0.
        input_ds['HU12'] = 2.
        input_ds['UKI5'] = 0.
        input_ds['UKI6'] = 0.
        input_ds['UKI7'] = 0.

    return input_ds

