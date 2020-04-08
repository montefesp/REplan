from os.path import join, dirname, abspath
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from shapely.ops import cascaded_union
from shapely.geometry import Polygon, MultiPolygon
from shapely.errors import TopologicalError

from src.data.geographics.manager import get_onshore_shapes, get_offshore_shapes, \
    match_points_to_region
from src.data.population_density.manager import load_population_density_data
# from src.data.geographics.manager import nuts3_to_nuts2, get_nuts_area,
# from src.data.topologies.ehighways import get_ehighway_clusters


# TODO: need to revise the functions in this file

# TODO: check what it takes for this function not to crash on the tyndp model
def append_non_EU_potential(input_ds: pd.DataFrame, tech: str) -> pd.DataFrame:

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/generated')

    data_potential = pd.read_excel(join(path_potential_data, 'RES_potential_non_EU.xlsx'), index_col=0)
    data_potential = data_potential[tech].dropna().copy()

    if tech in ['wind_onshore', 'pv_utility', 'pv_residential']:

        path_pop_dens_data = join(dirname(abspath(__file__)), '../../../data/population_density')
        data_population = pd.read_csv(join(path_pop_dens_data, 'pop_dens_nuts2.csv'), index_col=0, sep=';')

        potential_df = pd.DataFrame([])

        for item in data_potential.index:

            nuts_regions = [i for i in data_population.index if (i.startswith(item))]
            potential_dict = {key: None for key in nuts_regions}

            if tech in ['wind_onshore', 'pv_utility']:
                data_population = 1./data_population

            pop_dens_sum = data_population.loc[nuts_regions].sum()
            for region in nuts_regions:
                    potential_dict[region] = round(data_potential.loc[item]*(data_population.loc[region]/pop_dens_sum),6)

            potential_df = pd.concat([potential_df, pd.DataFrame.from_dict(potential_dict, orient='index')])

        potential_df = potential_df['pop_dens']

    else: # tech in ['wind_offshore', 'wind_floating']

        potential_df = data_potential
        potential_df.index = ['EZ'+string for string in data_potential.index]

    input_ds = pd.concat([input_ds, potential_df], axis=0, ignore_index=False)

    return input_ds








# TODO: need to change this - where does these values come from
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

    input_ds = append_non_EU_potential(input_ds, tech)

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

        # Update according to the Irish NUTS2 zones, shifting from 2 to 3 zones in 2014.
        input_ds.at['IE04'] = input_ds.at['IE01']
        input_ds.at['IE05'] = input_ds.at['IE02']
        input_ds.at['IE06'] = 0. # Dublin area.
        # Update according to the Lithuanian NUTS2 zones, shifting from 1 to 2 zones in 2016.
        input_ds.at['LT01'] = 0. # Region of Vilnius.
        input_ds.at['LT02'] = input_ds.at['LT00']
        # Update according to the Scottish NUTS2 zones in 2016.
        input_ds.at['UKM8'] = 0. # Glasgow area.
        input_ds.at['UKM9'] = input_ds.at['UKM3']
        # Update according to the Warsaw enclave.
        input_ds.at['PL92'] = input_ds.at['PL9']
        input_ds.at['PL91'] = 0. # Inner city of Warsaw.
        # Update according to the Budapest split in Budapest (enclave city) and Pest (the region).
        input_ds.at['HU11'] = 0. # Inner city of Budapest.
        input_ds.at['HU12'] = input_ds.at['HU10']
        # Inner London
        input_ds.at['UKI5'] = 0.
        input_ds.at['UKI6'] = 0.
        input_ds.at['UKI7'] = 0.

    elif tech == 'pv_residential':

        # Update according to the Irish NUTS2 zones, shifting from 2 to 3 zones in 2014.
        input_ds.at['IE04'] = input_ds.at['IE01']
        input_ds.at['IE05'] = input_ds.at['IE02']*(1/3)
        input_ds.at['IE06'] = input_ds.at['IE02']*(2/3) # Dublin region, two thirds of population.
        # Update according to the Lithuanian NUTS2 zones, shifting from 1 to 2 zones in 2016.
        input_ds.at['LT01'] = input_ds.at['LT00']*(1/3) # Capital region, one third of population.
        input_ds.at['LT02'] = input_ds.at['LT00']*(2/3) # Rest of the country.
        # Update according to the Scottish NUTS2 zones in 2016.
        input_ds.at['UKM8'] = input_ds.at['UKM3']*(1/3) # Glasgow region, split based on population share.
        input_ds.at['UKM9'] = input_ds.at['UKM3']*(2/3)
        # Update according to the Warsaw enclave.
        input_ds.at['PL92'] = input_ds.at['PL9']*(1/2) # Warsaw region, split based on population share.
        input_ds.at['PL91'] = input_ds.at['PL9']*(1/2)
        # Update according to the Budapest split in Budapest (enclave city) and Pest (the region).
        input_ds.at['HU11'] = input_ds.at['HU10']*(1/2)
        input_ds.at['HU12'] = input_ds.at['HU10']*(1/2)
        # Outer London. Values not assigned within ENSPRESO.
        input_ds.at['UKI5'] = 0.
        input_ds.at['UKI6'] = 0.
        input_ds.at['UKI7'] = 0.

    elif tech == 'pv_utility':

        # Update according of the Irish NUTS2 zones, shifting from 2 to 3 zones in 2014.
        input_ds.at['IE04'] = input_ds.at['IE01']
        input_ds.at['IE05'] = input_ds.at['IE02']
        input_ds.at['IE06'] = 0. # Dublin city area.
        # Update according of the Lithuanian NUTS2 zones, shifting from 1 to 2 zones in 2016.
        input_ds.at['LT01'] = 0. # Capital region.
        input_ds.at['LT02'] = input_ds.at['LT00'] # Rest of the country.
        # Update according to the Scottish NUTS2 zones in 2016.
        input_ds.at['UKM8'] = 0. # Glasgow area.
        input_ds.at['UKM9'] = input_ds.at['UKM3']
        # Update according to the Warsaw enclave.
        input_ds.at['PL92'] = input_ds.at['PL9']
        input_ds.at['PL91'] = 0. # Inner city of Warsaw.
        # Update according to the Budapest split in Budapest (enclave city) and Pest (the region).
        input_ds.at['HU11'] = 0. # Inner city of Budapest.
        input_ds.at['HU12'] = input_ds.at['HU10']
        # Inner London.
        input_ds.at['UKI5'] = 0.
        input_ds.at['UKI6'] = 0.
        input_ds.at['UKI7'] = 0.


    elif tech == 'wind_offshore':

        input_ds.at['EZUK'] = input_ds.at['EZGB']
        input_ds.at['EZEL'] = input_ds.at['EZGR']

    elif tech == 'wind_floating':

        input_ds.at['EZUK'] = input_ds.at['EZGB']
        input_ds.at['EZEL'] = input_ds.at['EZGR']

    regions_to_remove = ['AD00', 'SM00', 'CY00', 'LI00', 'FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5', 'ES63', 'ES64', 'ES70',
                         'HU10', 'IE01', 'IE02', 'LT00', 'UKM3']

    input_ds = input_ds.drop(regions_to_remove, errors='ignore')

    return input_ds


def capacity_potential_from_enspresso(tech: str) -> pd.DataFrame:
    """
    Returning capacity potential (in GW) per NUTS2 region for a given tech, based on the ENSPRESSO dataset.

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
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/source/ENSPRESO')
    # For wind, summing over all wind conditions is similar to considering taking all available land and a capacity per
    #  area of 5MW/km2
    # TODO: Ask david: why are we using high-restrictions onshore and low offshore?
    #  Why not use the reference scenario in both cases?
    if tech == 'wind_onshore':

        cap_potential_file = pd.read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                   sheet_name='Raw data', index_col=1, skiprows=5)

        onshore_wind = cap_potential_file[
            (cap_potential_file['ONOFF'] == 'Onshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide high restrictions') &
            (cap_potential_file['Subscenario - not cumulative'] == '2000m setback distance')]
        nuts2_capacity_potentials_ds = onshore_wind['GW_Morethan25%_2030_100m_ALLTIMESLICESAVERAGE_V112'].copy()

    elif tech == 'wind_offshore':

        offshore_categories = ['12nm zone, water depth 0-30m', '12nm zone, water depth 30-60m',
                               'Water depth 0-30m', 'Water depth 30-60m']

        cap_potential_file = pd.read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                           sheet_name='Wind Potential EU28 Full', index_col=1)

        offshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Offshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide low restrictions') &
            (cap_potential_file['Wind condition'] == 'CF > 25%') &
            (cap_potential_file['Offshore categories'].isin(offshore_categories))]
        nuts2_capacity_potentials_ds = offshore_wind.groupby(offshore_wind.index)['Value'].sum()

    elif tech == 'wind_floating':

        floating_categories = ['12nm zone, water depth 60-100m Floating',
                               'Water depth 60-100m Floating', 'Water depth 100-1000m Floating']

        cap_potential_file = pd.read_excel(join(path_potential_data, 'ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX'),
                                           sheet_name='Wind Potential EU28 Full', index_col=1)

        offshore_wind = cap_potential_file[
            (cap_potential_file['Unit'] == 'GWe') &
            (cap_potential_file['Onshore Offshore'] == 'Offshore') &
            (cap_potential_file['Scenario'] == 'EU-Wide low restrictions') &
            (cap_potential_file['Wind condition'] == 'CF > 25%') &
            (cap_potential_file['Offshore categories'].isin(floating_categories))]
        nuts2_capacity_potentials_ds = offshore_wind.groupby(offshore_wind.index)['Value'].sum()

    elif tech == 'pv_utility':

        cap_potential_file = pd.read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                           sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)
        nuts2_capacity_potentials_ds = cap_potential_file['PV - ground']

    else:  # 'pv_residential'

        cap_potential_file = pd.read_excel(join(path_potential_data, 'ENSPRESO_SOLAR_PV_CSP.XLSX'),
                                           sheet_name='NUTS2 170 W per m2 and 3%', skiprows=2, index_col=2)
        nuts2_capacity_potentials_ds = cap_potential_file['PV - roof/facades']

    return update_potential_files(nuts2_capacity_potentials_ds, tech)


def get_capacity_potential(tech_points_dict: Dict[str, List[Tuple[float, float]]], spatial_resolution: float,
                           regions: List[str], existing_capacity_ds: pd.Series = None) -> pd.Series:
    """
    Computes the capacity that can potentially be deployed at a series of points for different technologies

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
    capacity_potential_ds : pd.Series
        Gives for each pair of technology - point the associated capacity potential in GW
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in tech_points_dict.keys():
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    # Create a modified copy of regions to deal with UK and EL
    nuts0_problems = {"GB": "UK", "GR": "EL"}
    nuts0_regions = [nuts0_problems[r] if r in nuts0_problems else r for r in regions]

    array_pop_density = load_population_density_data(spatial_resolution)

    tech_coords_tuples = [(tech, point) for tech, points in tech_points_dict.items() for point in points]
    capacity_potential_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_coords_tuples))

    for tech, coords in tech_points_dict.items():

        # Compute potential for each NUTS2 or EEZ
        potential_per_region_df = capacity_potential_from_enspresso(tech)

        # Get NUTS2 and EEZ shapes
        # TODO: this is shit -> not generic enough, expl: would probably not work for us states
        #  would need to get this out of the loop
        if tech in ['wind_offshore', 'wind_floating']:
            onshore_shapes_union = \
                cascaded_union(get_onshore_shapes(regions, filterremote=True,
                                                  save_file_name=''.join(sorted(regions))
                                                                 + "_regions_on.geojson")["geometry"].values)
            filter_shape_data = get_offshore_shapes(regions, onshore_shape=onshore_shapes_union,
                                                    filterremote=True,
                                                    save_file_name=''.join(sorted(regions))
                                                                   + "_regions_off.geojson")
            filter_shape_data.index = ["EZ" + code for code in filter_shape_data.index]
        else:
            codes = [code for code in potential_per_region_df.index if code[:2] in nuts0_regions]
            filter_shape_data = get_onshore_shapes(codes, filterremote=True,
                                                   save_file_name=''.join(sorted(regions)) + "_nuts2_on.geojson")

        # Find the geographical region code associated to each coordinate
        coords_regions_ds = match_points_to_region(coords, filter_shape_data["geometry"]).dropna()
        coords = list(coords_regions_ds.index)
        coords_regions_df = pd.DataFrame(coords_regions_ds.values, coords_regions_ds.index,
                                               columns=["region"])

        if tech in ['wind_offshore', 'wind_floating']:

            # For offshore sites, divide the total potential of the region by the number of coordinates
            # associated to that region
            # TODO: change variable names
            region_freq_ds = coords_regions_df.groupby(['region'])['region'].count()
            region_freq_df = pd.DataFrame(region_freq_ds.values, index=region_freq_ds.index, columns=['freq'])
            region_freq_df["cap_pot"] = potential_per_region_df[region_freq_df.index]
            coords_regions_df = \
                coords_regions_df.merge(region_freq_df,
                                              left_on='region', right_on='region', right_index=True)
            capacity_potential = coords_regions_df["cap_pot"]/coords_regions_df["freq"]
            capacity_potential_ds.loc[tech, capacity_potential.index] = capacity_potential.values

        elif tech in ['wind_onshore', 'pv_utility', 'pv_residential']:

            # TODO: change variable names
            coords_regions_df['pop_dens'] = \
                 np.clip(array_pop_density.sel(locations=coords).values, a_min=1., a_max=None)

            if tech in ['wind_onshore', 'pv_utility']:
                coords_regions_df['pop_dens'] = 1./coords_regions_df['pop_dens']

            # Keep only the potential of regions in which points fall
            coords_to_regions_df_sum = coords_regions_df.groupby(['region']).sum()
            coords_to_regions_df_sum["cap_pot"] = potential_per_region_df[coords_to_regions_df_sum.index]
            coords_to_regions_df_sum.columns = ['sum_per_region', 'cap_pot']
            coords_to_regions_df_merge = \
                coords_regions_df.merge(coords_to_regions_df_sum,
                                        left_on='region', right_on='region', right_index=True)

            capacity_potential_per_coord = coords_to_regions_df_merge['pop_dens'] * \
                coords_to_regions_df_merge['cap_pot']/coords_to_regions_df_merge['sum_per_region']
            capacity_potential_ds.loc[tech, capacity_potential_per_coord.index] = capacity_potential_per_coord.values

    # Update capacity potential with existing potential if present
    if existing_capacity_ds is not None:
        underestimated_capacity = existing_capacity_ds > capacity_potential_ds
        capacity_potential_ds[underestimated_capacity] = existing_capacity_ds[underestimated_capacity]

    return capacity_potential_ds


def get_capacity_potential_for_regions(tech_regions_dict: Dict[str, List[Union[Polygon, MultiPolygon]]]) -> pd.Series:
    """
    Get capacity potential (in GW) for a series of technology for associated geographical regions

    Parameters
    ----------
    tech_regions_dict: Dict[str, List[Union[Polygon, MultiPolygon]]]
        Dictionary giving for each technology for which region we want to obtain potential capacity

    Returns
    -------
    capacity_potential_ds: pd.Series
        Gives for each pair of technology and region the associated potential capacity in GW

    """
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in tech_regions_dict.keys():
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    tech_regions_tuples = [(tech, i) for tech, points in tech_regions_dict.items() for i in range(len(points))]
    capacity_potential_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_regions_tuples))

    for tech, regions in tech_regions_dict.items():

        # Compute potential for each NUTS2 or EEZ
        potential_per_subregion_df = capacity_potential_from_enspresso(tech)

        # Get NUTS2 or EEZ shapes
        # TODO: this is shit -> not generic enough, expl: would probably not work for us states
        #  would need to get this out of the loop
        if tech in ['wind_offshore', 'wind_floating']:
            codes = [code[2:4] for code in potential_per_subregion_df.index.values]
            onshore_shapes_union = \
                cascaded_union(get_onshore_shapes(codes, filterremote=True,
                                                  save_file_name="cap_potential_regions_on.geojson")["geometry"].values)
            shapes = get_offshore_shapes(codes, onshore_shape=onshore_shapes_union,
                                         filterremote=True, save_file_name="cap_potential_regions_off.geojson")
            shapes.index = ["EZ" + code for code in shapes.index]
        else:
            shapes = get_onshore_shapes(potential_per_subregion_df.index.values, filterremote=True,
                                        save_file_name="cap_potential_regions_on.geojson")

        # Compute capacity potential for the regions given as argument
        for i, region in enumerate(regions):
            cap_pot = 0
            for index, shape in shapes.iterrows():
                try:
                    intersection = region.intersection(shape["geometry"])
                except TopologicalError:
                    print(f"Warning: Problem with shape for code {index}")
                    continue
                if intersection.is_empty or intersection.area == 0.:
                    continue
                cap_pot += potential_per_subregion_df[index]*intersection.area/shape["geometry"].area
                try:
                    region = region.difference(intersection)
                except TopologicalError:
                    print(f"Warning: Problem with shape for code {index}")
                if region.is_empty or region.area == 0.:
                    break
            capacity_potential_ds.loc[tech, i] = cap_pot

    return capacity_potential_ds

# missing_region_dict = {
#      "NO": ["SE"],
#      "CH": ["AT"],
#      "BA": ["HR"],
#      "ME": ["HR"],
#      "RS": ["BG"],
#      "AL": ["BG"],
#      "MK": ["BG"]
# }
#
# shitty because only working for e-highway -> should modify it or remove it from here
# need to add offshore potential computation
# improve based on similar function in generation.manager
# def get_potential_ehighway(bus_ids: List[str], tech: str) -> pd.DataFrame:
#     """
#     Returns the RES potential in GW/km2 for e-highway clusters
#
#     Parameters
#     ----------
#     bus_ids: List[str]
#         E-highway clusters identifier (used as bus_ids in the network)
#     tech: str
#         wind or pv
#
#     Returns
#     -------
#     total_capacities: pd.DataFrame indexed by bus_ids
#         Capacity per bus in GWe/km
#
#     """
#     # Get capacities
#     data_dir = join(dirname(abspath(__file__)), "../../../data/res_potential/source/ENSPRESO/")
#     capacities = []
#     if tech == "pv_utility":
#         unit = "GWe"
#         capacities = pd.read_excel(join(data_dir, "ENSPRESO_SOLAR_PV_CSP.XLSX"),
#                                    sheet_name="NUTS2 170 W per m2 and 3%",
#                                    usecols="C,H", header=2)
#         capacities.columns = ["code", "capacity"]
#         capacities["unit"] = pd.Series([unit]*len(capacities.index))
#
#     elif tech == "wind_onshore":
#         unit = "GWe"
#         # ODO: Need to pay attention to this scenario thing
#         scenario = "Reference - Large turbines"
#         # cap_factor = "20%  < CF < 25%"  # TDO: not to sure what to do with this
#         capacities = pd.read_excel(join(data_dir, "ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX"),
#                                    sheet_name="Wind Potential EU28 Full",
#                                    usecols="B,F,G,I,J")
#         capacities.columns = ["code", "scenario", "cap_factor", "unit", "capacity"]
#         capacities = capacities[capacities.scenario == scenario]
#         capacities = capacities[capacities.unit == unit]
#         # capacity = capacity[capacity.cap_factor == cap_factor]
#         capacities = capacities.groupby(["code", "unit"], as_index=False).agg('sum')
#     else:
#         print("This carrier is not supported")
#
#     # Transforming capacities to capacities per km2
#     area = get_nuts_area()
#     area.index.name = 'code'
#
#     nuts2_conversion_fn = join(dirname(abspath(__file__)),
#                                "../../../data/geographics/source/eurostat/NUTS2-conversion.csv")
#     nuts2_conversion = pd.read_csv(nuts2_conversion_fn, index_col=0)
#
#     # Convert index to new nuts2
#     for old_code in nuts2_conversion.index:
#         old_capacity = capacities[capacities.code == old_code]
#         old_area = area.loc[old_code]["2013"]
#         for new_code in nuts2_conversion.loc[old_code]["Code 2016"].split(";"):
#             new_area = area.loc[new_code]["2016"]
#             new_capacity = old_capacity.copy()
#             new_capacity.code = new_code
#             new_capacity.capacity = old_capacity.capacity*new_area/old_area
#             capacities = capacities.append(new_capacity, ignore_index=True)
#         capacities = capacities.drop(capacities[capacities.code == old_code].index)
#
#     # The areas are in square kilometre so we obtain GW/km2
#     def to_cap_per_area(x):
#         return x["capacity"]/area.loc[x["code"]]["2016"] if x["code"] in area.index else None
#     capacities["capacity"] = capacities[["code", "capacity"]].apply(lambda x: to_cap_per_area(x), axis=1)
#     capacities = capacities.set_index("code")
#
#     # Get codes of NUTS3 regions and countries composing the cluster
#     eh_clusters = get_ehighway_clusters()
#
#     total_capacities = np.zeros(len(bus_ids))
#     for i, bus_id in enumerate(bus_ids):
#
#         # ODO: would probably need to do sth more clever
#         #  --> just setting capacitities at seas as 10MW/km2
#         # if bus_id not in eh_clusters.index:
#         #     total_capacities[i] = 0.01 if tech == 'wind' else 0
#         #     continue
#
#         codes = eh_clusters.loc[bus_id].codes.split(",")
#
#         # ODO: this is a shitty hack
#         if codes[0][0:2] in missing_region_dict:
#             codes = missing_region_dict[codes[0][0:2]]
#
#         if len(codes[0]) != 2:
#             nuts2_codes = nuts3_to_nuts2(codes)
#             total_capacities[i] = np.average([capacities.loc[code]["capacity"] for code in nuts2_codes],
#                                              weights=[area.loc[code]["2016"] for code in codes])
#         else:
#             # If the code corresponds to a countries, get the correspond list of NUTS2
#             nuts2_codes = [code for code in capacities.index.values if code[0:2] == codes[0]]
#             total_capacities[i] = np.average([capacities.loc[code]["capacity"] for code in nuts2_codes],
#                                              weights=[area.loc[code]["2016"] for code in nuts2_codes])
#
#     return pd.DataFrame(total_capacities, index=bus_ids, columns=["capacity"]).capacity
