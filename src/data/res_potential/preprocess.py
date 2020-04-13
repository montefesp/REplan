from os.path import join, dirname, abspath

import pandas as pd


def get_non_eu28_potential(tech: str) -> pd.Series:
    """
    Returns capacity potential per NUTS2 or EEZ region of countries that are not in EU28 for a given technology.

    Parameters
    ----------
    tech: str
        One of ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']

    Returns
    -------
    capacity_potential_ds: pd.Series

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    # Load capacity potential per non-EU28 country (GW)
    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/source')
    capacity_potential_non_eu28 = pd.read_excel(join(path_potential_data, 'RES_potential_non_EU.xlsx'), index_col=0)
    capacity_potential_non_eu28 = capacity_potential_non_eu28[tech].dropna().copy()  # TODO: ask david: why a copy?

    if tech in ['wind_onshore', 'pv_utility', 'pv_residential']:

        # For onshore technologies, divide capacity among NUTS2 regions proportionally (for pv_residential)
        # or inversely proportional (for wind_onshore and pv_utility) to population
        # TODO: will need to use some more reliable data source or at least reference this one better
        path_pop_dens_data = join(dirname(abspath(__file__)), '../../../data/population_density')
        nuts2_pop_dens = pd.read_csv(join(path_pop_dens_data, 'pop_dens_nuts2.csv'), index_col=0, sep=';')

        # Compute NUTS2 capacities country by country
        capacity_potential_ds = pd.Series()
        for country_code in capacity_potential_non_eu28.index:

            country_capacity_potential = capacity_potential_non_eu28.loc[country_code]

            # Get all NUTS2 regions in country
            nuts2_codes = [nuts2_code for nuts2_code in nuts2_pop_dens.index if (nuts2_code.startswith(country_code))]
            pop_dens = nuts2_pop_dens.loc[nuts2_codes, "pop_dens"]

            if tech in ['wind_onshore', 'pv_utility']:
                pop_dens = 1./pop_dens

            capacity_potential_ds = \
                pd.concat([capacity_potential_ds,
                           country_capacity_potential * (pop_dens.loc[nuts2_codes]/pop_dens.sum())])

        capacity_potential_ds = capacity_potential_ds.round(6)

    else:  # tech in ['wind_offshore', 'wind_floating']

        capacity_potential_ds = capacity_potential_non_eu28
        capacity_potential_ds.index = "EZ" + capacity_potential_ds.index

    return capacity_potential_ds


def update_enspreso_capacity_potential(capacity_potential_ds: pd.Series, tech: str) -> pd.Series:
    """
    Updates potentials from ENSPRESO with re-indexed (2013 vs 2016) NUTS2 regions.

    Parameters
    ----------
    capacity_potential_ds: pd.Series
        Series giving for a series of NUTS2 regions the potential capacity for a given technology
    tech : str
        Technology for which we are computing potential capacities

    Returns
    -------
    capacity_potential_ds : pd.Series
        Updated Series
    """

    # Update names that have changed
    if tech in ['wind_onshore', 'pv_residential', 'pv_utility']:

        dict_regions_update = {'FR21': 'FRF2', 'FR22': 'FRE2', 'FR23': 'FRD1', 'FR24': 'FRB0', 'FR25': 'FRD2',
                               'FR26': 'FRC1', 'FR30': 'FRE1', 'FR41': 'FRF3', 'FR42': 'FRF1', 'FR43': 'FRC2',
                               'FR51': 'FRG0', 'FR52': 'FRH0', 'FR53': 'FRI3', 'FR61': 'FRI1', 'FR62': 'FRJ2',
                               'FR63': 'FRI2', 'FR71': 'FRK2', 'FR72': 'FRK1', 'FR81': 'FRJ1', 'FR82': 'FRL0',
                               'FR83': 'FRM0', 'PL11': 'PL71', 'PL12': 'PL9', 'PL31': 'PL81', 'PL32': 'PL82',
                               'PL33': 'PL72', 'PL34': 'PL84', 'UKM2': 'UKM7'}

        new_index = [dict_regions_update[x] if x in dict_regions_update else x for x in capacity_potential_ds.index]
        capacity_potential_ds.index = new_index

    # Update capacities for zones that have been divided
    if tech == 'wind_onshore':

        # Update according to the Irish NUTS2 zones, shifting from 2 to 3 zones in 2014.
        capacity_potential_ds.at['IE04'] = capacity_potential_ds.at['IE01']
        capacity_potential_ds.at['IE05'] = capacity_potential_ds.at['IE02']
        capacity_potential_ds.at['IE06'] = 0.  # Dublin area.
        # Update according to the Lithuanian NUTS2 zones, shifting from 1 to 2 zones in 2016.
        capacity_potential_ds.at['LT01'] = 0.  # Region of Vilnius.
        capacity_potential_ds.at['LT02'] = capacity_potential_ds.at['LT00']
        # Update according to the Scottish NUTS2 zones in 2016.
        capacity_potential_ds.at['UKM8'] = 0.  # Glasgow area.
        capacity_potential_ds.at['UKM9'] = capacity_potential_ds.at['UKM3']
        # Update according to the Warsaw enclave.
        capacity_potential_ds.at['PL92'] = capacity_potential_ds.at['PL9']
        capacity_potential_ds.at['PL91'] = 0.  # Inner city of Warsaw.
        # Update according to the Budapest split in Budapest (enclave city) and Pest (the region).
        capacity_potential_ds.at['HU11'] = 0.  # Inner city of Budapest.
        capacity_potential_ds.at['HU12'] = capacity_potential_ds.at['HU10']
        # Inner London
        capacity_potential_ds.at['UKI5'] = 0.
        capacity_potential_ds.at['UKI6'] = 0.
        capacity_potential_ds.at['UKI7'] = 0.

    elif tech == 'pv_residential':

        # Update according to the Irish NUTS2 zones, shifting from 2 to 3 zones in 2014.
        capacity_potential_ds.at['IE04'] = capacity_potential_ds.at['IE01']
        capacity_potential_ds.at['IE05'] = capacity_potential_ds.at['IE02']*(1/3)
        # Dublin region, two thirds of population.
        capacity_potential_ds.at['IE06'] = capacity_potential_ds.at['IE02']*(2/3)
        # Update according to the Lithuanian NUTS2 zones, shifting from 1 to 2 zones in 2016.
        # Capital region, one third of population.
        capacity_potential_ds.at['LT01'] = capacity_potential_ds.at['LT00']*(1/3)
        capacity_potential_ds.at['LT02'] = capacity_potential_ds.at['LT00']*(2/3)  # Rest of the country.
        # Update according to the Scottish NUTS2 zones in 2016.
        # Glasgow region, split based on population share.
        capacity_potential_ds.at['UKM8'] = capacity_potential_ds.at['UKM3']*(1/3)
        capacity_potential_ds.at['UKM9'] = capacity_potential_ds.at['UKM3']*(2/3)
        # Update according to the Warsaw enclave.
        # Warsaw region, split based on population share.
        capacity_potential_ds.at['PL92'] = capacity_potential_ds.at['PL9']*(1/2)
        capacity_potential_ds.at['PL91'] = capacity_potential_ds.at['PL9']*(1/2)
        # Update according to the Budapest split in Budapest (enclave city) and Pest (the region).
        capacity_potential_ds.at['HU11'] = capacity_potential_ds.at['HU10']*(1/2)
        capacity_potential_ds.at['HU12'] = capacity_potential_ds.at['HU10']*(1/2)
        # Outer London. Values not assigned within ENSPRESO.
        capacity_potential_ds.at['UKI5'] = 0.
        capacity_potential_ds.at['UKI6'] = 0.
        capacity_potential_ds.at['UKI7'] = 0.

    elif tech == 'pv_utility':

        # Update according of the Irish NUTS2 zones, shifting from 2 to 3 zones in 2014.
        capacity_potential_ds.at['IE04'] = capacity_potential_ds.at['IE01']
        capacity_potential_ds.at['IE05'] = capacity_potential_ds.at['IE02']
        capacity_potential_ds.at['IE06'] = 0.  # Dublin city area.
        # Update according of the Lithuanian NUTS2 zones, shifting from 1 to 2 zones in 2016.
        capacity_potential_ds.at['LT01'] = 0.  # Capital region.
        capacity_potential_ds.at['LT02'] = capacity_potential_ds.at['LT00']  # Rest of the country.
        # Update according to the Scottish NUTS2 zones in 2016.
        capacity_potential_ds.at['UKM8'] = 0.  # Glasgow area.
        capacity_potential_ds.at['UKM9'] = capacity_potential_ds.at['UKM3']
        # Update according to the Warsaw enclave.
        capacity_potential_ds.at['PL92'] = capacity_potential_ds.at['PL9']
        capacity_potential_ds.at['PL91'] = 0.  # Inner city of Warsaw.
        # Update according to the Budapest split in Budapest (enclave city) and Pest (the region).
        capacity_potential_ds.at['HU11'] = 0.  # Inner city of Budapest.
        capacity_potential_ds.at['HU12'] = capacity_potential_ds.at['HU10']
        # Inner London.
        capacity_potential_ds.at['UKI5'] = 0.
        capacity_potential_ds.at['UKI6'] = 0.
        capacity_potential_ds.at['UKI7'] = 0.

    # TODO: does it make sense to have two copies of the same data points? Wouldn't it create problems?
    elif tech == 'wind_offshore':

        capacity_potential_ds.at['EZUK'] = capacity_potential_ds.at['EZGB']
        capacity_potential_ds.at['EZEL'] = capacity_potential_ds.at['EZGR']

    elif tech == 'wind_floating':

        capacity_potential_ds.at['EZUK'] = capacity_potential_ds.at['EZGB']
        capacity_potential_ds.at['EZEL'] = capacity_potential_ds.at['EZGR']

    # Remove outdated regions
    regions_to_remove = ['AD00', 'SM00', 'CY00', 'LI00', 'FRY1', 'FRY2', 'FRY3', 'FRY4',
                         'FRY5', 'ES63', 'ES64', 'ES70', 'HU10', 'IE01', 'IE02', 'LT00', 'UKM3']

    capacity_potential_ds = capacity_potential_ds.drop(regions_to_remove, errors='ignore')

    return capacity_potential_ds


def get_capacity_potential_from_enspreso(tech: str) -> pd.Series:
    """
    Returning capacity potential (in GW) per NUTS2 region for a given technology, based on the ENSPRESO dataset.

    Parameters
    ----------
    tech : str
        Technology name among 'wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility' and 'pv_residential'

    Returns
    -------
    nuts2_capacity_potentials: pd.Series
        Series storing technical potential per NUTS2 region.
    """
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/source/ENSPRESO')
    # For wind, summing over all wind conditions is similar to considering taking all available land and a capacity per
    #  area of 5MW/km2
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

    updated_potential_per_tech = update_enspreso_capacity_potential(nuts2_capacity_potentials_ds, tech).round(6)

    return updated_potential_per_tech


def group_potentials_per_country(potential_ds: pd.Series) -> pd.Series:

    potential_ds = potential_ds.to_frame()
    potential_ds.columns = ['value']

    potential_ds['country'] = potential_ds.index.str[:2]

    potential_per_country = potential_ds.groupby(['country'])['value'].sum()

    return potential_per_country


# TODO: why is this taking topology argument either? Shouldn't take any argument to me, just generate for NUTS0
def built_capacity_potential_files(topology: str):
    """Saves capacity potentials (in GW) for NUTS2 (2016 version) and EEZ regions."""

    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/generated/')

    techs_offshore = ['wind_offshore', 'wind_floating']
    capacities_offshore = pd.DataFrame(columns=techs_offshore)

    for tech in capacities_offshore.columns:
        non_eu28_potentials = get_non_eu28_potential(tech)
        eu28_potentials = get_capacity_potential_from_enspreso(tech)
        capacity_potential = pd.concat([non_eu28_potentials, eu28_potentials], axis=0).sort_index()
        capacities_offshore[tech] = capacity_potential

    capacities_offshore.to_csv(path_potential_data + "eez_capacity_potentials_GW.csv")

    techs_onshore = ["pv_residential", "pv_utility", "wind_onshore"]
    capacities_onshore = pd.DataFrame(columns=techs_onshore)

    if topology == 'ehighway':

        for tech in capacities_onshore.columns:
            non_eu28_potentials = get_non_eu28_potential(tech)
            eu28_potentials = get_capacity_potential_from_enspreso(tech)
            capacity_potential = pd.concat([non_eu28_potentials, eu28_potentials], axis=0).sort_index()
            capacities_onshore[tech] = capacity_potential

        capacities_onshore.to_csv(path_potential_data + "nuts2_capacity_potentials_GW.csv")

    else:  # topology == 'countries'

        for tech in capacities_onshore.columns:
            non_eu28_potentials = group_potentials_per_country(get_non_eu28_potential(tech))
            eu28_potentials = group_potentials_per_country(get_capacity_potential_from_enspreso(tech))
            capacity_potential = pd.concat([non_eu28_potentials, eu28_potentials], axis=0).sort_index()
            capacities_onshore[tech] = capacity_potential

        capacities_onshore.to_csv(path_potential_data + "nuts0_capacity_potentials_GW.csv")


if __name__ == '__main__':

    # TODO: maybe do something cleaner here..
    topology = 'countries'

    built_capacity_potential_files(topology=topology)
