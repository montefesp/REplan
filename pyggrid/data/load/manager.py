from typing import List
from os import listdir

import pandas as pd

from pyggrid.data.geographics import get_subregions, get_nuts_area

from pyggrid.data import data_path


def get_yearly_country_load(country: str, year: int) -> int:
    """
    Retrieve the yearly load (in GWh) of country for a specific year.

    Parameters
    ----------
    country: str
        ISO code of a country
    year: int
        Year for which we want load

    Returns
    -------
    int
        Yearly load (in GWh)
    """

    iea_load_dir = f"{data_path}load/source/iea/"
    available_countries = [c.strip(".csv") for c in listdir(iea_load_dir) if c.endswith(".csv")]
    assert country in available_countries, f"Error: Data is not available for country {country}." \
                                           f"Please download data."

    yearly_load_fn = f"{iea_load_dir}{country}.csv"
    yearly_load_ds = pd.read_csv(yearly_load_fn, index_col=0, squeeze=True)
    assert year in yearly_load_ds.index, f"Error: Data for year {year} is not available for country {country}"

    return yearly_load_ds.loc[year]


def get_load(timestamps: pd.DatetimeIndex = None, years_range: List[int] = None,
             countries: List[str] = None, regions: List[str] = None, missing_data: str = "error") -> pd.DataFrame:
    """
    Compute hourly load time series (in GWh) for given countries or regions.

    The desired time slice can be either given as as series of time stamps or
    as a range of years for which we want full data.

    Parameters
    ----------
    timestamps: pd.DatetimeIndex (default: None)
        Datetime index
    years_range: List[int]
        Range of years (if you desire to obtain data for only one year just pass a list with twice the year)
    countries: List[str] (default: None)
        ISO codes of countries
    regions: List[str] (default: None)
        List of codes referring to regions made of several countries defined in 'data_path'/geographics/region_definition.csv
    missing_data: str (default: error)
        Defines how to deal with missing data. If value is 'error', throws an error. If value is 'interpolate', uses
        data from another country

    Returns
    -------
    pd.DataFrame (index = timestamps, columns = regions or countries)
        DataFrame associating to each country or region the corresponding its hourly load (in GWh)

    """

    assert (countries is None) != (regions is None), "Error: You must either specify a list of countries or " \
                                                     "a list of regions made of countries, but not both."
    assert (timestamps is None) != (years_range is None), "Error: You must either specify a range of years or " \
                                                          "a series of time stamps, but not both."
    assert years_range is None or (len(years_range) == 2 and years_range[0] <= years_range[1]), \
        f"The desired years range must be specified as a list of two ints (the first one being smaller" \
        f" or equal to the second one, received {years_range}"
    assert missing_data in ["error", "interpolate"], f"Error: missing_data must be one of 'error' or 'interpolate'"

    if years_range is not None:
        timestamps = pd.date_range(f"{years_range[0]}-01-01 00:00:00", f"{years_range[1]}-12-31 23:00:00", freq='1H')

    opsd_load_fn = f"{data_path}load/generated/opsd_load.csv"
    load = pd.read_csv(opsd_load_fn, index_col=0, engine='python')
    load.index = pd.DatetimeIndex(load.index)
    missing_timestamps = set(timestamps) - set(load.index)
    assert not missing_timestamps, f"Error: Load is not available " \
                                   f"for the following timestamps {sorted(list(missing_timestamps))}"

    # Slice on time and remove columns in which we don't have available data for the full time period
    load = load.loc[timestamps].dropna(axis=1)
    # Convert to GWh
    load = load * 1e-3

    def get_countries_load(countries_: List[str]):
        countries_load = pd.DataFrame(index=timestamps, columns=countries_)
        missing_countries = set(countries_) - set(load.columns)
        if missing_countries:
            if missing_data == "error":
                raise ValueError(f"Error: Load is not available for countries {sorted(list(missing_countries))} "
                                 f"for the required timestamps.")
            else:
                countries_load[list(missing_countries)] = \
                    get_load_from_source_country(list(missing_countries), load.index)
        countries_with_data = list(set(countries) - set(missing_countries))
        countries_load[countries_with_data] = load[countries_with_data]
        return countries_load

    # Get load per country
    if countries is not None:
        return get_countries_load(countries).round(6)
    # Get load aggregated by region
    elif regions is not None:
        load_per_region = pd.DataFrame(columns=regions, index=timestamps)
        for region in regions:
            # Get load date for all subregions and sum it
            countries = get_subregions(region)
            load_per_region[region] = get_countries_load(countries).sum(axis=1).values

        return load_per_region.round(6)


def get_load_from_source_country(target_countries: List[str], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute load for a list of countries.

    The load is obtained by retrieving the load for another country
    (listed in 'data_path'/load/source_load_countries.csv) and then resizing it
    proportionally to yearly load of the source and target countries.

    Parameters
    ----------
    target_countries: List[str]
        List of countries for which we want to obtain load
    timestamps: pd.DatetimeIndex
        List of time stamps

    Returns
    -------
    target_countries_load: pd.DataFrame
        Approximated load

    """

    opsd_load_fn = f"{data_path}load/generated/opsd_load.csv"
    load = pd.read_csv(opsd_load_fn, index_col=0, engine='python')
    load.index = pd.DatetimeIndex(load.index)
    load = load*1e-3
    load = load.round(6)

    years_range = range(timestamps[0].year, timestamps[-1].year+1)
    assert years_range[0] >= 2015 and years_range[-1] <= 2018, \
        "Error: This function only works for year 2015 to 2018"

    # Load the file indicating if load information is available for the given regions
    load_info_fn = f"{data_path}load/source_load_countries.csv"
    load_info = pd.read_csv(load_info_fn, index_col="Code")
    load_info = load_info.dropna()

    missing_countries = set(target_countries) - set(load_info.index)
    assert not missing_countries, f"Error: These target countries are not associated " \
                                  f"to any source country {sorted(list(missing_countries))}"

    # Source countries might change from year to year (depending on data availability)
    target_countries_load = pd.DataFrame(index=timestamps, columns=target_countries, dtype=float)
    for year in years_range:

        year_timestamps = timestamps[timestamps.year == year]

        # Get source countries for that year
        source_countries = load_info[str(year)].loc[target_countries].values
        # Get the load for the source countries
        source_countries_load = load.loc[year_timestamps, list(set(source_countries))]

        # Get the yearly load of the source regions
        yearly_load_source_countries = dict.fromkeys(source_countries)
        for c in source_countries:
            yearly_load_source_countries[c] = get_yearly_country_load(c, year)

        # Get the yearly load of the target regions
        yearly_load_target_countries = dict.fromkeys(target_countries)
        for c in target_countries:
            yearly_load_target_countries[c] = get_yearly_country_load(c, year)

        # Compute target load
        for i, target_c in enumerate(target_countries_load):
            source_c = source_countries[i]
            source_c_load = source_countries_load[source_c]
            yearly_load_source_country = yearly_load_source_countries[source_c]
            target_countries_load.loc[year_timestamps, target_c] = \
                source_c_load * yearly_load_target_countries[target_c] / yearly_load_source_country

    return target_countries_load


def get_load_from_nuts_codes(nuts_codes_lists: List[List[str]], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute the aggregated load in GWh for groups of NUTS regions.

    The load for each NUTS region is computed by downscaling the corresponding
    country load proportionally to population.

    Parameters
    ----------
    nuts_codes_lists: List[List[str]]
        List of lists of nuts codes (version 2016)
    timestamps: pd.DatetimeIndex
        List of time stamps

    Returns
    -------
    load: pd.DataFrame
        DataFrame with aggregated hourly load in GWh (columns are numbered from 0 to the number of groups minus one)
    """

    assert isinstance(nuts_codes_lists, list) and len(nuts_codes_lists) != 0, \
        "Error: 'nuts_codes_lists' must be a list of lists"
    for i, nuts_codes in enumerate(nuts_codes_lists):
        assert isinstance(nuts_codes, list), "Error: 'nuts_codes_lists' must be a list of lists"
        assert len(nuts_codes) != 0, f"Error: NUTS codes list number {i} is empty"
    pop_dens_fn = f"{data_path}geographics/source/eurostat/demo_r_d3dens.xls"
    pop_dens = pd.read_excel(pop_dens_fn, header=8, index_col=0)[:2213]
    pop_dens.index.name = 'code'

    area = get_nuts_area()
    area.index.name = 'code'

    # gdp_fn = f"{data_path}geographics/source/eurostat/nama_10r_3gdp.xls"
    # gdp = pd.read_excel(gdp_fn, header=8, index_col=0)[:1945]
    # gdp.index.name = 'code'

    # Check that we have data for all required NUTS regions
    all_nuts_codes = set.union(*[set(region_list) for region_list in nuts_codes_lists])
    area_missing_countries = all_nuts_codes - set(area.index)
    assert not area_missing_countries, f"Error: Area is not available for following " \
                                       f"NUTS regions: {sorted(list(area_missing_countries))}"

    pop_missing_countries = all_nuts_codes - set(pop_dens.index)
    assert not pop_missing_countries, f"Error: Population density is not available for following " \
                                      f"NUTS regions: {sorted(list(pop_missing_countries))}"

    # Get load at country level
    countries = set([code[:2] for code in all_nuts_codes])
    # Replace UK and EL
    nuts_to_iso = {"UK": "GB", "EL": "GR"}
    countries = [nuts_to_iso[r] if r in nuts_to_iso else r for r in countries]
    countries_load = get_load(timestamps=timestamps, countries=countries, missing_data='interpolate')

    # Convert to NUTS level
    total_country_people = pd.Series(0., index=countries, dtype=float)
    load = pd.DataFrame(0., index=timestamps, columns=range(len(nuts_codes_lists)))
    for idx, nuts_codes_list in enumerate(nuts_codes_lists):
        for nuts_code in nuts_codes_list:

            # Get the load of the country in which the NUTS region resides
            nuts_country_code = nuts_code[:2]
            country_code = \
                nuts_to_iso[nuts_country_code] if nuts_country_code in nuts_to_iso else nuts_country_code
            country_load = countries_load[country_code]

            # Down scale country load to the regions of interest by using population and gdp
            regions_pop_dens = pop_dens.loc[nuts_code]['2016']
            regions_area = area.loc[nuts_code]['2016']
            # regions_gdp = gdp.loc[region_codes_list]
            country_pop_dens = pop_dens.loc[nuts_country_code]['2016']
            country_area = area.loc[nuts_country_code]['2016']
            # country_gdp = gdp.loc[country_code]

            total_country_people[country_code] += regions_area*regions_pop_dens

            load[idx] += country_load * regions_pop_dens * regions_area / (country_pop_dens * country_area)

    return load
