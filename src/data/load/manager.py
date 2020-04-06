from typing import List
from os.path import join, dirname, abspath

import pandas as pd

from src.data.geographics.manager import get_subregions, get_nuts_area


def get_yearly_country_load(country: str, year: int = 2016) -> int:
    """
    Returns the yearly load of country for a specific year

    Parameters
    ----------
    country: str
        Code of the country for which we want load
    year: int (default: 2016)
        Year for which we want load

    Returns
    -------
    Yearly load: int
    """

    key_ind_fn = join(dirname(abspath(__file__)), "../../../data/key_indicators/generated/" + country + ".csv")
    return pd.read_csv(key_ind_fn, index_col=0).loc[year, "Electricity consumption (TWh)"]


def get_load(timestamps: pd.DatetimeIndex = None, years_range: List[int] = None,
             countries: List[str] = None, regions: List[str] = None, missing_data: str = "error") -> pd.DataFrame:
    """
    Returns hourly load time series (in GWh) for given countries or regions.
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
        List of codes referring to regions made of several countries defined in data/region_definition.csv
    missing_data: str (default: error)
        Defines how to deal with missing data. If value is 'error', throws an error. If value is 'interpolate', uses
        data from another country

    Returns
    -------
    pd.DataFrame (index = timestamps, columns = regions or countries)
        DataFrame associating to each country or region the corresponding load data (in GWh)

    """

    assert (countries is None) != (regions is None), "Error: You must either specify a list of countries or " \
                                                     "a list of regions made of countries, but not both."
    assert (timestamps is None) != (years_range is None), "Error: You must either specify a range of years or " \
                                                          "a series of time stamps, but not both."
    assert years_range is None or len(years_range) == 2, \
        f"The desired years range must be specified as a list of two ints, received {years_range}"
    assert missing_data in ["error", "interpolate"], f"Error: missing_data must be one of 'error' or 'interpolate'"

    if years_range is not None:
        timestamps = pd.date_range(f"{years_range[0]}-01-01 00:00:00", f"{years_range[1]}-12-31 00:00:00", freq='1H')

    opsd_load_fn = join(dirname(abspath(__file__)), "../../../data/load/generated/opsd_load.csv")
    load = pd.read_csv(opsd_load_fn, index_col=0)
    load.index = pd.DatetimeIndex(load.index)

    # Slice on time and remove columns in which we don't have available data for the full time period
    load = load.loc[timestamps].dropna(axis=1)
    # Convert to GWh
    load = load * 1e-3
    # Round to kWh
    load = load.round(6)

    def get_countries_load(countries: List[str]):
        countries_load = pd.DataFrame(index=timestamps, columns=countries)
        missing_countries = set(countries) - set(load.columns)
        if missing_countries:
            if missing_data == "error":
                raise ValueError(f"Error: Load is not available for countries {sorted(list(missing_countries))}")
            else:
                countries_load[list(missing_countries)] = \
                    get_load_from_source_country(list(missing_countries), load.index)
        countries_with_data = list(set(countries) - set(missing_countries))
        countries_load[countries_with_data] = load[countries_with_data]
        return countries_load

    # Get load per country
    if countries is not None:
        return get_countries_load(countries)
    # Get load aggregated by region
    else:  # regions is not None
        load_per_region = pd.DataFrame(columns=regions, index=timestamps)
        for region in regions:
            # Get load date for all subregions and sum it
            countries = get_subregions(region)
            load_per_region[region] = get_countries_load(countries).sum(axis=1).values

        return load_per_region


def get_load_from_source_country(target_countries: List[str], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns load for a list of countries. The load is obtained by retrieving the load for another country (listed
    in data/load/source_load_countries.csv and then resizing it proportionally to yearly load of the source
    and target countries.

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

    opsd_load_fn = join(dirname(abspath(__file__)), "../../../data/load/generated/opsd_load.csv")
    load = pd.read_csv(opsd_load_fn, index_col=0)
    load.index = pd.DatetimeIndex(load.index)
    load = load*1e-3
    load = load.round(6)

    years_range = range(timestamps[0].year, timestamps[-1].year+1)
    assert years_range[0] >= 2015 and years_range[-1] <= 2018, \
        "Error: This function only works for year 2015 to 2018"

    # Load the file indicating if load information is available for the given regions
    load_info_fn = join(dirname(abspath(__file__)), "../../../data/load/source_load_countries.csv")
    load_info = pd.read_csv(load_info_fn, index_col="Code")
    load_info = load_info.dropna()

    # Source countries might change from year to year (depending on data availability)
    target_countries_load = pd.DataFrame(index=timestamps, columns=target_countries)
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
            target_countries_load.loc[year_timestamps, target_c] = \
                source_countries_load[source_c] \
                * yearly_load_target_countries[target_c] / yearly_load_source_countries[source_c]

    return target_countries_load


def get_load_from_nuts_codes(nuts_codes_lists: List[List[str]], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns the aggregated load in GWh for groups of NUTS regions.
    The load for each NUTS region is computed by downscaling the corresponding
    country load proportionally to population.

    Parameters
    ----------
    nuts_codes_lists: List[List[str]]
        List of list of nuts codes. A load series will be returned for each list of regions.
    timestamps: pd.DatetimeIndex
        List of time stamps

    Returns
    -------
    load: pd.DataFrame
        DataFrame with aggregated load in GWh (columns are numbers from 0 to the number of groups minus one)
    """

    pop_dens_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/demo_r_d3dens.xls")
    pop_dens = pd.read_excel(pop_dens_fn, header=8, index_col=0)[:2213]
    pop_dens.index.name = 'code'

    area = get_nuts_area()
    area.index.name = 'code'

    gdp_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/nama_10r_3gdp.xls")
    gdp = pd.read_excel(gdp_fn, header=8, index_col=0)[:1945]
    gdp.index.name = 'code'

    # Get load at country level
    all_countries = set.union(*[set([region[:2] for region in region_list]) for region_list in nuts_codes_lists])
    # Replace UK and EL
    nuts0_problems = {"UK": "GB", "EL": "GR"}
    all_countries = [nuts0_problems[r] if r in nuts0_problems else r for r in all_countries]
    countries_load = get_load(timestamps=timestamps, countries=all_countries, missing_data='interpolate')

    # Convert to NUTS level
    load = pd.DataFrame(0., index=timestamps, columns=range(len(nuts_codes_lists)))
    for idx, nuts_codes_list in enumerate(nuts_codes_lists):
        for nuts_code in nuts_codes_list:

            # Get the load of the country in which the NUTS region resides
            nuts_country_code = nuts_code[:2]
            country_code = \
                nuts0_problems[nuts_country_code] if nuts_country_code in nuts0_problems else nuts_country_code
            country_load = countries_load[country_code]

            # Down scale country load to the regions of interest by using population and gdp
            # TODO: doing the thing only on population for now but need to update that
            #  -> would be nice if we had yearly load
            regions_pop_dens = pop_dens.loc[nuts_code]['2016']
            regions_area = area.loc[nuts_code]['2016']
            # regions_gdp = gdp.loc[region_codes_list]
            country_pop_dens = pop_dens.loc[nuts_country_code]['2016']
            country_area = area.loc[nuts_country_code]['2016']
            # country_gdp = gdp.loc[country_code]

            load[idx] += country_load * regions_pop_dens * regions_area / (country_pop_dens * country_area)

    return load


# def get_prepared_load(timestamps: pd.DatetimeIndex = None, countries: List[str] = None,
#                       regions: List[str] = None) -> pd.DataFrame:
#     """
#     Returns hourly load time series (in GWh) for given regions and time horizon.
#
#     Parameters
#     ----------
#     timestamps: pd.DatetimeIndex (default: None)
#         Datetime index
#     countries: List[str] (default: None)
#         ISO codes of countries
#     regions: List[str] (default: None)
#         List of codes referring to regions made of several countries defined in data/region_definition.csv
#
#     Returns
#     -------
#     load_per_region: pd.DataFrame (index = timestamps, columns = regions)
#         DataFrame associating to each region code the corresponding load data (in GWh)
#
#     """
#
#     assert bool(countries) != bool(regions), "Error: You must either specify a list of countries or " \
#                                              "a list of regions made of countries, but not both."
#
#     assert timestamps is None or timestamps[0].year >= 2015 and timestamps[-1].year <= 2018, \
#         "Error: Data is only available from 2015 to 2018"
#
#     years_range = "2015_2018" if timestamps is None or timestamps[0].year == 2015 else "2016_2018"
#
#     opsd_load_fn = join(dirname(abspath(__file__)), f'../../../data/load/generated/load_opsd_{years_range}.csv')
#     load = pd.read_csv(opsd_load_fn, index_col=0, low_memory=False, date_parser=pd.to_datetime)
#
#     # Slice data on time if needed
#     if timestamps is not None:
#         missing_load = set(timestamps) - set(load.index)
#         assert not missing_load, f"Error: Load is not available for time-stamps {missing_load}"
#         load = load.loc[timestamps]
#
#     if countries is not None:
#         missing_countries = set(countries) - set(load.columns)
#         assert not missing_countries, f"Error: Load is not available for countries {sorted(list(missing_countries))}"
#         return load[countries]*1e-3
#     else:  # regions is not None
#
#         load_per_region = pd.DataFrame(columns=regions, index=timestamps)
#         for region in regions:
#             # Get load date for all subregions, sum it and transform to GWh
#             countries = get_subregions(region)
#             missing_countries = set(countries) - set(load.columns)
#             assert not missing_countries, f"Error: Load is not available for countries " \
#                                           f"{sorted(list(missing_countries))} in region {region}"
#             load_per_region[region] = load[countries].sum(axis=1).values * 1e-3
#
#         return load_per_region

# def get_countries_load(countries: List[str], nb_years: int = 1,
#                        days_range_start: datetime.date = datetime.date(1, 1, 1),
#                        days_range_end: datetime.date = datetime.date(1, 12, 31),
#                        years: List[int] = None):
#     """
#     Returns a pd.Dataframe where the columns corresponds the load time-series (in MWh) for a series of countries.
#     The time period for which the load is returned can be specified in various ways. The default behavior is to
#     return the most recent year of data available for each countries. The load for a specific set of years can
#     also be asked for. Note that data for a given year can be absent for some countries and will therefore lead
#     to an error when asked for. Finally, a specific set of days in a year can be asked for (e.g. 100 first days
#     of the year). In that case, either a specific year can be asked for through the 'years' argument or the most
#     recent year will be used.
#
#     Parameters
#     ----------
#     countries: List[str]
#         List of country ISO codes
#     nb_years: int (default: 1)
#         Number of years for which we want load
#     days_range_start: datetime.date (default: datetime.date(1, 1, 31))
#         Date specifying the first day of the year for which we want load. Note that the year that is specified in that
#         date is not important.
#     days_range_end: datetime.date (default: datetime.date(1, 12, 31))
#         Date specifying the last day of the year for which we want load. Note that the year that is specified in that
#         date is not important.
#     years: List[int] ODO: to implement
#         List of years for which we want data
#
#     Returns
#     -------
#     load_data_time_sliced: pd.DataFrame
#         DataFrame containing the load for each source country and indexed with integers
#         from 0 to nb_years*((days_range_end-days_range_start).days+1)*24 and where
#     """
#
#     # ODO: if years is specified should take the precedence over nb_years
#
#     assert days_range_start.month < days_range_end.month or \
#         (days_range_start.month == days_range_end.month and days_range_start.day <= days_range_end.day), \
#         "ERROR: The days_range_start month-day pair must happen before the days_range_end month-day pair."
#
#     load_data_fn = join(dirname(abspath(__file__)), "../../../data/load/generated/opsd_load.csv")
#     load_data = pd.read_csv(load_data_fn, index_col=0)
#     load_data.index = pd.DatetimeIndex(load_data.index)
#
#     # Check data is available for those countries
#     missing_countries = set(countries) - set(load_data.keys())
#     assert not missing_countries, f"No data present for the following codes: {missing_countries}"
#
#     load_data = load_data[countries]
#
#     # Select time period
#     if years is None:
#
#         # TDO: need to take implement a way to take into account leap years... -
#         #  probably just remove the addtional day
#
#         load_data_time_sliced = pd.DataFrame(index=range(nb_years*((days_range_end-days_range_start).days+1)*24),
#                                              columns=load_data.keys())
#         for code in load_data.keys():
#             load_data_for_code = load_data[code].dropna()
#
#             # Get the most recent year of data that has up to the day we need
#             end_year = (load_data_for_code.last_valid_index() -
#                         datetime.timedelta(
#                             days=(days_range_end-datetime.date(days_range_end.year, 1, 1)).days, hours=23)).year
#             end_date = datetime.datetime(year=end_year, month=days_range_end.month, day=days_range_end.day, hour=23)
#
#             # Check then if we have enough years
#             start_year = end_year-nb_years+1
#             start_date = datetime.datetime(year=start_year, month=days_range_start.month, day=days_range_start.day)
#
#             assert start_date in load_data_for_code.index, f"There is no sufficient data for region of code {code}"
#             load_data_time_sliced[code] = load_data_for_code.loc[start_date:end_date].values
#
#         return load_data_time_sliced
#
#     else:
#
#         # Detect number of leap years
#         nb_leap_years = sum([1 for year in years if year % 4 == 0 or year % 100 == 0])
#
#         # ODO: this is ugly
#         index = pd.date_range(start=datetime.datetime(year=years[0],
#         month=days_range_start.month, day=days_range_start.day),
#                   end=datetime.datetime(year=years[-1], month=days_range_end.month,
#                   day=days_range_end.day, hour=23), freq='1H')
#
#         load_data_time_sliced = \
#             pd.DataFrame(index=index,
#                          columns=load_data.keys())
#
#         sorted_years = sorted(years)
#         for code in load_data.keys():
#             load_data_for_code = load_data[code].dropna()
#
#             # Check if we have data for the first required year and last required year
#             first_date = datetime.datetime(year=sorted_years[0],
#             month=days_range_start.month, day=days_range_start.day)
#             assert first_date in load_data_for_code.index, \
#                 f"There is no sufficient data for the date {first_date} for region of code {code}"
#             last_date = datetime.datetime(year=sorted_years[-1], month=days_range_end.month, day=days_range_end.day)
#             assert last_date in load_data_for_code.index, \
#                 f"There is no sufficient data for the date {last_date} for region of code {code}"
#
#             # Add the time-series for each year
#             load = np.array([])
#             for year in years:
#                 start_date = datetime.datetime(year=year, month=days_range_start.month, day=days_range_start.day)
#                 end_date = datetime.datetime(year=year, month=days_range_end.month, day=days_range_end.day, hour=23)
#                 load = np.append(load, load_data_for_code.loc[start_date:end_date].values)
#             load_data_time_sliced[code] = load
#
#         return load_data_time_sliced


if __name__ == '__main__':
    ts = pd.date_range('2015-01-01T00:00', '2016-12-31T23:00', freq='1H')
    print(get_load(timestamps=ts, countries=["BE", "UA"], missing_data='interpolate'))
    print(get_load(timestamps=ts, regions=["EU"], missing_data='interpolate'))
