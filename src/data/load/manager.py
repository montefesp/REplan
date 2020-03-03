import datetime
import pandas as pd
import numpy as np
from typing import *
import os
import shapely
from shapely.geometry import Point
from scipy import stats
import matplotlib.pyplot as plt
from src.data.geographics.manager import get_subregions, get_nuts_area

# --- Data pre-processing --- #


def filter_outliers(time_series: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a time-series and removes its outliers by looking at monthly patterns

    Parameters
    ----------
    time_series: pd.DataFrame
        Dataframe with datetime index and one column containing a time-series with a value for each time-stamp (which
        can be NaN)

    Returns
    -------
    time_series: pd.DataFrame
        Corrected dataframe
    """

    # Removing outliers using monthly patterns
    max_z = 3
    for month in range(1, 13):
        monthly_time_series = time_series[time_series.index.month == month]
        outliers = np.abs(stats.zscore(monthly_time_series.dropna())) > max_z
        time_series.loc[monthly_time_series.dropna()[outliers].index] = np.nan

    return time_series


def fill_gaps(time_series):
    """
    Takes in a time-series and fills gaps (i.e. value equals to NaN) inside the time-series (leading and ending missing
    values are untouched)

    Parameters
    ----------
    time_series: pd.DataFrame
        Dataframe with datetime index and one column containing a time-series with a value for each time-stamp (which
        can be NaN)

    Returns
    -------
    time_series: pd.DataFrame
        Corrected dataframe
    """

    # First remove starting and ending nans
    time_series_trim = time_series.loc[time_series.first_valid_index():time_series.last_valid_index()]

    # For each remaining nan, we replace its value by the value of an identical hour in another day for which we have
    # data
    time_series_trim_valid = time_series_trim.dropna()
    nan_indexes = time_series_trim.index[time_series_trim.apply(np.isnan)]
    for index in nan_indexes:
        # Get all elements which have are on the same day, same hour
        similar_hours = time_series_trim_valid[time_series_trim_valid.index.map(lambda x: x.weekday() == index.weekday()
                                                                                and x.hour == index.hour)]
        # Find closest valid hour
        closest_valid_hour_index = similar_hours.index[np.argmin(abs((similar_hours.index - index).days))]

        time_series_trim[index] = time_series_trim_valid[closest_valid_hour_index]

    time_series[time_series_trim.index] = time_series_trim.values

    return time_series


def correct_time_series(time_series: pd.DataFrame, plot: bool = False) \
        -> pd.DataFrame:
    """
    Takes in a time series and improves it by removing outliers and filling gaps via interpolation

    Parameters
    ----------
    time_series: pd.DataFrame
        Dataframe with datetime index and one column containing a time-series with a value for each time-stamp (which
        can be NaN)
    plot: bool (default: False)
        Whether to plot or not

    Returns
    -------
    time_series: pd.DataFrame
        Corrected time-series
    """

    if plot:
        plt.plot(time_series.values)

    time_series = filter_outliers(time_series)
    if plot:
        plt.plot(time_series.values)

    time_series = fill_gaps(time_series)
    if plot:
        plt.plot(time_series.values, alpha=0.5)
        plt.show()

    return time_series


def preprocess():
    """
    This function builds a complete and easy to use database of load by formating and correcting some open-source data
    contained in the folder data/load/source.
    The results are saved in the folder data/load/generated.
    """

    source_data_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "../../../data/load/source/time_series_60min_singleindex_filtered.csv")
    source_data = pd.read_csv(source_data_fn, index_col='utc_timestamp')
    source_data = source_data.drop(["cet_cest_timestamp"], axis=1)
    source_data = source_data[1:-23]

    # Selecting first the keys from entsoe transparency platform
    all_keys = [key for key in source_data.keys() if 'load_actual_entsoe_transparency' in key]

    # Adding then the keys from power statistics corresponding to missing countries
    all_keys_short = [key.split("_load")[0] for key in all_keys]
    all_keys += [key for key in source_data.keys()
                 if 'load_actual_entsoe_power' in key and key.split("_load")[0] not in all_keys_short]

    # Finally add the keys from tso corresponding to other missing countries
    all_keys_short = [key.split("_load")[0] for key in all_keys]
    all_keys += [key for key in source_data.keys()
                 if 'load_actual_tso' in key and key.split("_load")[0] not in all_keys_short]

    final_data = source_data[all_keys]
    final_data.columns = [key.split("_load")[0] for key in final_data.keys()]

    # Remove some shitty data by inspection
    final_data = final_data.drop(["CS", "IE_sem", "DE_LU", "GB_NIR", "GB_UKM"], axis=1)

    # Change GB_GBN to GB
    final_data = final_data.rename(columns={"GB_GBN": "GB"})

    # Change index to pandas.DatetimeIndex
    final_data.index = pd.DatetimeIndex(final_data.index)

    final_data = final_data.reindex(sorted(final_data.columns), axis=1)

    # Correct time series by removing outliers and filling gaps
    for key in final_data.keys():
        print(key)
        final_data[key] = correct_time_series(final_data[key])

    final_data_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../../../data/load/generated/opsd_load.csv")
    final_data.to_csv(final_data_fn)


# --- Accessing data --- #

def get_countries_load(country_codes: List[str], nb_years: int = 1,
                       days_range_start: datetime.date = datetime.date(1, 1, 1),
                       days_range_end: datetime.date = datetime.date(1, 12, 31),
                       years: List[int] = None):
    """
    Returns a pd.Dataframe where the columns corresponds the load time-series (in GWh) for a series of countries.
    The time periodfor which the load is returned can be specified in various ways. The default behavior is to
    return the most recent year of data available for each countries. The load for a specific set of years can
    also be asked for. Note that data for a given year can be absent for some countries and will therefore lead
    to an error when asked for. Finally, a specific set of days in a year can be asked for (e.g. 100 first days
    of the year). In that case, either a specific year can be asked for through the 'years' argument or the most
    recent year will be used.

    Parameters
    ----------
    country_codes: List[str]
        List of country codes
    nb_years: int (default: 1)
        Number of years for which we want load
    days_range_start: datetime.date (default: datetime.date(1, 1, 31))
        Date specifying the first day of the year for which we want load. Note that the year that is specified in that
        date is not important.
    days_range_end: datetime.date (default: datetime.date(1, 12, 31))
        Date specifying the last day of the year for which we want load. Note that the year that is specified in that
        date is not important.
    years: List[int] TODO: to implement
        List of years for which we want data

    Returns
    -------
    load_data_time_sliced: pd.DataFrame
        DataFrame containing the load for each source country and indexed with integers
        from 0 to nb_years*((days_range_end-days_range_start).days+1)*24 and where
    """

    # TODO: if years is specified should take the precedence over nb_years

    assert days_range_start.month < days_range_end.month or \
        (days_range_start.month == days_range_end.month and days_range_start.day <= days_range_end.day), \
        "ERROR: The days_range_start month-day pair must happen before the days_range_end month-day pair."

    load_data_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../data/load/generated/opsd_load.csv")
    load_data = pd.read_csv(load_data_fn, index_col=0)
    load_data.index = pd.DatetimeIndex(load_data.index)  # TODO: is there an automatic way to do this

    # Check data is available for those countries
    missing_code_in_data = sorted(list(set(country_codes) - set(load_data.keys())))
    assert len(missing_code_in_data) == 0, "No data present for the following codes: {}".format(missing_code_in_data)

    load_data = load_data[country_codes]

    # Select time period
    if years is None:

        # TODO: need to take implement a way to take into account leap years... - probably just remove the addtional day

        load_data_time_sliced = pd.DataFrame(index=range(nb_years*((days_range_end-days_range_start).days+1)*24),
                                             columns=load_data.keys())
        for code in load_data.keys():
            load_data_for_code = load_data[code].dropna()

            # Get the most recent year of data that has up to the day we need
            end_year = (load_data_for_code.last_valid_index() -
                        datetime.timedelta(
                            days=(days_range_end-datetime.date(days_range_end.year, 1, 1)).days, hours=23)).year
            end_date = datetime.datetime(year=end_year, month=days_range_end.month, day=days_range_end.day, hour=23)

            # Check then if we have enough years
            start_year = end_year-nb_years+1
            start_date = datetime.datetime(year=start_year, month=days_range_start.month, day=days_range_start.day)

            assert start_date in load_data_for_code.index, \
                "There is no sufficient data for region of code {}".format(code)
            load_data_time_sliced[code] = load_data_for_code.loc[start_date:end_date].values

        return load_data_time_sliced
    else:

        # Detect number of leap years
        nb_leap_years = sum([1 for year in years if year % 4 == 0 or year % 100 == 0])

        # TODO: this is ugly
        index = pd.date_range(start=datetime.datetime(year=years[0], month=days_range_start.month, day=days_range_start.day),
                  end=datetime.datetime(year=years[-1], month=days_range_end.month, day=days_range_end.day, hour=23), freq='1H')

        load_data_time_sliced = \
            pd.DataFrame(index=index,
                         columns=load_data.keys())

        sorted_years = sorted(years)
        for code in load_data.keys():
            load_data_for_code = load_data[code].dropna()

            # Check if we have data for the first required year and last required year
            first_date = datetime.datetime(year=sorted_years[0], month=days_range_start.month, day=days_range_start.day)
            assert first_date in load_data_for_code.index, \
                "There is no sufficient data for the date {} for region of code {}".format(first_date, code)
            last_date = datetime.datetime(year=sorted_years[-1], month=days_range_end.month, day=days_range_end.day)
            assert last_date in load_data_for_code.index, \
                "There is no sufficient data for the date {} for region of code {}".format(last_date, code)

            # Add the time-series for each year
            load = np.array([])
            for year in years:
                start_date = datetime.datetime(year=year, month=days_range_start.month, day=days_range_start.day)
                end_date = datetime.datetime(year=year, month=days_range_end.month, day=days_range_end.day, hour=23)
                load = np.append(load, load_data_for_code.loc[start_date:end_date].values)
            load_data_time_sliced[code] = load

        return load_data_time_sliced


def get_countries_load_interpolated(target_countries: List[str], source_countries: List[str], nb_years: int = 1,
                                    days_range_start: datetime.date = datetime.date(1, 1, 1),
                                    days_range_end: datetime.date = datetime.date(1, 12, 31),
                                    years: List[int] = None):
    """
    Returns a pd.Dataframe where the columns corresponds the load time-series (in GWh) for a series of target countries.
    The load is computed based on the load data of another series of source countries.
    Simply stated for each target country, the load is obtained by summing the load of all the source countries,
    dividing it by the sum of the yearly load of these countries and the multiplying it by the yearly load of the
    source country.

    Parameters
    ----------
    target_countries: List[str]
        List of codes of countries for which we want to obtain the load
    source_countries: List[str]
        List of codes of countries on which the load computation will be based
    nb_years: int (default: 1)
        Number of years for which we want load
    days_range_start: datetime.date (default: datetime.date(1, 1, 31))
        Date specifying the first day of the year for which we want load. Note that the year that is specified in that
        date is not important.
    days_range_end: datetime.date (default: datetime.date(1, 12, 31))
        Date specifying the last day of the year for which we want load. Note that the year that is specified in that
        date is not important.
    years: List[int]
        List of years for which we want data

    Returns
    -------
    target_countries_load: pd.DataFrame
        DataFrame containing the load for each source country and indexed with integers
        from 0 to nb_years*((days_range_end-days_range_start).days+1)*24 and where
    """

    # Source countries
    # Get the load for the source countries
    source_countries_load = get_countries_load(source_countries, nb_years, days_range_start, days_range_end, years)

    # Load the yearly load of the source regions
    yearly_load_source_countries = 0
    for c in source_countries:
        yearly_load_source_countries += get_yearly_country_load(c, 2016)
    total_source_countries_load = source_countries_load.sum(axis=1)/yearly_load_source_countries

    # Target countries
    yearly_load_target_countries = dict.fromkeys(target_countries)
    for c in target_countries:
        yearly_load_target_countries[c] = get_yearly_country_load(c, 2016)

    target_countries_load = pd.DataFrame(index=source_countries_load.index, columns=target_countries)
    for c in target_countries_load:
        target_countries_load[c] = total_source_countries_load*yearly_load_target_countries[c]

    return target_countries_load


def get_load_from_nuts_codes(region_codes_lists: List[List[str]], nb_years: int = 1,
                             days_range_start: datetime.date = datetime.date(1, 1, 1),
                             days_range_end: datetime.date = datetime.date(1, 12, 31),
                             years: List[int] = None) -> pd.DataFrame:
    """Returns a pandas.DataFrame containing the load values in GWh for a list of group of regions
    for a given number of hours.

    Parameters
    ----------
    region_codes_lists: List[List[str]]
        List of list of codes of regions. A load series will be returned for each list of regions.
    nb_years: int (default: 1)
        Number of years for which we want load
    days_range_start: datetime.date (default: datetime.date(1, 1, 31))
        Date specifying the first day of the year for which we want load. Note that the year that is specified in that
        date is not important.
    days_range_end: datetime.date (default: datetime.date(1, 12, 31))
        Date specifying the last day of the year for which we want load. Note that the year that is specified in that
        date is not important.
    years: List[int]
        List of years for which we want data

    Returns
    -------
    load: pd.DataFrame
        DataFrame with index being time stamps and each column corresponding to the load of a given list of regions
    """

    # Load the file indicating if load information is available for the given regions
    load_info_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../data/load/source_load_countries.csv")
    load_info = pd.read_csv(load_info_fn, index_col="Code")
    load_info = load_info.dropna()

    pop_dens_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../../../data/geographics/source/eurostat/demo_r_d3dens.xls")
    pop_dens = pd.read_excel(pop_dens_fn, header=8, index_col=0)[:2213]
    pop_dens.index.name = 'code'

    # TODO: Maybe do a function to open this file
    area = get_nuts_area()
    area.index.name = 'code'

    gdp_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../../data/geographics/source/eurostat/nama_10r_3gdp.xls")
    gdp = pd.read_excel(gdp_fn, header=8, index_col=0)[:1945]
    gdp.index.name = 'code'

    # Load data at countries level
    index = range(((days_range_end-days_range_start).days+1)*nb_years*24)
    if years is not None:
        range(((days_range_end-days_range_start).days+1) * len(years))

    all_countries = set.union(*[set([region[:2] for region in region_list])
                                for region_list in region_codes_lists])

    # Replace some codes if present
    if "EL" in all_countries:
        all_countries.remove("EL")
        all_countries.add("GR")
    if "UK" in all_countries:
        all_countries.remove("UK")
        all_countries.add("GB")

    all_countries = list(all_countries)

    # Divide countries for which we directly have access to data or not
    source_countries = [load_info.loc[c]["Source region"] for c in all_countries]
    countries_to_interpolate = [c for i, c in enumerate(all_countries) if c != source_countries[i]]
    countries_to_interpolate_source = [c for i, c in enumerate(source_countries) if c != all_countries[i]]
    countries_not_to_interpolate = [c for i, c in enumerate(all_countries) if c == source_countries[i]]

    countries_load = pd.DataFrame(index=index, columns=list(all_countries))
    countries_load[countries_not_to_interpolate] = get_countries_load(countries_not_to_interpolate, nb_years,
                                                                      days_range_start, days_range_end, years)
    countries_load[countries_to_interpolate] =\
        get_countries_load_interpolated(countries_to_interpolate, countries_to_interpolate_source, nb_years,
                                        days_range_start, days_range_end, years)

    load = pd.DataFrame(0, index=index, columns=range(len(region_codes_lists)))
    for idx, region_codes_list in enumerate(region_codes_lists):
        for code in region_codes_list:
            nuts_country_code = code[:2]

            country_code = nuts_country_code
            if nuts_country_code == "EL":
                country_code = "GR"
            if nuts_country_code == "UK":
                country_code = "GB"

            country_load = countries_load[country_code]

            # Once we have the load for the countries, down scale it to the regions of interest by using population
            # and gdp
            regions_pop_dens = pop_dens.loc[code]['2016']
            regions_area = area.loc[code]['2016']
            # regions_gdp = gdp.loc[region_codes_list]
            # TODO: doing the thing only on population for now but need to update that
            #  -> would be nice if we had yearly load
            country_pop_dens = pop_dens.loc[nuts_country_code]['2016']
            country_area = area.loc[nuts_country_code]['2016']
            # country_gdp = gdp.loc[country_code]

            load[idx] += country_load * regions_pop_dens * regions_area / (country_pop_dens * country_area)

    return load


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

    key_ind_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../../data/key_indicators/generated/" + country + ".csv")
    return pd.read_csv(key_ind_fn, index_col=0).loc[year, "Electricity consumption (TWh)"]


def available_load():

    load_data_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../data/load/generated/opsd_load.csv")
    load_data = pd.read_csv(load_data_fn, index_col=0)
    load_data.index = pd.DatetimeIndex(load_data.index)

    load_data_start_of_years = load_data[(load_data.index.month == 1) & (load_data.index.day == 1) & (load_data.index.hour == 0)]
    load_data_end_of_years = load_data[(load_data.index.month == 12) & (load_data.index.day == 31) & (load_data.index.hour == 23)]

    available_data = pd.DataFrame(index=load_data.keys(), columns=["start", "end"])
    for key in load_data.keys():
        load_start_indexes = load_data_start_of_years[key].dropna().index
        load_end_indexes = load_data_end_of_years[key].dropna().index
        available_data.loc[key, 'start'] = min(load_start_indexes).year
        available_data.loc[key, 'end'] = max(load_end_indexes).year
        print(available_data.loc[key])

    available_data.to_csv("available_load_years.csv")

    return available_data


# TODO:
#   Need to merge this function
def retrieve_load_data(regions: List[str], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns load time series for given regions and time horizon.

    Parameters
    ----------
    regions: List[str]
        Code of regions
    timestamps: pd.DatetimeIndex
        Datetime index

    Returns
    -------
    load_per_region: pd.DataFrame (index = timestamps, columns = regions)
        DataFrame associating to each region code the corresponding load data

    """

    path_load_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../../../data/load/generated/load_opsd_2015_2018.csv')
    load_data = pd.read_csv(path_load_data, index_col=0, sep=';')
    load_data.index = pd.to_datetime(load_data.index)

    assert timestamps[0] in load_data.index, "Error: Start datetime not in load data"
    assert timestamps[-1] in load_data.index, "Error: End datetime not in load data"

    # Slice data on time
    load_data = load_data.loc[timestamps]

    # Slice data on region
    load_per_region = pd.DataFrame(columns=regions, index=timestamps)
    for region in regions:
        # Get load date for all subregions, sum it and transform to GW # TODO: check it is indeed GW
        load_per_region[region] = load_data[get_subregions(region)].sum(axis=1).values * 1e-3

    return load_per_region


"""
available_loads = available_load()
available_loads_countries = available_loads[[len(idx) == 2 for idx in available_loads.index]]
# Load from 2015 to 2018
load_2015_2018_countries = available_loads_countries[(available_loads_countries.start <= 2015) & (available_loads_countries.end >= 2018)].index
load_2015_2018 = get_countries_load(load_2015_2018_countries, years=[2015, 2016, 2017, 2018])
load_2015_2018.to_csv("load_opsd_2015_2018.csv")

load_2016_2018_countries = available_loads_countries[(available_loads_countries.start <= 2016) & (available_loads_countries.end >= 2018)].index
load_2016_2018 = get_countries_load(load_2016_2018_countries, years=[2016, 2017, 2018])
load_2016_2018.to_csv("load_opsd_2016_2018.csv")
"""
# def attach_load(network):
#
#     # Get load per countries
#     # load = get_load(self.regions, self.start_date, self.end_date, self.time_resolution)
#     load = get_load_bis(network.regions, len(network.timestamps))
#
#     # Compute area of each countries # TODO: maybe do this somewhere else
#     network.onshore_shapes['area'] = network.onshore_shapes.apply(lambda p: p['geometry'].area, axis=1)
#
#     for bus_id in network.buses.id.values:
#
#         # TODO: this is just an example where the load
#         #  per bus is define proportionally to area of its associated region
#         #  -> would need to base it on population and gdp
#         # Find in which region the bus is
#         x = network.buses['x'].sel(id=bus_id).item()
#         y = network.buses['y'].sel(id=bus_id).item()
#         region = network.onshore_shapes\
#             .loc[network.onshore_shapes['geometry'].apply(lambda p: p.contains(Point((x, y))))]
#
#         # Get the load for the region
#         region_load = load[region.index.values[0]].values
#         region_area = region['area'].values[0]
#
#         # Get the size of the region
#         bus_region = network.buses['region'].sel(id=bus_id)
#         bus_region_shape = bus_region.item()
#         if isinstance(bus_region_shape, str):
#             bus_region_shape = shapely.wkt.loads(bus_region_shape)
#         bus_area = bus_region_shape.area
#
#         args = {"bus": [bus_id], "p_set": [region_load*float(bus_area)/region_area]}
#         network.add("load", ["load " + str(bus_id)], args)
