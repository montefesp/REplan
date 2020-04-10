from os.path import join, dirname, abspath

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from src.data.load.manager import get_load


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
    This function builds a complete and easy to use database of load by formatting and
    correcting some open-source data contained in the folder data/load/source.
    The results are saved in the folder data/load/generated.
    """

    source_data_fn = join(dirname(abspath(__file__)),
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

    final_data.index = final_data.index.strftime('%Y-%m-%d %H:%M:%S')
    final_data_fn = join(dirname(abspath(__file__)), "../../../data/load/generated/opsd_load.csv")
    final_data.to_csv(final_data_fn)


def get_load_full_years_range(save: bool = False) -> pd.DataFrame:
    """
    Returns (and optionally save) for each region for which we have load data in data/load/opsd_load.csv, the first
    and last year for which we have hourly data for everyday of the year.
    Note that data/load/opsd_load.csv is supposed to contain only contiguous time series (i.e. no NaN values in the
    middle of it)

    Parameters
    ----------
    save: bool (default: False)
        Whether to save the output data to a file

    Returns
    -------
    load_full_years_range: pd.DataFrame
        Contains for each region the first and last year for which we have data for every hour

    """

    opsd_load_fn = join(dirname(abspath(__file__)), "../../../data/load/generated/opsd_load.csv")
    load = pd.read_csv(opsd_load_fn, index_col=0)
    load.index = pd.DatetimeIndex(load.index)

    # Extract the load of the first and last hour of the year for every region
    load_first_hour_of_year = load[(load.index.month == 1) & (load.index.day == 1) & (load.index.hour == 0)]
    load_last_hour_of_year = load[(load.index.month == 12) & (load.index.day == 31) & (load.index.hour == 23)]

    # For each region, get the first year for which the first hour is not NaN and the last year for which it is not NaN
    load_full_years_range = pd.DataFrame(index=load.keys(), columns=["start", "end"])
    for key in load.keys():
        load_start_indexes = load_first_hour_of_year[key].dropna().index
        load_end_indexes = load_last_hour_of_year[key].dropna().index
        load_full_years_range.loc[key, 'start'] = min(load_start_indexes).year
        load_full_years_range.loc[key, 'end'] = max(load_end_indexes).year

    if save:
        load_full_years_range.to_csv("available_load_years.csv")

    return load_full_years_range


def create_countries_load_files():
    """
    This function creates two files: one containing hourly load data from 2015 to 2018 for countries for which we have
    data for each hour of those years and the other one created the same way but for years 2016 to 2018.

    """
    available_years = get_load_full_years_range()
    available_years_countries = available_years[[len(idx) == 2 for idx in available_years.index]]

    load_dir = join(dirname(abspath(__file__)), "../../../data/load/generated/")

    load_2015_2018_countries = available_years_countries[(available_years_countries.start <= 2015) &
                                                         (available_years_countries.end >= 2018)].index
    load_2015_2018 = get_load(countries=load_2015_2018_countries.values, years_range=[2015, 2018])
    load_2015_2018.to_csv(f"{load_dir}load_opsd_2015_2018.csv")

    load_2016_2018_countries = available_years_countries[(available_years_countries.start <= 2016) &
                                                         (available_years_countries.end >= 2018)].index
    load_2016_2018 = get_load(countries=load_2016_2018_countries.values, years_range=[2016, 2018])
    load_2016_2018.to_csv(f"{load_dir}load_opsd_2016_2018.csv")


if __name__ == "__main__":
    # preprocess()
    create_countries_load_files()
