from os.path import join, dirname, abspath

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt


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

    final_data_fn = join(dirname(abspath(__file__)), "../../../data/load/generated/opsd_load.csv")
    final_data.to_csv(final_data_fn)


if __name__ == "__main__":
    preprocess()