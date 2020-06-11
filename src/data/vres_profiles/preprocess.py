from os.path import join, dirname, abspath
from os import listdir

import pandas as pd


def built_cap_factor_for_countries_files():
    """
    Create files containing capacity factors for each country
    """

    ninja_dir = join(dirname(abspath(__file__)), "../../../data/vres_profiles/source/ninja_europe_2019/")
    all_files = listdir(ninja_dir)

    # Wind
    wind_file_names = [file for file in all_files if file.startswith('ninja_wind')]

    # Add first data from current fleet
    current_wind_file_names = sorted([file for file in wind_file_names if "current" in file])

    ts = pd.date_range('2015-01-01T00:00', '2018-12-31T23:00', freq='1H')
    onshore_capacity_factors_df = pd.DataFrame(index=ts, dtype=float)
    offshore_capacity_factors_df = pd.DataFrame(index=ts, dtype=float)
    for fn in current_wind_file_names:
        country = fn.split('_')[3]
        header = pd.read_csv(f"{ninja_dir}{fn}", nrows=1, skiprows=1).columns[0]
        capacity_factors_df = pd.read_csv(f"{ninja_dir}{fn}", index_col=0, skiprows=2)
        capacity_factors_df.index = pd.to_datetime(capacity_factors_df.index)
        capacity_factors_df = capacity_factors_df.loc[ts]
        if 'onshore' in header and 'offshore' in header:
            onshore_capacity_factors_df[country] = capacity_factors_df['onshore']
            offshore_capacity_factors_df[country] = capacity_factors_df['offshore']
        elif 'onshore' in header:
            onshore_capacity_factors_df[country] = capacity_factors_df['national']
        elif 'offshore' in header:
            offshore_capacity_factors_df[country] = capacity_factors_df['national']
        else:
            raise ValueError("No offshore or onshore detected.")

    # If data not yet available for certain countries, add them from future fleet
    long_wind_file_names = sorted([file for file in wind_file_names if "long" in file])
    for fn in long_wind_file_names:
        country = fn.split('_')[3]
        header = pd.read_csv(f"{ninja_dir}{fn}", nrows=1, skiprows=1).columns[0]
        capacity_factors_df = pd.read_csv(f"{ninja_dir}{fn}", index_col=0, skiprows=2)
        capacity_factors_df.index = pd.to_datetime(capacity_factors_df.index)
        capacity_factors_df = capacity_factors_df.loc[ts]
        if 'offshore' in header:
            if country not in offshore_capacity_factors_df.columns:
                offshore_capacity_factors_df[country] = capacity_factors_df['national']
        else:
            raise ValueError("No offshore detected.")

    onshore_capacity_factors_df = onshore_capacity_factors_df.sort_index()
    offshore_capacity_factors_df = offshore_capacity_factors_df.sort_index()

    # PV
    pv_file_names = sorted([file for file in all_files if file.startswith('ninja_pv')])
    ts = pd.date_range('2015-01-01T00:00', '2018-12-31T23:00', freq='1H')
    pv_capacity_factors_df = pd.DataFrame(index=ts, dtype=float)
    for fn in pv_file_names:
        country = fn.split('_')[3]
        capacity_factors_ds = pd.read_csv(f"{ninja_dir}{fn}", index_col=0, skiprows=2)["national"]
        capacity_factors_ds.index = pd.to_datetime(capacity_factors_ds.index)
        pv_capacity_factors_df[country] = capacity_factors_ds[ts]

    # Save all files
    resource_dir = join(dirname(abspath(__file__)), "../../../data/vres_profiles/generated/")
    onshore_capacity_factors_df.to_csv(f"{resource_dir}onshore_wind_cap_factors.csv")
    offshore_capacity_factors_df.to_csv(f"{resource_dir}offshore_wind_cap_factors.csv")
    pv_capacity_factors_df.to_csv(f"{resource_dir}pv_cap_factors.csv")


if __name__ == '__main__':
    built_cap_factor_for_countries_files()
