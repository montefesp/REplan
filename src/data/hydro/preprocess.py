from os.path import join, abspath, dirname
from os import listdir

import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import datetime

from src.data.geographics import match_points_to_regions


def compute_ror_series(dataset_runoff: xr.Dataset, region_points: list, flood_event_threshold: float) -> pd.DataFrame:
    """
    Computing ROR p.u. time series as directly proportional to runoff for a given grid cell/area.

    Parameters
    ----------
    dataset_runoff: xarray.Dataset
        Contains runoff data, in this case expressed in m.
    region_points: list
        List of points (lon, lat) within a region.
    flood_event_threshold: float
        Quantile clipping runoff timeseries (stems from the assumption that ROR plants are designed for a, e.g. p80 flow).

    Returns
    -------
    ts_norm: pd.DataFrame
        Timeseries of p.u. capacity factors for ROR plants.
    """

    # Summation of runoffs over all points within the NUTS region.
    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro.load()
    # Clipping according to the flood_event_threshold
    ts_clip = ts.clip(max=ts.quantile(q=flood_event_threshold).values)
    # Normalizing for p.u. representation.
    ts_norm = ts_clip / ts_clip.max()

    return ts_norm


def compute_sto_inflows(dataset_runoff: xr.Dataset, region_points: list, unit_area: float = 1225,
                        g: float = 9.81, rho: float = 1000.)-> pd.DataFrame:
    """
     Computing STO inflow time series (GWh).

     Parameters
     ----------
     dataset_runoff: xarray.Dataset
         Contains runoff data, in this case expressed in m.
     region_points: list
         List of points (lon, lat) within a region.
     unit_area: float
         Area of one grid cell, in this case represented as a 0.5deg x 0.5deg square.
         Assuming an average distance of 35km between points (at 50deg latitude), a square area equals 1225km2.
     g: float
        Gravitational constant [m/s2].
     rho: float
        Water density [kg/m3].

     Returns
     -------
     ts_gwh: pd.DataFrame
         Timeseries of STO inflows.
     """

    # Summation of runoffs over all points within the NUTS region.
    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro
    # Convert from the runoff unit (m) to some equivalent water volume (m3).
    ts_cm = ts * (unit_area * 1e6)
    # Convert from J to WH and then to GWh.
    ts_gwh = ts_cm * g * rho * (1/3600) * 1e-9

    return ts_gwh


# TODO comment
def generate_eu_hydro_files(topology_unit: str, timestamps: pd.DatetimeIndex, flood_event_threshold: float):

    assert topology_unit in ["NUTS0", "NUTS2"], "Error: topology_unit must be NUTS2 or NUTS0"

    # Read region shapes
    # TODO: can we not use geographics manager? probably yes
    geographics_dir = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/")
    if topology_unit == 'NUTS2':
        # TODO: need to change that (i.e. need to add BA in a better way)
        nuts_shapes_fn = f"{geographics_dir}NUTS_RG_01M_2016_4326_LEVL_2_incl_BA.geojson"
        # nuts_shapes_fn = f"{geographics_dir}NUTS_RG_01M_2016_4326_LEVL_2.geojson"
    else:  # topology_unit == 'NUTS0':
        nuts_shapes_fn = f"{geographics_dir}NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson"
    nuts_shapes = gpd.read_file(nuts_shapes_fn).set_index('NUTS_ID')

    # Read hydro plants data from powerplantmatching tool from which we can retrieve capacity (in MW)
    hydro_plants_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/pp_fresna_hydro_updated.csv")
    hydro_plants_df = pd.read_csv(hydro_plants_fn, sep=';', usecols=[0, 1, 3, 5, 6, 8, 13, 14], index_col=0)

    # Find to which NUTS region each plant belongs
    # TODO: maybe we should first check countries
    hydro_plants_locs = hydro_plants_df[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values
    points_nuts_ds = match_points_to_regions(hydro_plants_locs, nuts_shapes["geometry"]).dropna()

    def add_region(lon, lat):
        nuts_ = points_nuts_ds[lon, lat]
        # Need the if because some points are exactly at the same position
        return nuts_ if isinstance(nuts_, str) else nuts_.iloc[0]
    hydro_plants_df["NUTS"] = hydro_plants_df[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)
    hydro_plants_df = hydro_plants_df[~hydro_plants_df['NUTS'].isnull()]

    # ROR plants
    # Get ROR plants and aggregate capacity (to GW) per NUTS region
    ror_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Run-Of-River']
    ror_nuts_capacity_ds = ror_plants_df.groupby(ror_plants_df['NUTS'])['Capacity'].sum()*1e-3

    # STO plants
    # Get STO plants and aggregate capacity (to GW) per NUTS region
    hydro_sto = hydro_plants_df[hydro_plants_df['Technology'] == 'Reservoir']
    sto_nuts_capacity_df = hydro_sto.groupby(hydro_sto['NUTS'])['Capacity'].sum().to_frame()*1e-3

    # Get energy capacity per country (in GWh)
    nuts_hydro_storage_capacities_fn = join(dirname(abspath(__file__)),
                                            "../../../data/hydro/source/hydro_storage_capacities_updated.csv")
    nuts_hydro_storage_energy_cap_ds = pd.read_csv(nuts_hydro_storage_capacities_fn, sep=';',
                                                   usecols=['Country', 'E_store[TWh]'], index_col='Country',
                                                   squeeze=True)*1e3

    # Divide this energy capacity between NUTS regions proportionally to power capacity
    for nuts in sto_nuts_capacity_df.index:
        nuts0 = nuts[:2]
        country_storage_potential = nuts_hydro_storage_energy_cap_ds[nuts0]
        country_capacity_potential = \
            sto_nuts_capacity_df.loc[sto_nuts_capacity_df.index.str.startswith(nuts0), "Capacity"].sum()
        sto_nuts_capacity_df.loc[nuts, "Energy"] = \
            (sto_nuts_capacity_df.loc[nuts, "Capacity"] / country_capacity_potential) * country_storage_potential

    # PHS plants
    # Get PHS plants and aggregate capacity (to GW) per NUTS region
    php_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Pumped Storage']
    php_nuts_capacity_df = php_plants_df.groupby(php_plants_df['NUTS'])['Capacity'].sum().to_frame()*1e-3

    # Compute PHS energy capacity
    # Eurelectric 2011 study. Assumed 12h for the rest. Optimistic from a existing storage perspective.
    php_duration_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/php_duration.csv")
    php_durations_ds = pd.read_csv(php_duration_fn, squeeze=True, index_col=0)
    countries = [item[:2] for item in php_nuts_capacity_df.index]
    php_nuts_capacity_df["Energy"] = php_nuts_capacity_df["Capacity"]*php_durations_ds.loc[countries].values

    # Group all capacities in one dataframe (+ round to MWh)
    capacities_df = pd.DataFrame(index=sorted(nuts_shapes.index),
                                 columns=['ROR_CAP [GW]', 'STO_CAP [GW]', 'STO_EN_CAP [GWh]',
                                          'PSP_CAP [GW]', 'PSP_EN_CAP [GWh]'])
    capacities_df['ROR_CAP [GW]'] = ror_nuts_capacity_ds
    capacities_df['STO_CAP [GW]'] = sto_nuts_capacity_df['Capacity']
    capacities_df['STO_EN_CAP [GWh]'] = sto_nuts_capacity_df['Energy']
    capacities_df['PSP_CAP [GW]'] = php_nuts_capacity_df['Capacity']
    capacities_df['PSP_EN_CAP [GWh]'] = php_nuts_capacity_df['Energy']
    capacities_df = capacities_df.round(3)

    # Some NUTS2 regions are associated with, e.g., metropolitan areas, thus will be manually set to 0.
    if topology_unit == 'NUTS2':

        regions_ror = ['PT20', 'PT30', 'DE50', 'DE60', 'DE71', 'DE80', 'DEA4', 'DE24',
                       'DE72', 'DE92', 'DEA1', 'DEB3', 'HU22', 'DED2', 'DED5', 'NO02',
                       'NO03', 'PL61', 'PL62', 'PL71', 'UKC1', 'UKF1', 'UKC2', 'AT13']
        capacities_df.loc[regions_ror, 'ROR_CAP [GW]'] = None

        regions_sto = ['BE33', 'BE34', 'IE05', 'AT13']
        capacities_df.loc[regions_sto, 'STO_CAP [GW]'] = None
        capacities_df.loc[regions_sto, 'STO_EN_CAP [GWh]'] = None

    # TODO: this (still) take ages, maybe still away to improve speed
    # Compute time-series of inflow for STO (in GWh) and ROR (per unit of capacity)
    runoff_dir = join(dirname(abspath(__file__)), "../../../data/land_data/source/ERA5/runoff/")
    runoff_files = [join(runoff_dir, fn) for fn in listdir(runoff_dir) if fn.endswith(".nc")]
    runoff_dataset = xr.open_mfdataset(runoff_files, combine='by_coords')
    runoff_dataset = runoff_dataset.stack(locations=('longitude', 'latitude'))
    runoff_dataset = runoff_dataset.sel(time=timestamps)

    # Find to which nuts region each of the runoff points belong
    # TODO: instead of doing that here, we could apply it twice with shapes filtered on nuts_with_ror_index
    #  and nuts_with_sto_index, but I want first to refactor match_points_to_region
    points_nuts_ds = match_points_to_regions(runoff_dataset.locations.values, nuts_shapes["geometry"]).dropna()

    # ROR inflow (per unit of capacity)
    nuts_with_ror_index = capacities_df[capacities_df['ROR_CAP [GW]'].notnull()].index
    df_ror = pd.DataFrame(index=timestamps, columns=nuts_with_ror_index)
    for nuts in nuts_with_ror_index:
        nuts_points = points_nuts_ds[points_nuts_ds == nuts].index.to_list()
        # TODO: if nuts_points is an empty list, what happens?
        df_ror[nuts] = compute_ror_series(runoff_dataset, nuts_points, flood_event_threshold)

    nuts_with_sto_index = capacities_df[capacities_df['STO_CAP [GW]'].notnull()].index
    df_sto = pd.DataFrame(index=timestamps, columns=nuts_with_sto_index)
    for nuts in nuts_with_sto_index:
        nuts_points = points_nuts_ds[points_nuts_ds == nuts].index.to_list()
        df_sto[nuts] = compute_sto_inflows(runoff_dataset, nuts_points)

    # STO inflow (in GWh)
    if topology_unit == 'NUTS0':

        hydro_production_fn = join(dirname(abspath(__file__)),
                                   "../../../data/hydro/source/Eurostat_hydro_net_generation.xls")
        hydro_production_ds = pd.read_excel(hydro_production_fn, skiprows=12, index_col=0)
        hydro_production = hydro_production_ds[[str(y) for y in timestamps.year.unique()]].\
                                                                                        fillna(method='bfill', axis=1)

        ror_production = df_ror.groupby(df_ror.index.year).sum()\
                                        .multiply(capacities_df['ROR_CAP [GW]'].dropna(), axis=1).transpose()
        ror_production.index = ror_production.index.map(str)

        sto_production = hydro_production.copy()
        for nuts in ror_production.index:
            sto_production.loc[nuts, :] = hydro_production.loc[nuts, :].values - \
                                          ror_production.loc[nuts, :].values
        # LV and IE seem to have more ROR potential than the Eurostat total hydro generation.
        sto_production_sum = sto_production.clip(lower=0.).sum(axis=1)

        sto_inflows_sum = df_sto.sum(axis=0)
        sto_multipliers_ds = sto_inflows_sum.copy()
        for nuts in sto_multipliers_ds.index:
            sto_multipliers_ds.loc[nuts] = sto_production_sum.loc[nuts] / sto_inflows_sum.loc[nuts]
        df_sto = df_sto.multiply(sto_multipliers_ds)

        sto_multipliers_save_fn = "../../../data/hydro/source/hydro_sto_multipliers_NUTS0.csv"
        sto_multipliers_ds.to_csv(sto_multipliers_save_fn)

    # Quite a hack here that should work fine for the beginning. Basically, NUTS0 topology needs to be ran first to
    # create the multipliers file that is read down below. TODO: look into computing multipliers for NUTS2 directly.
    else: # topology_unit == 'NUTS2'

        sto_multipliers_fn = join(dirname(abspath(__file__)),
                                   "../../../data/hydro/source/hydro_sto_multipliers_NUTS0.csv")
        sto_multipliers_ds = pd.read_csv(sto_multipliers_fn, index_col=0).squeeze()
        for nuts in df_sto.columns:
            df_sto[nuts] *= sto_multipliers_ds.loc[nuts[:2]]

        # For some reason, there is no point associated with ITC2. Fill with neighboring area.
        df_ror['ITC2'] = df_ror['ITC3']
        # Same situation here, some wild hacks for the moment to fill
        # them with data from regions with similar capacities.
        df_sto['ES52'] = df_sto['ES24']
        df_sto['EL42'] = df_sto['EL61']
        df_sto['ITC2'] = df_sto['ITC4'] / 2.

    # Saving files
    save_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    capacities_save_fn = f"{save_dir}hydro_capacities_per_{topology_unit}.csv"
    capacities_df.to_csv(capacities_save_fn)
    ror_save_fn = f"{save_dir}hydro_ror_time_series_per_{topology_unit}_pu.csv"
    df_ror.to_csv(ror_save_fn)
    sto_save_fn = f"{save_dir}hydro_sto_inflow_time_series_per_{topology_unit}_GWh.csv"
    df_sto.to_csv(sto_save_fn)


if __name__ == '__main__':

    # IEA validation data in 2016 (TWh produced per year)
    # hydro_production_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/hydro_production.csv")
    # hydro_production_ds = pd.read_csv(hydro_production_fn, squeeze=True, index_col=0)

    nuts_type = 'NUTS2'
    ror_flood_threshold = 0.8
    start = datetime(2014, 1, 1, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)
    ts = pd.date_range(start, end, freq='H')

    generate_eu_hydro_files(nuts_type, ts, ror_flood_threshold)
