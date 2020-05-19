from datetime import datetime
from os import listdir
from os.path import join, abspath, dirname, isfile
from typing import Union

from shapely.geometry import MultiPolygon, Polygon
import pickle
import pandas as pd
import xarray as xr
import numpy as np

from src.data.geographics import match_points_to_regions, get_nuts_shapes, get_natural_earth_shapes

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

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
        Quantile clipping runoff time series (stems from the assumption that ROR plants
        are designed for a, e.g. p80 flow).

    Returns
    -------
    ts_norm: pd.DataFrame
        Time series of p.u. capacity factors for ROR plants.
    """
    # Mean of runoffs over all points within the NUTS region.
    ts = dataset_runoff.ro.sel(locations=region_points).mean(dim='locations').load()
    # Compute quantile from xarray object.
    q = ts.quantile(q=flood_event_threshold)
    # Clipping according to the flood_event_threshold
    ts[ts > q] = q
    # Normalizing for p.u. representation.
    return ts / ts.max()


def compute_sto_inflows(dataset_runoff: xr.Dataset, region_points: list, unit_area: float,
                        g: float = 9.81, rho: float = 1000.) -> pd.DataFrame:
    """
     Computing STO inflow time series (GWh).

     Parameters
     ----------
     dataset_runoff: xarray.Dataset
         Contains runoff data, in this case expressed in m.
     region_points: list
         List of points (lon, lat) within a region.
     unit_area: float
         Area of one grid cell, depending on the runoff data resolution used.
         E.g., assuming an average distance of 20km between points (at 50deg latitude), a square area equals 628km2.
     g: float
        Gravitational constant [m/s2].
     rho: float
        Water density [kg/m3].

     Returns
     -------
     ts_gwh: pd.DataFrame
         Time series of STO inflows.
     """

    # Summation of runoffs over all points within the NUTS region.
    ts = dataset_runoff.ro.sel(locations=region_points).sum(dim='locations')
    # Convert from the runoff unit (m) to some equivalent water volume (m3). Convert from J to Wh and then to GWh.
    ts_cm = ts * (unit_area * 1e6) * g * rho * (1 / 3600) * 1e-9

    return ts_cm


def compute_sto_multipliers(timestamps: pd.DatetimeIndex, df_ror: pd.DataFrame, df_sto: pd.DataFrame,
                        capacities_df: pd.DataFrame) -> pd.Series:
    """
     Computing STO multipliers mapping cell runoff to approximated hourly-sampled reservoir inflows.

     Parameters
     ----------

     timestamps: pd.DatetimeIndex
        Time horizon.

     df_ror: pd.DataFrame
        Data frame with ROR (p.u.) capacity factors for each geographical unit across the time horizon considered.

     df_sto: pd.DataFrame
        Data frame with STO (GWh) inflow time series for each geographical unit across the time horizon considered.

     capacities_df: pd.DataFrame
        Data frame with hydro capacities for each geographical unit considered.

     Returns
     -------
     sto_multipliers_ds: pd.Series
         STO multipliers per country.
     """
    hydro_production_fn = join(dirname(abspath(__file__)),
                               "../../../data/hydro/source/Eurostat_hydro_net_generation.xls")
    hydro_production_ds = pd.read_excel(hydro_production_fn, skiprows=12, index_col=0)
    hydro_production = \
        hydro_production_ds[[str(y) for y in timestamps.year.unique()]].fillna(method='bfill', axis=1).dropna()

    ror_production = \
        df_ror.groupby(df_ror.index.year).sum().multiply(capacities_df['ROR_CAP [GW]'].dropna(), axis=1).transpose()
    ror_production.index = ror_production.index.map(str)

    ror_production_aggregate = ror_production.groupby(ror_production.index.str[:2]).sum()
    sto_production_aggregate = hydro_production.copy()

    for nuts in ror_production_aggregate.index:
        sto_production_aggregate.loc[nuts, :] = hydro_production.loc[nuts, :].values - \
                                                ror_production_aggregate.loc[nuts, :].values
    # LV and IE seem to have more ROR potential than the Eurostat total hydro generation.
    sto_production_sum = sto_production_aggregate.clip(lower=0.).sum(axis=1)

    sto_inflows_sum = df_sto.sum().groupby(df_sto.columns.str[:2]).sum()
    sto_multipliers_ds = sto_inflows_sum.copy()
    for nuts in sto_multipliers_ds.index:
        sto_multipliers_ds.loc[nuts] = sto_production_sum.loc[nuts] / sto_inflows_sum.loc[nuts]

    return sto_multipliers_ds


def read_runoff_data(resolution: float, timestamps: pd.DatetimeIndex) -> xr.Dataset:
    """
     Reading runoff data.

     Parameters
     ----------
     resolution: float
        Reanalysis data spatial resolution.
     timestamps: pd.DatetimeIndex
        Time horizon.

     Returns
     -------
     runoff_dataset: xr.Dataset

     """
    runoff_dir = join(dirname(abspath(__file__)), f"../../../data/hydro/source/ERA5/runoff/{resolution}")
    runoff_files = [join(runoff_dir, fn) for fn in listdir(runoff_dir) if fn.endswith(".nc")]
    runoff_dataset = xr.open_mfdataset(runoff_files, combine='by_coords')
    runoff_dataset = runoff_dataset.stack(locations=('longitude', 'latitude'))
    runoff_dataset = runoff_dataset.sel(time=timestamps)

    return runoff_dataset


def build_capacities_frame(region_shapes: Union[Polygon, MultiPolygon], ror_capacities: pd.DataFrame,
                           sto_capacities: pd.DataFrame, php_capacities: pd.DataFrame):
    """

     Parameters
     ----------

     region_shapes: Union[Polygon, MultiPolygon]

     ror_capacities: pd.DataFrame

     sto_capacities: pd.DataFrame

     php_capacities: pd.DataFrame

     Returns
     -------
     capacities_df: pd.DataFrame

     """

    # Group all capacities in one dataframe (+ round to MWh)
    capacities_df = pd.DataFrame(index=sorted(region_shapes.index),
                                 columns=['ROR_CAP [GW]', 'STO_CAP [GW]', 'STO_EN_CAP [GWh]',
                                          'PSP_CAP [GW]', 'PSP_EN_CAP [GWh]'])
    capacities_df['ROR_CAP [GW]'] = ror_capacities
    capacities_df['STO_CAP [GW]'] = sto_capacities['Capacity']
    capacities_df['STO_EN_CAP [GWh]'] = sto_capacities['Energy']
    capacities_df['PSP_CAP [GW]'] = php_capacities['Capacity']
    capacities_df['PSP_EN_CAP [GWh]'] = php_capacities['Energy']
    capacities_df = capacities_df.dropna(how='all').round(3)

    # Temporary hack to ensure "lost" points are not accounted for. KeyErrors raised otherwise.
    non_eu_regions = ['TR', 'UA', 'TN', 'MA']
    capacities_df = capacities_df[~capacities_df.index.str.contains('|'.join(non_eu_regions))]

    return capacities_df


def generate_eu_hydro_files(resolution: float, topology_unit: str, timestamps: pd.DatetimeIndex,
                            flood_event_threshold: float, sto_unit_area: float):
    """
     Generating hydro files, i.e., capacities and inflows.

     Parameters
     ----------
     resolution: float
         Runoff data spatial resolution.
     topology_unit: str
         Topology in use ('countries', 'NUTS2', 'NUTS3').
     timestamps: pd.DatetimeIndex
         Time horizon for which inflows are computed.
     flood_event_threshold: float
         Quantile clipping runoff time series (see compute_ror_series for usage).
     sto_unit_area: float
         Average area of reanalysis grid cell, used for estimating inflows in reservoir-based hydro plants.
         Depends on the resolution of the underlying runoff data.

     """

    assert topology_unit in ["countries", "NUTS2", "NUTS3"], "Error: requested topology_unit not available."

    # Load shapes based on topology
    if topology_unit == 'countries':
        shapes = get_natural_earth_shapes()
        shapes.rename(index={'GR': 'EL', 'GB': 'UK'}, inplace=True)
    else:  # topology == 'NUTS2', 'NUTS3'
        shapes = get_nuts_shapes(topology_unit[-1:])

    # Read hydro plants data from powerplantmatching tool from which we can retrieve capacity (in MW)
    hydro_plants_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/pp_fresna_hydro_updated.csv")
    hydro_plants_df = pd.read_csv(hydro_plants_fn, sep=';', usecols=[0, 1, 3, 5, 6, 8, 13, 14], index_col=0)

    # Read energy capacity per country data (in GWh)
    hydro_storage_capacities_fn = join(dirname(abspath(__file__)),
                                            "../../../data/hydro/source/hydro_storage_capacities_updated.csv")
    hydro_storage_energy_cap_ds = pd.read_csv(hydro_storage_capacities_fn, sep=';',
                                                   usecols=['Country', 'E_store[TWh]'], index_col='Country',
                                                   squeeze=True) * 1e3

    # Read PHS duration data (hrs)
    php_duration_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/php_duration.csv")
    php_durations_ds = pd.read_csv(php_duration_fn, squeeze=True, index_col=0)

    # Find to which region each plant belongs
    hydro_plants_locs = hydro_plants_df[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values
    points_nuts_ds = match_points_to_regions(hydro_plants_locs, shapes).dropna()

    def add_region(lon, lat):
        try:
            nuts_ = points_nuts_ds[lon, lat]
            # Need the if because some points are exactly at the same position
            return nuts_ if isinstance(nuts_, str) else nuts_.iloc[0]
        except KeyError:
            return None

    hydro_plants_df["NUTS"] = hydro_plants_df[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)
    hydro_plants_df = hydro_plants_df[~hydro_plants_df['NUTS'].isnull()]

    # ROR plants
    # Get ROR plants and aggregate capacity (to GW) per NUTS region
    ror_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Run-Of-River']
    ror_nuts_capacity_ds = ror_plants_df.groupby(ror_plants_df['NUTS'])['Capacity'].sum() * 1e-3

    # STO plants
    # Get STO plants and aggregate capacity (to GW) per NUTS region
    hydro_sto = hydro_plants_df[hydro_plants_df['Technology'] == 'Reservoir']
    sto_nuts_capacity_df = hydro_sto.groupby(hydro_sto['NUTS'])['Capacity'].sum().to_frame() * 1e-3

    # PHS plants
    # Get PHS plants and aggregate capacity (to GW) per NUTS region
    php_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Pumped Storage']
    php_nuts_capacity_df = php_plants_df.groupby(php_plants_df['NUTS'])['Capacity'].sum().to_frame() * 1e-3

    # Compute energy capacity of STO plants between NUTS regions proportionally to their GW capacity
    for nuts in sto_nuts_capacity_df.index:

        country_storage_potential = hydro_storage_energy_cap_ds[nuts[:2]]
        country_capacity_potential = \
            sto_nuts_capacity_df.loc[sto_nuts_capacity_df.index.str.startswith(nuts[:2]), "Capacity"].sum()
        sto_nuts_capacity_df.loc[nuts, "Energy"] = \
            (sto_nuts_capacity_df.loc[nuts, "Capacity"] / country_capacity_potential) * country_storage_potential

    # Compute energy capacity of PHS plants based on Eurelectric 2011 study.
    # Assumed 12h duration where data is missing. Note: Optimistic values from an existing storage perspective.
    countries = [item[:2] for item in php_nuts_capacity_df.index]
    php_nuts_capacity_df["Energy"] = php_nuts_capacity_df["Capacity"] * php_durations_ds.loc[countries].values

    # Build capacities DataFrame.
    capacities_df = build_capacities_frame(shapes, ror_nuts_capacity_ds, sto_nuts_capacity_df, php_nuts_capacity_df)
    logger.info('Capacities computed.')

    # Compute time-series of inflow for STO (in GWh) and ROR (per unit of capacity)
    runoff_dataset = read_runoff_data(resolution, timestamps)

    # Find to which nuts region each of the runoff points belong
    # TODO: instead of doing that here, we could apply it twice with shapes filtered on nuts_with_ror_index
    #  and nuts_with_sto_index, but I want first to refactor match_points_to_region. David: I would not do that,
    #  since it would lead to quite a duplication of the same call across the two technologies (as they share
    #  similar locations). Takes a long time to run this, especially with high-res ERA5 and NUTS3 shapes.
    points_to_regions_fn = join(dirname(abspath(__file__)),
                                f"../../../data/hydro/generated/mapping_points_regions_{topology_unit}_{resolution}.p")
    if isfile(points_to_regions_fn):
        points_nuts_ds = pickle.load(open(points_to_regions_fn, 'rb'))
    else:
        points_nuts_ds = match_points_to_regions(runoff_dataset.locations.values, shapes).dropna()
        pickle.dump(points_nuts_ds, open((points_to_regions_fn), 'wb'))
    logger.info('Runoff measurement points mapped to regions shapes.')

    # ROR inflow (p.u.)
    df_ror = pd.DataFrame(index=timestamps, columns=capacities_df[capacities_df['ROR_CAP [GW]'].notnull()].index)
    for nuts in df_ror.columns:
        nuts_points = points_nuts_ds[points_nuts_ds == nuts].index.to_list()
        if nuts_points:
            df_ror[nuts] = compute_ror_series(runoff_dataset, nuts_points, flood_event_threshold)
    df_ror.dropna(axis=1, inplace=True)
    capacities_df.loc[~capacities_df.index.isin(df_ror.columns), 'ROR_CAP [GW]'] = None
    missing_ror = capacities_df.loc[~capacities_df.index.isin(df_ror.columns)]['ROR_CAP [GW]'].dropna().sum()
    logger.info(f'ROR capacity factors computed. '
                f'{missing_ror} GW removed because of ERA5 point unavailability in {topology_unit} regions.')

    # STO inflow (in GWh)
    df_sto = pd.DataFrame(index=timestamps, columns=capacities_df[capacities_df['STO_CAP [GW]'].notnull()].index)
    for nuts in df_sto.columns:
        nuts_points = points_nuts_ds[points_nuts_ds == nuts].index.to_list()
        if nuts_points:
            df_sto[nuts] = compute_sto_inflows(runoff_dataset, nuts_points, sto_unit_area).values
    df_sto.dropna(axis=1, inplace=True)
    capacities_df.loc[~capacities_df.index.isin(df_sto.columns), 'STO_CAP [GW]'] = None
    capacities_df.loc[~capacities_df.index.isin(df_sto.columns), 'STO_EN_CAP [GWh]'] = None
    missing_sto_gw = capacities_df.loc[~capacities_df.index.isin(df_sto.columns)]['STO_CAP [GW]'].dropna().sum()
    missing_sto_gwh = capacities_df.loc[~capacities_df.index.isin(df_sto.columns)]['STO_EN_CAP [GWh]'].dropna().sum()
    logger.info(f'STO inflows computed., '
                f'{missing_sto_gw} GW / {missing_sto_gwh} GWh removed because '
                f'of ERA5 point unavailability in {topology_unit} regions.')

    sto_multipliers_ds = compute_sto_multipliers(timestamps, df_ror, df_sto, capacities_df)
    for nuts in df_sto.columns:
        df_sto[nuts] *= sto_multipliers_ds[nuts[:2]]
    logger.info('STO multipliers computed.')

    capacities_df = capacities_df.dropna(how='all')

    # Saving files
    save_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    capacities_save_fn = f"{save_dir}hydro_capacities_per_{topology_unit}.csv"
    capacities_df.to_csv(capacities_save_fn)
    ror_save_fn = f"{save_dir}hydro_ror_time_series_per_{topology_unit}_pu.csv"
    df_ror.to_csv(ror_save_fn)
    sto_save_fn = f"{save_dir}hydro_sto_inflow_time_series_per_{topology_unit}_GWh.csv"
    df_sto.to_csv(sto_save_fn)
    sto_multipliers_save_fn = f"{save_dir}hydro_sto_multipliers_per_{topology_unit}.csv"
    sto_multipliers_ds.to_csv(sto_multipliers_save_fn)
    logger.info('Files saved to disk.')

if __name__ == '__main__':

    nuts_type = 'NUTS3'
    resolution = 0.28125 #0.5
    sto_unit_area = 628. #1225
    ror_flood_threshold = 0.8

    start = datetime(2014, 1, 1, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)
    timestamps_ = pd.date_range(start, end, freq='H')

    generate_eu_hydro_files(resolution, nuts_type, timestamps_, ror_flood_threshold, sto_unit_area)