from typing import List, Tuple
from datetime import datetime
from os import listdir
from os.path import join, abspath, dirname, isfile

import pickle
import pandas as pd
import xarray as xr

from src.data.geographics import match_points_to_regions, get_nuts_shapes, get_natural_earth_shapes, \
    replace_uk_el_codes, convert_country_codes
from src.data.generation import get_hydro_production

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: document how input files were created or automate their generation


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


def build_phs_data(php_capacity_ds: pd.Series) -> pd.DataFrame:
    """
    Compute total PHS power (GW) and energy (GWh) capacities for a series of regions.

    Parameters
    ----------
    php_capacity_ds: pd.Series
        Series containing PHS power (GW) capacity per plant, indexed by the region in which the plant is located.

    Returns
    -------
    php_capacity_df: pd.DataFrame
        Dataframe containing PHS power (GW) and energy (GWh) capacity
    """
    php_capacity_df = php_capacity_ds.groupby(php_capacity_ds.index).sum().to_frame() * 1e-3

    # Compute energy capacity of PHS plants based on Eurelectric 2011 study.
    # Assumed 12h duration where data is missing. Note: Optimistic values from an existing storage perspective.
    php_duration_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/php_duration.csv")
    php_durations_ds = pd.read_csv(php_duration_fn, squeeze=True, index_col=0)
    countries = replace_uk_el_codes([item[:2] for item in php_capacity_df.index])
    php_capacity_df["Energy"] = php_capacity_df["Capacity"] * php_durations_ds.loc[countries].values

    return php_capacity_df


def compute_ror_series(runoff_dataset: xr.Dataset, region_points: List[Tuple[float, float]],
                       flood_event_threshold: float) -> pd.DataFrame:
    """
    Computing ROR p.u. time series as directly proportional to runoff for a given grid cell/area.

    Parameters
    ----------
    runoff_dataset: xarray.Dataset
        Contains runoff data, in this case expressed in m.
    region_points: List[Tuple[float, float]]
        List of points (lon, lat) within a region.
    flood_event_threshold: float
        Quantile clipping runoff time series (stems from the assumption that ROR plants
        are designed for a, e.g. p80 flow).

    Returns
    -------
    ts_norm: pd.DataFrame
        Time series of p.u. capacity factors for ROR plants.
    """
    # Mean of runoffs over all points within the region.
    ts = runoff_dataset.ro.sel(locations=region_points).mean(dim='locations').load()
    # Compute quantile from xarray object.
    q = ts.quantile(q=flood_event_threshold)
    # Clipping according to the flood_event_threshold
    ts[ts > q] = q
    # Normalizing for p.u. representation.
    return ts / ts.max()


def build_ror_data(ror_capacity_ds: pd.Series, timestamps: pd.DatetimeIndex,
                   runoff_dataset: xr.Dataset, runoff_points_region_ds: pd.Series,
                   flood_event_threshold: float) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute total ROR capacities (in GW) and inflow (p.u. of capacity) for a series of regions.

    Parameters
    ----------
    ror_capacity_ds: pd.Series
        Series containing ROR power (GW) capacity per plant, indexed by the region in which the plant is located.
    timestamps: pd.DatetimeIndex
        Time stamps over which the inflows must be computed.
    runoff_dataset: xr.Dataset
        ERA5 runoff dataset
    runoff_points_region_ds: pd.Series
        Indicates in which region each ERA5 point falls.
    flood_event_threshold: float
         Quantile clipping runoff time series (see compute_ror_series for usage).

    Returns
    -------
    ror_capacity_ds: pd.Series
        Series containing ROR power (GW) capacity per region.
    ror_inflows_df: pd.DataFrame
        ROR inflow time-series (p.u. of power capacity) for each region.
    """

    ror_capacity_ds = ror_capacity_ds.groupby(ror_capacity_ds.index).sum() * 1e-3

    ror_inflows_df = pd.DataFrame(index=timestamps, columns=ror_capacity_ds.index)
    for region in ror_capacity_ds.index:
        points = runoff_points_region_ds[runoff_points_region_ds == region].index.to_list()
        if points:
            ror_inflows_df[region] = compute_ror_series(runoff_dataset, points, flood_event_threshold)
    ror_inflows_df.dropna(axis=1, inplace=True)
    missing_inflows_indexes = ~ror_capacity_ds.index.isin(ror_inflows_df.columns)
    missing_ror = ror_capacity_ds.loc[missing_inflows_indexes].dropna().sum()
    ror_capacity_ds = ror_capacity_ds[ror_inflows_df.columns]
    logger.info(f'ROR capacity factors computed. '
                f'{missing_ror} GW removed because of ERA5 point unavailability in regions.')

    return ror_capacity_ds, ror_inflows_df


def compute_sto_inflows(runoff_dataset: xr.Dataset, points: List[Tuple[float, float]],
                        unit_area: float) -> pd.DataFrame:
    """
     Computing STO inflow time series (GWh).

     Parameters
     ----------
     runoff_dataset: xarray.Dataset
         Contains runoff data, in this case expressed in m.
     points: List[Tuple[float, float]]
         List of points (lon, lat).
     unit_area: float
         Area of one grid cell, depending on the runoff data resolution used.
         E.g., assuming an average distance of 20km between points (at 50deg latitude), a square area equals 628km2.

     Returns
     -------
     ts_gwh: pd.DataFrame
         Time series of STO inflows.
     """

    # Summation of runoffs over all points within the region.
    ts = runoff_dataset.ro.sel(locations=points).sum(dim='locations')
    # Convert from the runoff unit (m) to some equivalent water volume (m3). Convert from J to Wh and then to GWh.
    g = 9.81  # Gravitational constant [m/s2].
    rho = 1000.  # Water density [kg/m3].
    ts_cm = ts * (unit_area * 1e6) * g * rho * (1 / 3600) * 1e-9

    return ts_cm


def compute_countries_sto_multipliers(years: List[int], countries: List[str], sto_inflows_df: pd.DataFrame,
                                      ror_inflows_df: pd.DataFrame, ror_capacity_ds: pd.Series) -> pd.Series:
    """
     Computing STO multipliers mapping cell runoff to approximated hourly-sampled reservoir inflows.

     Parameters
     ----------

     years: List[int]
        List of years.
     countries: List[str]
        ISO codes of the countries for which we want to obtain STO multipliers.
     sto_inflows_df: pd.DataFrame
        Data frame with STO (GWh) inflow time series for each geographical unit across the time horizon considered.
     ror_inflows_df: pd.DataFrame
        Data frame with ROR (p.u.) capacity factors for each geographical unit across the time horizon considered.
     ror_capacity_ds: pd.Series
        Series with ROR hydro capacities (GW) for each geographical unit considered.

     Returns
     -------
     sto_multipliers_ds: pd.Series
         STO multipliers per country.
     """

    # Compute yearly per country ror electricity production
    ror_inflows_yearly = ror_inflows_df.groupby(ror_inflows_df.index.year).sum()
    ror_production_yearly = ror_inflows_yearly.multiply(ror_capacity_ds.dropna(), axis=1).transpose()
    ror_production_yearly_per_country = ror_production_yearly.groupby(ror_production_yearly.index.str[:2]).sum()

    # Get total hydro-electric production and remove ROR production to get STO production
    sto_production_yearly_per_country = get_hydro_production(years=years, countries=countries)
    countries_with_ror = set(countries).intersection(set(ror_production_yearly_per_country.index))
    sto_production_yearly_per_country.loc[countries_with_ror] -= \
        ror_production_yearly_per_country.loc[countries_with_ror]
    # For some countries (like LV and IE), computed ROR potential is bigger than the Eurostat total hydro generation
    # leading to negative STO production values so we clip it.
    sto_production_per_country = sto_production_yearly_per_country.clip(lower=0.).sum(axis=1)

    sto_inflows_per_country = sto_inflows_df.sum().groupby(sto_inflows_df.columns.str[:2]).sum()
    sto_multipliers_ds = sto_production_per_country/sto_inflows_per_country

    return sto_multipliers_ds


def build_sto_data(sto_capacity_ds: pd.Series, timestamps: pd.DatetimeIndex,
                   runoff_dataset: xr.Dataset, runoff_points_region_ds: pd.Series,
                   ror_capacity_ds: pd.Series, ror_inflows_df: pd.DataFrame,
                   sto_unit_area: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute total STO power (GW) and energy( (GWh) capacities and inflow (GWh) for a series of regions.

    Parameters
    ----------
    sto_capacity_ds: pd.Series
        Series containing STO power (GW) capacity per plant, indexed by the region in which the plant is located.
    timestamps: pd.DatetimeIndex
        Time stamps over which the inflows must be computed.
    runoff_dataset: xr.Dataset
        ERA5 runoff dataset
    runoff_points_region_ds: pd.Series
        Indicates in which region each ERA5 point falls.
    ror_inflows_df: pd.DataFrame
        Data frame with ROR (p.u.) capacity factors for each geographical unit across the time horizon considered.
    ror_capacity_ds: pd.Series
        Series with ROR hydro capacities (GW) for each geographical unit considered.
    sto_unit_area: float
        Average area of reanalysis grid cell, used for estimating inflows in reservoir-based hydro plants.
        Depends on the resolution of the underlying runoff data.

    Returns
    -------
    sto_capacity_df: pd.DataFrame
        Series containing STO power (GW) capacity per region.
    sto_inflows_df: pd.DataFrame
        STO inflow time-series (GWh) for each region.
    sto_multipliers_ds: pd.Series
         STO multipliers per country.
    """

    sto_capacity_df = sto_capacity_ds.groupby(sto_capacity_ds.index).sum().to_frame() * 1e-3

    # Compute energy capacity of STO plants between regions proportionally to their GW capacity
    source_dir = join(dirname(abspath(__file__)), "../../../data/hydro/source/")
    hydro_storage_capacities_fn = f"{source_dir}hydro_storage_capacities_updated.csv"
    hydro_storage_energy_cap_ds = pd.read_csv(hydro_storage_capacities_fn, sep=';',
                                              usecols=['Country', 'E_store[TWh]'], index_col='Country',
                                              squeeze=True) * 1e3
    for nuts in sto_capacity_df.index:

        iso_code = replace_uk_el_codes([nuts[:2]])[0]
        country_storage_potential = hydro_storage_energy_cap_ds[iso_code]
        country_capacity_potential = \
            sto_capacity_df.loc[sto_capacity_df.index.str.startswith(nuts[:2]), "Capacity"].sum()
        sto_capacity_df.loc[nuts, "Energy"] = \
            (sto_capacity_df.loc[nuts, "Capacity"] / country_capacity_potential) * country_storage_potential

    # STO inflow (in GWh)
    sto_inflows_df = pd.DataFrame(index=timestamps, columns=sto_capacity_df.index)
    for region in sto_capacity_df.index:
        points = runoff_points_region_ds[runoff_points_region_ds == region].index.to_list()
        if points:
            sto_inflows_df[region] = compute_sto_inflows(runoff_dataset, points, sto_unit_area).values
    sto_inflows_df.dropna(axis=1, inplace=True)
    missing_inflows_indexes = ~sto_capacity_df.index.isin(sto_inflows_df.columns)
    missing_sto_gw = sto_capacity_df.loc[missing_inflows_indexes]['Capacity'].dropna().sum()
    missing_sto_gwh = sto_capacity_df.loc[missing_inflows_indexes]['Energy'].dropna().sum()
    sto_capacity_df = sto_capacity_df.loc[sto_inflows_df.columns]
    logger.info(f'STO inflows computed., '
                f'{missing_sto_gw} GW / {missing_sto_gwh} GWh removed because '
                f'of ERA5 point unavailability in regions.')

    # Compute STO multipliers
    years = [y for y in timestamps.year.unique()]
    countries = replace_uk_el_codes(list(set([code[:2] for code in sto_inflows_df.columns])))
    sto_multipliers_ds = compute_countries_sto_multipliers(years, countries, sto_inflows_df,
                                                           ror_inflows_df, ror_capacity_ds)

    # Apply multipliers to STO inflows
    for nuts in sto_inflows_df.columns:
        sto_inflows_df[nuts] *= sto_multipliers_ds[nuts[:2]]

    return sto_capacity_df, sto_inflows_df, sto_multipliers_ds


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
    else:  # topology in ['NUTS2', 'NUTS3']
        shapes = get_nuts_shapes(topology_unit[-1:])
    countries = replace_uk_el_codes(list(set([code[:2] for code in shapes.index])))

    # Runoff data
    runoff_dataset = read_runoff_data(resolution, timestamps)
    # Find to which nuts region each of the runoff points belong
    points_to_regions_fn = join(dirname(abspath(__file__)),
                                f"../../../data/hydro/generated/mapping_points_regions_{topology_unit}_{resolution}.p")
    if isfile(points_to_regions_fn):
        runoff_points_region_ds = pickle.load(open(points_to_regions_fn, 'rb'))
    else:
        # TODO: maybe no need to save it anymore?
        runoff_points_region_ds = match_points_to_regions(runoff_dataset.locations.values, shapes, keep_outside=False).dropna()
        pickle.dump(runoff_points_region_ds, open(points_to_regions_fn, 'wb'))
    logger.info('Runoff measurement points mapped to regions shapes.')

    # Read hydro plants data from powerplantmatching tool from which we can retrieve capacity (in MW)
    hydro_plants_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/pp_fresna_hydro_updated.csv")
    hydro_plants_df = pd.read_csv(hydro_plants_fn, sep=';', usecols=[0, 1, 3, 5, 6, 8, 13, 14], index_col=0)

    # TODO: this is shit, anyway need to deal with this ppm from file thing
    def correct_countries(c: str):
        if c == "Macedonia, Republic of":
            return "North Macedonia"
        if c == "Czech Republic":
            return "Czechia"
        return c

    hydro_plants_df["Country"] = hydro_plants_df["Country"].apply(lambda c: correct_countries(c))
    hydro_plants_df["Country"] = convert_country_codes(hydro_plants_df["Country"].values, 'name', 'alpha_2', True)
    hydro_plants_df = hydro_plants_df[hydro_plants_df["Country"].isin(countries)]

    print(hydro_plants_df[hydro_plants_df["lon"] == 22.469872])

    # Find to which region each plant belongs
    hydro_plants_locs = hydro_plants_df[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values
    plants_region_ds = match_points_to_regions(hydro_plants_locs, shapes).dropna()

    # TODO: should we create a function in generation that does this?
    def add_region(lon, lat):
        try:
            region_code = plants_region_ds[lon, lat]
            # Need the if because some points are exactly at the same position
            return region_code if isinstance(region_code, str) else region_code.iloc[0]
        except KeyError:
            # TODO: why would there be a key error?
            return None

    hydro_plants_df["region_code"] = hydro_plants_df[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)
    hydro_plants_df = hydro_plants_df[~hydro_plants_df['region_code'].isnull()]

    # Build ROR data
    ror_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Run-Of-River']
    ror_capacity_ds, ror_inflows_df = build_ror_data(ror_plants_df.set_index(["region_code"])["Capacity"], timestamps,
                                                     runoff_dataset, runoff_points_region_ds, flood_event_threshold)

    # Build STO data
    sto_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Reservoir']
    sto_capacity_df, sto_inflows_df, sto_multipliers_ds = \
        build_sto_data(sto_plants_df.set_index(["region_code"])["Capacity"], timestamps,
                       runoff_dataset, runoff_points_region_ds, ror_capacity_ds, ror_inflows_df, sto_unit_area)

    # Build PHS data
    php_plants_df = hydro_plants_df[hydro_plants_df['Technology'] == 'Pumped Storage']
    php_capacity_df = build_phs_data(php_plants_df.set_index(["region_code"])["Capacity"])

    # Merge capacities DataFrame.
    capacities_df = pd.concat([ror_capacity_ds, sto_capacity_df, php_capacity_df], axis=1, sort=True).round(3)
    capacities_df.columns = ['ROR_CAP [GW]', 'STO_CAP [GW]', 'STO_EN_CAP [GWh]', 'PSP_CAP [GW]', 'PSP_EN_CAP [GWh]']

    # Saving files
    save_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    capacities_df.to_csv(f"{save_dir}hydro_capacities_per_{topology_unit}.csv")
    ror_inflows_df.to_csv(f"{save_dir}hydro_ror_time_series_per_{topology_unit}_pu.csv")
    sto_inflows_df.to_csv(f"{save_dir}hydro_sto_inflow_time_series_per_{topology_unit}_GWh.csv")
    sto_multipliers_ds.to_csv(f"{save_dir}hydro_sto_multipliers_per_{topology_unit}.csv")
    logger.info('Files saved to disk.')


if __name__ == '__main__':

    nuts_type_ = 'NUTS2'
    resolution_ = 0.5  # 0.28125
    sto_unit_area_ = 1225  # 628. # TODO: this is not great
    ror_flood_threshold_ = 0.8  # TODO: this value should be saved somehow with the files

    start = datetime(2018, 12, 31, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)
    timestamps_ = pd.date_range(start, end, freq='H')

    generate_eu_hydro_files(resolution_, nuts_type_, timestamps_, ror_flood_threshold_, sto_unit_area_)
