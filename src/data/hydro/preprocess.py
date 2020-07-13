from typing import List, Tuple
from datetime import datetime
from os import listdir
from os.path import join, abspath, dirname, isdir

import yaml
import geopy.distance
import pandas as pd
import xarray as xr
import xlrd

import numpy as np
import geopandas as gpd

from src.data.geographics import match_points_to_regions, get_nuts_shapes, get_natural_earth_shapes, \
    replace_iso2_codes, convert_country_codes, revert_old_country_names, convert_old_country_names
from src.data.generation import get_hydro_production, get_powerplants, match_powerplants_to_regions

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


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
    assert isdir(runoff_dir), f"Error: No data found for resolution {resolution} (directory {runoff_dir} not found)."
    runoff_files = [join(runoff_dir, fn) for fn in listdir(runoff_dir) if fn.endswith(".nc")]
    runoff_dataset = xr.open_mfdataset(runoff_files, combine='by_coords')
    runoff_dataset = runoff_dataset.stack(locations=('longitude', 'latitude'))
    missing_ts = set(timestamps) - set(pd.to_datetime(runoff_dataset.time.values))
    assert not missing_ts, f"Error: Data is not available for following timestamps: {sorted(list(missing_ts))}."
    runoff_dataset = runoff_dataset.sel(time=timestamps)

    # Add area to dataset
    runoff_dataset = add_runoff_area(runoff_dataset, resolution)

    return runoff_dataset


def add_runoff_area(dataset: xr.Dataset, resolution: float) -> xr.Dataset:
    """
     Computing cell area for each (lon, lat) pair.

     Parameters
     ----------
     dataset: xarray.Dataset
         Contains runoff data, in this case expressed in m.
     resolution: float
         Runoff data spatial resolution.

     Returns
     -------
     dataset: xr.Dataset
         Same input dataset with 'area' variable added.
     """
    # Get distance between two latitudes. Does not depend on the geo-location, thus an arbitrary point is considered.
    p1 = (0.0, 0.0)
    p2 = (p1[0] + resolution, 0.0)
    dist_latitude = geopy.distance.distance(p1, p2).km

    lon = dataset['longitude'].values
    lat = dataset['latitude'].values
    # Get vectors of 'bordering' longitudes.
    lonplus = lon + resolution / 2
    lonmin = lon - resolution / 2

    # Initialize a zero-vector and compute distances between longitude pairs.
    dist = np.zeros(len(lat))
    for idx in np.arange(len(lat)):
        dist[idx] = geopy.distance.distance((lat[idx], lonplus[idx]), (lat[idx], lonmin[idx])).km

    # Compute cell area and attach it to the dataset.
    dataset['area'] = ('locations', dist * dist_latitude)

    return dataset


def get_phs_storage_capacities(phs_capacity_df: pd.DataFrame, default_phs_duration: float) -> pd.DataFrame:
    """
     Assigning storage capacities (MWh) to PHS plants.

     Parameters
     ----------
     phs_capacity_df: pd.Series
        Series containing PHS rated power data (indexed by NUTS code).
     default_phs_duration: float
        Default duration for PHS plants, in case data does not exist for plant or country.

     Returns
     -------
     phs_cap_storage: pd.DataFrame
        Frame containing PHS power and energy ratings.

     """

    phs_geth_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/Geth_2015_EU_PHS_review.xlsx")
    phs_geth_all = pd.read_excel(phs_geth_fn, sheet_name='overall', index_col=0).dropna(subset=['Estor [GWh]'])

    # Iterate through all PHS plants
    for idx in phs_capacity_df.index:
        # Retrieve ISO2 country code for index checks
        # code = convert_country_codes([phs_capacity_df.loc[idx, 'Country']], 'name', 'alpha_2', True)[0]
        iso_code = phs_capacity_df.loc[idx, 'ISO2']
        if iso_code in phs_geth_all.index:
            # Compute country-specific PHS duration, based on Geth
            default_duration = phs_geth_all.loc[iso_code, 'Estor [GWh]'] / phs_geth_all.loc[iso_code, 'Pd,nom [GW]']
        else:
            # If country data is missing, impose default value
            default_duration = default_phs_duration
        # If ISO2 in file sheets (detailed country data exists), read file...
        try:
            phs_geth = pd.read_excel(phs_geth_fn, sheet_name=iso_code, index_col='JRC_HPDB_id')
            if idx in phs_geth.index and not np.isnan(phs_geth.loc[idx, 'Estor [GWh]']):
                # If storage content is provided, fetch it directly...
                phs_capacity_df.loc[idx, 'Energy'] = phs_geth.loc[idx, 'Estor [GWh]'] * 1e3
            else:
                # ... otherwise, consider default duration.
                phs_capacity_df.loc[idx, 'Energy'] = phs_capacity_df.loc[idx, 'Capacity'] * default_duration
        # ...else, impose default duration.
        except xlrd.biffh.XLRDError:
            phs_capacity_df.loc[idx, 'Energy'] = phs_capacity_df.loc[idx, 'Capacity'] * default_duration

    return phs_capacity_df[['Name', 'Capacity', 'Energy', 'region_code']]


def build_phs_data(phs_plants_df: pd.DataFrame, default_phs_duration: float) -> pd.DataFrame:
    """
    Compute total PHS power (GW) and energy (GWh) capacities for a series of regions.

    Parameters
    ----------
    phs_plants_df: pd.DataFrame
        Frame containing PHS power plant data.
    default_phs_duration: float
        Default duration for PHS plants.

    Returns
    -------
    php_capacity_df: pd.DataFrame
        Dataframe containing PHS power (GW) and energy (GWh) capacity
    """

    phs_storage_df = get_phs_storage_capacities(phs_plants_df, default_phs_duration)
    phs_capacity_df = phs_storage_df.groupby(phs_storage_df['region_code']).sum() * 1e-3

    return phs_capacity_df


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
                   runoff_dataset: xr.Dataset, runoff_points_region_ds: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
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

    Returns
    -------
    ror_capacity_ds: pd.Series
        Series containing ROR power (GW) capacity per region.
    ror_inflows_df: pd.DataFrame
        ROR inflow time-series (p.u. of power capacity) for each region.
    """

    ror_thresholds_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/ror_flood_event_thresholds.csv")
    ror_thresholds = pd.read_csv(ror_thresholds_fn, index_col=0)

    ror_capacity_ds = ror_capacity_ds.groupby(ror_capacity_ds.index).sum() * 1e-3

    ror_inflows_df = pd.DataFrame(index=timestamps, columns=ror_capacity_ds.index)
    for region in ror_capacity_ds.index:
        points = runoff_points_region_ds[runoff_points_region_ds == region].index.to_list()
        flood_event_threshold = ror_thresholds.loc[replace_iso2_codes([region[:2]])[0], 'value']
        if points:
            ror_inflows_df[region] = compute_ror_series(runoff_dataset, points, flood_event_threshold)
    ror_inflows_df.dropna(axis=1, inplace=True)
    missing_inflows_indexes = ~ror_capacity_ds.index.isin(ror_inflows_df.columns)
    missing_ror = ror_capacity_ds.loc[missing_inflows_indexes].dropna().sum()
    ror_capacity_ds = ror_capacity_ds[ror_inflows_df.columns]
    logger.info(f'ROR capacity factors computed. '
                f'{missing_ror} GW removed because of ERA5 point unavailability in regions.')

    return ror_capacity_ds, ror_inflows_df


def get_country_storage_from_grand(country_name: str) -> float:
    """
     Estimating STO energy storage capacity (in GWh) per country based on the GRanD dataset.

     Parameters
     ----------
     country_name: str
         Country name.

     Returns
     -------
     float
         Estimated energy storage potential (in GWh) for the country of interest.

     Notes
     -----
     If the country name is not present in the database, returns 0.

    """

    source_dir = join(dirname(abspath(__file__)), "../../../data/hydro/source/GDW/GRanD_Version_1_3/")
    grand_reservoirs_fn = f"{source_dir}GRanD_reservoirs_v1_3.shp"
    reservoirs_df = pd.DataFrame(gpd.read_file(grand_reservoirs_fn)).set_index('GRAND_ID')
    # See get_nuts_storage_distribution_from_grand for explanation
    reservoirs_df = reservoirs_df[reservoirs_df['RES_NAME'] != 'Vanern']

    # Filtering out reservoirs whose purpose is not for hydro power generation.
    reservoirs_hydropower_df = reservoirs_df[reservoirs_df['USE_ELEC'].isin(['Main', 'Sec', 'Major'])]
    # Filtering out reservoirs outside the country of interest.
    reservoirs_hydropower_df = reservoirs_hydropower_df[reservoirs_hydropower_df['COUNTRY'] == country_name]

    # Filtering out reservoirs whose dam height data is not available (value of -99 in the dataset).
    reservoirs_hydropower_df = reservoirs_hydropower_df[reservoirs_hydropower_df['DAM_HGT_M'] > 0.]
    # Computing equivalent energy content (GWh) via hydro power equation.
    g = 9.81  # Gravitational constant [m/s2].
    rho = 1000.  # Water density [kg/m3].
    reservoirs_hydropower_df['EN_POT'] = \
        rho * g * reservoirs_hydropower_df['DAM_HGT_M'] * reservoirs_hydropower_df['CAP_MCM'] / (3.6 * 1e6)

    return reservoirs_hydropower_df['EN_POT'].sum()


def get_nuts_storage_distribution_from_grand(nuts_codes: List[str]) -> pd.Series:
    """
     Estimating STO energy storage distribution per NUTS sub-divisions.

     Parameters
     ----------
     nuts_codes: List[str]
         List of NUTS (e.g., "NUTS2", "NUTS3") codes for which data is retrieved.

     Returns
     -------
     storage_distribution_ds: pd.DataFrame
         DataFrame containing STO energy storage distribution keys per NUTS regions.

    """

    assert len(nuts_codes) != 0, "Error: Empty list of NUTS codes."

    # Read GRanD database
    source_dir = join(dirname(abspath(__file__)), "../../../data/hydro/source/GDW/GRanD_Version_1_3/")
    grand_reservoirs_fn = f"{source_dir}GRanD_reservoirs_v1_3.shp"
    reservoirs_df = pd.DataFrame(gpd.read_file(grand_reservoirs_fn)).set_index('GRAND_ID')
    # A particular reservoir is manually removed (others could follow). The Vanern lake (SE) is labeled as a reservoir
    # with hydro power activities, though information online suggests otherwise. Its presence in the associated NUTS
    # region leads to inconsistencies in the distribution of Swedish storage potential across the country.
    reservoirs_df = reservoirs_df[reservoirs_df['RES_NAME'] != 'Vanern']

    # Get NUTS0, ISO2 and countries names of the countries which NUTS regions are part of
    nuts0_codes = list(set([nuts[:2] for nuts in nuts_codes]))
    iso2_codes = replace_iso2_codes(nuts0_codes)
    countries_names = convert_country_codes(iso2_codes, 'alpha_2', 'name', True)

    # Get NUTS region shapes
    shapes = get_nuts_shapes(str(len(nuts_codes[0]) - 2), nuts_codes)
    shapes_countries = replace_iso2_codes([c[:2] for c in shapes.index])

    # Filtering out reservoirs whose purpose is not for hydro power generation.
    reservoirs_df["COUNTRY"] = reservoirs_df["COUNTRY"].apply(convert_old_country_names)
    reservoirs_hydropower_df = reservoirs_df[(reservoirs_df['USE_ELEC'].isin(['Main', 'Sec', 'Major'])) &
                                             (reservoirs_df['COUNTRY'].isin(countries_names))].copy()

    # Associating each plant to the corresponding NUTS region
    reservoirs_hydropower_df["ISO2"] = \
        convert_country_codes(reservoirs_hydropower_df['COUNTRY'], 'name', 'alpha_2', True)
    reservoirs_hydropower_df["region_code"] = \
        match_powerplants_to_regions(reservoirs_hydropower_df.rename(columns={'LONG_DD': 'lon', 'LAT_DD': 'lat'}),
                                     shapes, shapes_countries)
    reservoirs_hydropower_df = reservoirs_hydropower_df[~reservoirs_hydropower_df['region_code'].isnull()]

    # Aggregating storage capacity per NUTS region
    storage_by_nuts_ds = reservoirs_hydropower_df.groupby(by=reservoirs_hydropower_df['region_code'])['CAP_MCM'].sum()

    # Computing storage distribution keys per NUTS by dividing the capacity
    # per NUTS by the total capacity of all NUTS in the same country.
    storage_distribution_ds = pd.Series()
    for nuts0_code, iso2_code in zip(nuts0_codes, iso2_codes):
        storage_sum_per_country = \
            reservoirs_hydropower_df[reservoirs_hydropower_df['ISO2'] == iso2_code]['CAP_MCM'].sum()
        storage_ds_temp = storage_by_nuts_ds[storage_by_nuts_ds.index.str.contains(nuts0_code)]
        storage_ds_temp /= storage_sum_per_country
        storage_distribution_ds = storage_distribution_ds.append(storage_ds_temp)

    return storage_distribution_ds


def compute_storage_capacities(sto_capacity_ds: pd.Series) -> pd.Series:
    """
     Computing STO energy capacities (TWh) per unit region.

     Parameters
     ----------
     sto_capacity_ds: pd.Series
         DataFrame containing STO installed capacities per unit region (e.g., "countries", "NUTS3")

     Returns
     -------
     hydro_storage_energy_cap_ds: pd.Series
         DataFrame containing STO energy storage ratings.

    """
    source_dir = join(dirname(abspath(__file__)), "../../../data/hydro/source/")
    # Initially reading modelled data from Hartel et. al (2017)
    hydro_storage_capacities_fn = f"{source_dir}Hartel_2017_EU_hydro_storage_capacities.xlsx"
    hydro_storage_energy_cap_ds = pd.read_excel(hydro_storage_capacities_fn, skiprows=1,
                                                usecols=['ISO2', 'Eq. Storage'], index_col='ISO2', squeeze=True) * 1e3

    # Get storage capacities for countries which are not in the Hartel study
    iso2_codes = sorted(replace_iso2_codes(list(set([region_code[:2] for region_code in sto_capacity_ds.index]))))
    hydro_storage_energy_cap_ds = hydro_storage_energy_cap_ds.reindex(iso2_codes)
    for iso2_code in iso2_codes:
        # If c is not covered in the Hartel study...
        # if iso2_code not in hydro_storage_energy_cap_ds.index:
        if np.isnan(hydro_storage_energy_cap_ds[iso2_code]):
            country_name = revert_old_country_names(convert_country_codes([iso2_code], 'alpha_2', 'name', True)[0])
            try:
                # ...look-up for ENTSO-E reservoir data...
                hydro_storage_capacities_entsoe_fn = f"{source_dir}ENTSOE/Water Reservoirs and Hydro Storage Plants" \
                    f"_201412290000-201912300000_{iso2_code}.csv"
                hydro_storage_energy_cap = pd.read_csv(hydro_storage_capacities_entsoe_fn, index_col=0)
                max_storage = np.nanmax(np.nan_to_num(hydro_storage_energy_cap.values.flatten()))
                if max_storage > 0.:
                    hydro_storage_energy_cap_ds.loc[iso2_code] = max_storage * 1e-3
                else:
                    # ...if ENTSO-E data is missing (NaNs replaced by 0s), approximate storage via GRanD v1.3
                    hydro_storage_energy_cap_ds.loc[iso2_code] = get_country_storage_from_grand(country_name)
            except FileNotFoundError:
                # ...if ENTSO-E file is missing altogether, approximate storage via GRanD v1.3
                hydro_storage_energy_cap_ds.loc[iso2_code] = get_country_storage_from_grand(country_name)

    # If topology unit is "countries", return series directly
    if len(sto_capacity_ds.index[0]) == 2:
        return hydro_storage_energy_cap_ds.round(3)
    else:
        # If some NUTS-based topology in place, storage distribution among regions is done via GRanD v1.3
        storage_distribution_by_nuts = get_nuts_storage_distribution_from_grand(sto_capacity_ds.index)
        for nuts in storage_distribution_by_nuts.index:
            storage_distribution_by_nuts.loc[nuts] *= hydro_storage_energy_cap_ds.loc[replace_iso2_codes([nuts[:2]])[0]]
        hydro_storage_energy_cap_ds = storage_distribution_by_nuts.copy()
        return hydro_storage_energy_cap_ds.round(3)


def compute_sto_inflows(runoff_dataset: xr.Dataset, points: List[Tuple[float, float]]) -> pd.DataFrame:
    """
     Computing STO inflow time series (GWh).

     Parameters
     ----------
     runoff_dataset: xarray.Dataset
         Contains runoff data, in this case expressed in m.
     points: List[Tuple[float, float]]
         List of points (lon, lat).

     Returns
     -------
     ts_gwh: pd.DataFrame
         Time series of STO inflows.
     """
    region_runoff_dataset = runoff_dataset.sel(locations=points)
    # Convert from the runoff unit (m) to some equivalent energy storage in water (m3).
    region_ts = (region_runoff_dataset['ro'] * region_runoff_dataset['area']).sum(dim='locations')
    g = 9.81  # Gravitational constant [m/s2].
    rho = 1000.  # Water density [kg/m3].
    en_scalar = (1/3600) * 1e-9  # Converting from J to GWh
    area_scalar = 1e6  # Converting from km2 to m2
    ts_gwh = region_ts * g * rho * en_scalar * area_scalar

    return ts_gwh


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
                   ror_capacity_ds: pd.Series, ror_inflows_df: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute total STO power (GW) and energy( (GWh) capacities and inflow (GWh) for a series of regions.

    Parameters
    ----------
    sto_capacity_ds: pd.Series
        Series containing STO power (GW) capacity per plant, indexed by the region in which the plant is located.
    timestamps: pd.DatetimeIndex
        Time stamps over which the inflows must be computed.
    runoff_dataset: xr.Dataset
        ERA5 runoff dataset with area per dataset point.
    runoff_points_region_ds: pd.Series
        Indicates in which region each ERA5 point falls.
    ror_inflows_df: pd.DataFrame
        Data frame with ROR (p.u.) capacity factors for each geographical unit across the time horizon considered.
    ror_capacity_ds: pd.Series
        Series with ROR hydro capacities (GW) for each geographical unit considered.

    Returns
    -------
    sto_capacity_df: pd.DataFrame
        Series containing STO power (GW) capacity per region.
    sto_inflows_df: pd.DataFrame
        STO inflow time-series (GWh) for each region.
    sto_multipliers_ds: pd.Series
         STO multipliers per country.
    """
    sto_capacity_ds = sto_capacity_ds.groupby(sto_capacity_ds.index).sum() * 1e-3

    # Compute energy capacity of STO plants by regions
    storage_capacities = compute_storage_capacities(sto_capacity_ds)
    sto_capacity_df = pd.concat([sto_capacity_ds, storage_capacities], axis=1, ignore_index=True, sort=True)
    sto_capacity_df.columns = ['Capacity', 'Energy']
    # Some regions can end up without any storage capacities
    sto_capacity_df = sto_capacity_df.dropna()

    # STO inflow (in GWh)
    sto_inflows_df = pd.DataFrame(index=timestamps, columns=sto_capacity_df.index)
    for region in sto_capacity_df.index:
        points = runoff_points_region_ds[runoff_points_region_ds == region].index.to_list()
        if points:
            sto_inflows_df[region] = compute_sto_inflows(runoff_dataset, points).values
    sto_inflows_df.dropna(axis=1, inplace=True)
    missing_inflows_indexes = ~sto_capacity_df.index.isin(sto_inflows_df.columns)
    missing_sto_gw = sto_capacity_df.loc[missing_inflows_indexes]['Capacity'].dropna().sum()
    missing_sto_gwh = sto_capacity_df.loc[missing_inflows_indexes]['Energy'].dropna().sum()
    sto_capacity_df = sto_capacity_df.loc[sto_inflows_df.columns]
    logger.info(f'STO inflows computed., '
                f'{missing_sto_gw} GW / {missing_sto_gwh} GWh removed because '
                f'of ERA5 point unavailability in regions.')

    # Compute STO multipliers
    years = list(timestamps.year.unique())
    countries = replace_iso2_codes(list(set([code[:2] for code in sto_inflows_df.columns])))
    sto_multipliers_ds = compute_countries_sto_multipliers(years, countries, sto_inflows_df,
                                                           ror_inflows_df, ror_capacity_ds)

    # Apply multipliers to STO inflows
    for nuts in sto_inflows_df.columns:
        sto_inflows_df[nuts] *= sto_multipliers_ds[nuts[:2]]

    return sto_capacity_df, sto_inflows_df, sto_multipliers_ds


def generate_eu_hydro_files(resolution: float, topology_unit: str,
                            timestamps: pd.DatetimeIndex):
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

     """

    assert topology_unit in ["countries", "NUTS2", "NUTS3"], "Error: requested topology_unit not available."

    # Load shapes based on topology
    if topology_unit == 'countries':
        shapes = get_natural_earth_shapes()
    else:  # topology in ['NUTS2', 'NUTS3']
        shapes = get_nuts_shapes(topology_unit[-1:])
    shapes_countries = replace_iso2_codes([code[:2] for code in shapes.index])
    countries = sorted(list(set(shapes_countries)))

    tech_dir = join(dirname(abspath(__file__)), "../../../data/technologies/")
    tech_config = yaml.load(open(join(tech_dir, 'tech_config.yml')), Loader=yaml.FullLoader)

    # Runoff data
    runoff_dataset = read_runoff_data(resolution, timestamps)

    # Find to which nuts region each of the runoff points belong
    runoff_points_region_ds = \
        match_points_to_regions(runoff_dataset.locations.values, shapes, keep_outside=False).dropna()
    logger.info('Runoff measurement points mapped to regions shapes.')

    def add_region_code(pp_df: pd.DataFrame):
        if topology_unit == "countries":
            pp_df['region_code'] = pp_df["ISO2"]
        else:
            pp_df['region_code'] = match_powerplants_to_regions(pp_df, shapes, shapes_countries)
            pp_df = pp_df[~pp_df['region_code'].isnull()]
        return pp_df

    # Build ROR data
    # Get all ROR powerplants in the countries of interest and add region name
    logging.info('Building ROR data')
    ror_plants_df = get_powerplants('ror', countries)
    ror_plants_df = add_region_code(ror_plants_df)
    # Get capacity and inflow per region (for which inflow data exists)
    ror_capacity_ds, ror_inflows_df = build_ror_data(ror_plants_df.set_index(["region_code"])["Capacity"], timestamps,
                                                     runoff_dataset, runoff_points_region_ds)

    # Build STO data
    logging.info('Building STO data')
    sto_plants_df = get_powerplants('sto', countries)
    sto_plants_df = add_region_code(sto_plants_df)
    sto_capacity_df, sto_inflows_df, sto_multipliers_ds = \
        build_sto_data(sto_plants_df.set_index(["region_code"])["Capacity"], timestamps,
                       runoff_dataset, runoff_points_region_ds, ror_capacity_ds, ror_inflows_df)

    # Build PHS data
    logging.info('Building PHS data')
    default_phs_duration = tech_config['phs']['default_duration']

    phs_plants_df = get_powerplants('phs', countries)
    phs_plants_df = add_region_code(phs_plants_df)
    phs_capacity_df = build_phs_data(phs_plants_df, default_phs_duration)

    # Merge capacities DataFrame.
    capacities_df = pd.concat([ror_capacity_ds, sto_capacity_df, phs_capacity_df], axis=1, sort=True).round(3)
    capacities_df.columns = ['ROR_CAP [GW]', 'STO_CAP [GW]', 'STO_EN_CAP [GWh]', 'PSP_CAP [GW]', 'PSP_EN_CAP [GWh]']
    capacities_df.replace(0., np.nan, inplace=True)
    capacities_df.dropna(how='all', inplace=True)
    ror_inflows_df = ror_inflows_df[capacities_df['ROR_CAP [GW]'].dropna().index]
    sto_inflows_df = sto_inflows_df[capacities_df['STO_CAP [GW]'].dropna().index]

    # Saving files
    save_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    capacities_df.to_csv(f"{save_dir}hydro_capacities_per_{topology_unit}.csv")
    ror_inflows_df.to_csv(f"{save_dir}hydro_ror_time_series_per_{topology_unit}_pu.csv")
    sto_inflows_df.to_csv(f"{save_dir}hydro_sto_inflow_time_series_per_{topology_unit}_GWh.csv")
    sto_multipliers_ds.to_csv(f"{save_dir}hydro_sto_multipliers_per_{topology_unit}.csv", header=['multiplier'])
    logger.info('Files saved to disk.')


if __name__ == '__main__':

    nuts_type_ = 'NUTS3'
    resolution_ = 0.5  # 0.28125

    start = datetime(2018, 1, 1, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)
    timestamps_ = pd.date_range(start, end, freq='H')

    generate_eu_hydro_files(resolution_, nuts_type_, timestamps_)
