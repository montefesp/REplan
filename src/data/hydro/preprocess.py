import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import datetime
from os import listdir
from os.path import join, abspath, dirname


from src.data.geographics.manager import match_points_to_regions


# TODO: need to comment
def compute_ror_series(dataset_runoff, region_points, flood_event_threshold):

    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro.load()
    ts_clip = ts.clip(max=ts.quantile(q=flood_event_threshold).values)
    ts_norm = ts_clip / ts_clip.max()

    return ts_norm


# TODO: need to comment
def compute_sto_inflows(dataset_runoff, region_points, sto_multiplier):

    unit_area = 1e10  # TODO: why is this not given as argument?

    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro
    # assume a grid box area of 10'000 km2
    ts_cm = ts * unit_area
    ts_gwh = ts_cm * 9.81 * 1000 * (1/3.6) * 1e-12 * sto_multiplier

    # prod = ts_GWh.sum(dim='time').values*1e-3
    # exp = hydro_production_dict[region[:2]]

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

    # TODO: comment
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
    runoff_dir = join(dirname(abspath(__file__)), "../../../data/land_data/runoff")
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

    # STO inflow (in GWh)
    # Inflow multiplier (proportional to the water head) required to reach IEA production levels throughout the year.
    sto_multiplier_fn = join(dirname(abspath(__file__)), "../../../data/hydro/source/sto_multipliers.csv")
    sto_multipliers_ds = pd.read_csv(sto_multiplier_fn, squeeze=True, index_col=0)

    nuts_with_sto_index = capacities_df[capacities_df['STO_CAP [GW]'].notnull()].index
    df_sto = pd.DataFrame(index=timestamps, columns=nuts_with_sto_index)
    for nuts in nuts_with_sto_index:
        nuts_points = points_nuts_ds[points_nuts_ds == nuts].index.to_list()
        df_sto[nuts] = compute_sto_inflows(runoff_dataset, nuts_points, sto_multipliers_ds[nuts[:2]])

    # Do some 'hand' correcting work
    if topology_unit == 'NUTS2':
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

    # TODO: can we remove that?
    # # IEA validation data in 2016 (TWh produced per year)
    # hydro_production_dict = {'AT': 14., 'BE': 0.5, 'BG': 5.7, 'HR': 6.4, 'CZ': 1.8, 'FI': 16.8, 'FR': 54.5,
    #                          'DE': 19., 'EL': 6.1, 'IE': 0.8, 'IT': 45.5, 'LV': 1.9, 'NO': 137.9, 'PL': 1.8,
    #                          'PT': 8.9, 'RO': 16.6, 'SK': 3.9, 'SI': 3.8, 'CH': 36., 'ES': 28.1, 'SE': 75.3,
    #                          'UK': 6.3, 'AL': 6., 'BA': 5., 'ME': 1.5, 'MK': 1.9, 'RS': 9.7, 'HU': 1., 'LT': 1.}

    nuts_type = 'NUTS0'
    ror_flood_threshold = 0.8
    start = datetime(2014, 1, 1, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)
    ts = pd.date_range(start, end, freq='H')

    generate_eu_hydro_files(nuts_type, ts, ror_flood_threshold)
