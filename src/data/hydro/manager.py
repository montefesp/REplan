import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from os import listdir
from os.path import join, abspath, dirname


from src.data.geographics.manager import match_points_to_regions


# TODO: the two following functions might be already implemented somewhere else (geographics/manager.py)


def retrieve_points_within_region(shapefile_data, start_coordinates, region):

    l = []
    for coord in start_coordinates:
        p = Point(coord)
        if p.within(shapefile_data.loc[region, 'geometry']):
            l.append(coord)

    return l


def compute_ror_series(dataset_runoff, region, shapefile, flood_event_threshold):

    coordinates = list(zip(dataset_runoff.longitude.values, dataset_runoff.latitude.values))

    region_points = retrieve_points_within_region(shapefile, coordinates, region)

    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro.load()
    ts_clip = ts.clip(max=ts.quantile(q=flood_event_threshold).values)
    ts_norm = ts_clip / ts_clip.max()

    return ts_norm


def compute_sto_inflows(dataset_runoff, region, shapefile, sto_multipliers):

    unit_area = 1e10
    coordinates = list(zip(dataset_runoff.longitude.values, dataset_runoff.latitude.values))

    region_points = retrieve_points_within_region(shapefile, coordinates, region)

    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro
    # assume a grid box area of 10'000 km2
    ts_cm = ts * unit_area
    ts_GWh = ts_cm * 9.81 * 1000 * (1/3.6) * 1e-12 * sto_multipliers[region[:2]]

    # prod = ts_GWh.sum(dim='time').values*(1e-3)
    # exp = hydro_production_dict[region[:2]]

    return ts_GWh


# TODO comment and change arguments
def retrieve_hydro_eu_data(topology_unit: str, php_duration_dict, sto_multipliers,
                           flood_event_threshold, timestamps):

    assert topology_unit in ["NUTS0", "NUTS2"], "Error: topology_unit must be NUTS2 or NUTS0"

    # Read region shapes
    # TODO: can we not use geogaphics manager? probably yes
    geographics_dir = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/")
    if topology_unit == 'NUTS2':
        # TODO: need to change that
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
        nuts = points_nuts_ds[lon, lat]
        # Need the if because some points are exactly at the same position
        return nuts if isinstance(nuts, str) else nuts.iloc[0]
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
    for item in php_nuts_capacity_df.index:
        php_nuts_capacity_df.loc[item, 'Energy'] = php_nuts_capacity_df.loc[item, 'Capacity'] * \
                                                   php_duration_dict[item[:2]]

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

    # TODO: david, why do we do that?
    if topology_unit == 'NUTS2':

        regions_ror = ['PT20', 'PT30', 'DE50', 'DE60', 'DE71', 'DE80', 'DEA4', 'DE24',
                       'DE72', 'DE92', 'DEA1', 'DEB3', 'HU22', 'DED2', 'DED5', 'NO02',
                       'NO03', 'PL61', 'PL62', 'PL71', 'UKC1', 'UKF1', 'UKC2', 'AT13']
        capacities_df.loc[regions_ror, 'ROR_CAP [GW]'] = None

        regions_sto = ['BE33', 'BE34', 'IE05', 'AT13']
        capacities_df.loc[regions_sto, 'STO_CAP [GW]'] = None
        capacities_df.loc[regions_sto, 'STO_EN_CAP [GWh]'] = None

    # TODO: this take ages, why?
    # Compute time-series of inflow for STO (in GWh) and ROR (per unit of capacity)
    runoff_dir = join(dirname(abspath(__file__)), "../../../data/land_data/runoff")
    runoff_files = [join(runoff_dir, fn) for fn in listdir(runoff_dir) if fn.endswith(".nc")]
    runoff_dataset = xr.open_mfdataset(runoff_files, combine='by_coords')
    runoff_dataset = runoff_dataset.stack(locations=('longitude', 'latitude'))

    df_ror = pd.DataFrame(index=timestamps, columns=list(capacities_df[capacities_df['ROR_CAP [GW]'].notnull()].index))
    for region in capacities_df[capacities_df['ROR_CAP [GW]'].notnull()].index:
        df_ror[region] = compute_ror_series(runoff_dataset, region, nuts_shapes, flood_event_threshold)

    df_sto = pd.DataFrame(index=timestamps, columns=list(capacities_df[capacities_df['STO_CAP [GW]'].notnull()].index))
    for region in capacities_df[capacities_df['STO_CAP [GW]'].notnull()].index:
        df_sto[region] = compute_sto_inflows(runoff_dataset, region, nuts_shapes, sto_multipliers)

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
    capacities_df.to_csv(capacities_save_fn, sep=';')
    ror_save_fn = f"{save_dir}hydro_ror_time_series_per_{topology_unit}_pu.csv"
    df_ror.to_csv(ror_save_fn, sep=';')
    sto_save_fn = f"{save_dir}hydro_sto_inflow_time_series_per_{topology_unit}_GWh.csv"
    df_sto.to_csv(sto_save_fn, sep=';')  # TODO; why use ; instead of , ?


if __name__ == '__main__':

    # TODO: Store in file?
    # # IEA validation data in 2016 (TWh produced per year)
    # hydro_production_dict = {'AT': 14., 'BE': 0.5, 'BG': 5.7, 'HR': 6.4, 'CZ': 1.8, 'FI': 16.8, 'FR': 54.5,
    #                          'DE': 19., 'EL': 6.1, 'IE': 0.8, 'IT': 45.5, 'LV': 1.9, 'NO': 137.9, 'PL': 1.8,
    #                          'PT': 8.9, 'RO': 16.6, 'SK': 3.9, 'SI': 3.8, 'CH': 36., 'ES': 28.1, 'SE': 75.3,
    #                          'UK': 6.3, 'AL': 6., 'BA': 5., 'ME': 1.5, 'MK': 1.9, 'RS': 9.7, 'HU': 1., 'LT': 1.}

    # Inflow multiplier (proportional to the water head) required to reach IEA production levels throughout the year.
    sto_multipliers = {'AT': 35., 'BE': 1., 'BG': 71., 'HR': 27., 'CZ': 38., 'FI': 1., 'FR': 1., 'DE': 1.,
                       'EL': 30., 'IE': 1., 'IT': 21., 'LV': 1., 'NO': 44., 'PL': 60., 'PT': 30., 'RO': 42.,
                       'SK': 110., 'SI': 36., 'CH': 60., 'ES': 59., 'SE': 35., 'UK': 1., 'AL': 48.,
                       'BA': 1., 'ME': 10., 'MK': 48., 'RS': 120., 'HU': 1., 'LT': 1.}

    # Eurelectric 2011 study. Assumed 12h for the rest. Optimistic from a existing storage perspective.
    # TODO: need to move to a file and make pd.series with it, and read it directly when used
    php_durations = {'BG': 12, 'BE': 7, 'CH': 246, 'CZ': 6, 'ES': 340, 'FR': 41, 'AT': 42, 'PT': 107, 'LT': 49, 'DE': 6,
                     'UK': 11, 'EL': 30, 'PL': 6, 'LU': 6, 'SK': 8, 'IE': 7,
                     'SI': 4, 'NO': 12, 'IT': 12, 'SE': 12, 'HR': 12}

    topology_unit = 'NUTS2'
    ror_flood_threshold = 0.8
    start = datetime(2014, 1, 1, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)
    timestamps = pd.date_range(start, end, freq='H')

    retrieve_hydro_eu_data(topology_unit, php_durations, sto_multipliers, ror_flood_threshold, timestamps)