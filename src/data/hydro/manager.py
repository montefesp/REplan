import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from os import listdir
from os.path import join, abspath, dirname

def file_list(path):
    """

    :param path:
    :type path:
    :return:
    :rtype:
    """

    list_file = []
    for file in listdir(path):
        if file.endswith(".nc"):
            list_file.append(join(path, file))

    return list_file

# TODO: the two following functions might be already implemented somewhere else (geographics/manager.py)
def assign_region_to_point(shapefile, lon, lat):
    """

    :param shapefile:
    :type shapefile:
    :param lon:
    :type lon:
    :param lat:
    :type lat:
    :return:
    :rtype:
    """

    p = Point(lon, lat)
    region_inc = None

    for region in shapefile.index:

        if shapefile.loc[region]['geometry'].contains(p):

            region_inc = shapefile.loc[region].name

    return region_inc


def RetrievePointsWithinRegion(shapefile_data, start_coordinates, region):
    """

    :param shapefile_data:
    :type shapefile_data:
    :param start_coordinates:
    :type start_coordinates:
    :param region:
    :type region:
    :return:
    :rtype:
    """

    l = []
    for coord in start_coordinates:
        p = Point(coord)
        if p.within(shapefile_data.loc[region, 'geometry']):
            l.append(coord)

    return l


def ComputeRORSeries(dataset_runoff, region, shapefile, flood_event_threshold):

    coordinates = list(zip(dataset_runoff.longitude.values, dataset_runoff.latitude.values))

    region_points = RetrievePointsWithinRegion(shapefile, coordinates, region)

    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro.load()
    ts_clip = ts.clip(max=ts.quantile(q=flood_event_threshold).values)
    ts_norm = ts_clip / ts_clip.max()

    return ts_norm



def ComputeSTOInflows(dataset_runoff, region, shapefile, sto_multipliers):
    """

    :param dataset_runoff:
    :type dataset_runoff:
    :param region:
    :type region:
    :param shapefile:
    :type shapefile:
    :param sto_multipliers:
    :type sto_multipliers:
    :return:
    :rtype:
    """

    unit_area = 1e10
    coordinates = list(zip(dataset_runoff.longitude.values, dataset_runoff.latitude.values))

    region_points = RetrievePointsWithinRegion(shapefile, coordinates, region)

    ts = dataset_runoff.sel(locations=region_points).sum(dim='locations').ro
    # assume a grid box area of 10'000 km2
    ts_cm = ts * unit_area
    ts_GWh = ts_cm * 9.81 * 1000 * (1/3.6) * (1e-12) * sto_multipliers[region[:2]]

    # prod = ts_GWh.sum(dim='time').values*(1e-3)
    # exp = hydro_production_dict[region[:2]]

    return ts_GWh


def retrieve_hydro_EU_data(topology_unit, php_duration_dict, sto_multipliers, flood_event_threshold, start_date, end_date):
    """

    :param topology_unit:
    :type topology_unit:
    :param php_duration_dict:
    :type php_duration_dict:
    :param sto_multipliers:
    :type sto_multipliers:
    :param flood_event_threshold:
    :type flood_event_threshold:
    :param start_date:
    :type start_date:
    :param end_date:
    :type end_date:
    :return:
    :rtype:
    """

    if topology_unit == 'NUTS2':
        nuts_shapefile = \
            gpd.read_file(join(dirname(abspath(__file__)),
                       '../../../data/shapefiles/NUTS_RG_01M_2016_4326_LEVL_2_incl_BA.geojson')).set_index('NUTS_ID')
    elif topology_unit == 'NUTS0':
        nuts_shapefile = \
            gpd.read_file(join(dirname(abspath(__file__)),
                       '../../../data/shapefiles/NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson')).set_index('NUTS_ID')

    hydro_data = pd.read_csv(join(dirname(abspath(__file__)),
                       '../../../data/hydro/source/pp_fresna_hydro_updated.csv'),
                                  sep=';', index_col=0, usecols=[1, 3, 5, 6, 8, 13, 14])
    storage_data = pd.read_csv(join(dirname(abspath(__file__)),
                       '../../../data/hydro/source/hydro_storage_capacities_updated.csv'), sep=';').set_index('Country')

    hydro_data['NUTS'] = hydro_data.apply(lambda x: assign_region_to_point(nuts_shapefile, x['lon'], x['lat']), axis=1)
    hydro_data = hydro_data[~hydro_data['NUTS'].isnull()]

    # ROR plants

    hydro_ror = hydro_data[hydro_data['Technology'] == 'Run-Of-River']
    hydro_ror_agg = (hydro_ror.groupby(hydro_ror['NUTS'])['Capacity'].sum()*(1e-3))

    # STO plants

    hydro_sto = hydro_data[hydro_data['Technology'] == 'Reservoir']
    hydro_sto_agg = (hydro_sto.groupby(hydro_sto['NUTS'])['Capacity'].sum()*(1e-3))

    storage_data = storage_data['E_store[TWh]']

    hydro_sto_storage_agg = hydro_sto_agg.copy()
    for item in hydro_sto_agg.index:
        country_storage_potential = storage_data[item[:2]]
        country_capacity_potential = hydro_sto_agg[hydro_sto_agg.index.str.startswith(item[:2])].sum()
        hydro_sto_storage_agg.loc[item] = (hydro_sto_agg[item] / country_capacity_potential) * country_storage_potential

    # PHS plants

    hydro_php = hydro_data[hydro_data['Technology'] == 'Pumped Storage']
    hydro_php_agg = (hydro_php.groupby(hydro_php['NUTS'])['Capacity'].sum()*(1e-3))

    hydro_php_agg = hydro_php_agg.to_frame()
    for item in hydro_php_agg.index:
        hydro_php_agg.loc[item, 'Energy'] = hydro_php_agg.loc[item, 'Capacity'] * php_duration_dict[item[:2]]





    df_capacities = pd.DataFrame(index=sorted(nuts_shapefile.index),
                                 columns=['ROR_CAP [GW]', 'STO_CAP [GW]', 'STO_EN_CAP [GWh]',
                                          'PSP_CAP [GW]', 'PSP_EN_CAP [GWh]'])
    df_capacities['ROR_CAP [GW]'] = hydro_ror_agg
    df_capacities['STO_CAP [GW]'] = hydro_sto_agg
    df_capacities['STO_EN_CAP [GWh]'] = hydro_sto_storage_agg*1e3
    df_capacities['PSP_CAP [GW]'] = hydro_php_agg['Capacity']
    df_capacities['PSP_EN_CAP [GWh]'] = hydro_php_agg['Energy']

    if topology_unit == 'NUTS2':

        regions_ROR = ['PT20', 'PT30', 'DE50', 'DE60', 'DE71', 'DE80', 'DEA4', 'DE24',
                       'DE72', 'DE92', 'DEA1', 'DEB3', 'HU22', 'DED2', 'DED5', 'NO02',
                       'NO03', 'PL61', 'PL62', 'PL71', 'UKC1', 'UKF1', 'UKC2', 'AT13']
        regions_STO = ['BE33', 'BE34', 'IE05', 'AT13']

        for region in regions_ROR:
            df_capacities['ROR_CAP [GW]'].loc[region] = None
        for region in regions_STO:
            df_capacities['STO_CAP [GW]'].loc[region] = None
            df_capacities['STO_EN_CAP [GWh]'].loc[region] = None

    df_capacities = df_capacities.round(3)
    df_capacities.to_csv(join(dirname(abspath(__file__)),
                       '../../../data/hydro/generated/hydro_capacities_per_'+str(topology_unit)+'.csv'), sep=';')

    date_range = pd.date_range(start_date, end_date, freq='H')

    path_runoff = join(dirname(abspath(__file__)), '../../../data/land_data/runoff')
    list_file_runoff = file_list(path_runoff)
    dataset_runoff = xr.open_mfdataset(list_file_runoff, combine='by_coords')
    dataset_runoff = dataset_runoff.stack(locations=('longitude', 'latitude'))

    df_ROR = pd.DataFrame(index=date_range, columns=list(df_capacities[df_capacities['ROR_CAP [GW]'].notnull()].index))
    for region in df_capacities[df_capacities['ROR_CAP [GW]'].notnull()].index:
        df_ROR[region] = ComputeRORSeries(dataset_runoff, region, nuts_shapefile, flood_event_threshold)

    df_STO = pd.DataFrame(index=date_range, columns=list(df_capacities[df_capacities['STO_CAP [GW]'].notnull()].index))
    for region in df_capacities[df_capacities['STO_CAP [GW]'].notnull()].index:
        df_STO[region] = ComputeSTOInflows(dataset_runoff, region, nuts_shapefile, sto_multipliers)

    if topology_unit == 'NUTS2':
        # For some reason, there is no point associated with ITC2. Fill with neighboring area.
        df_ROR['ITC2'] = df_ROR['ITC3']

        # Same situation here, some wild hacks for the moment to fill them with data from regions with similar capacities.
        df_STO['ES52'] = df_STO['ES24']
        df_STO['EL42'] = df_STO['EL61']
        df_STO['ITC2'] = df_STO['ITC4'] / 2.

    df_ROR.to_csv(join(dirname(abspath(__file__)),
                       '../../../data/hydro/generated/hydro_ror_time_series_per_'+str(topology_unit)+'_pu.csv'),
                       sep=';')
    df_STO.to_csv(join(dirname(abspath(__file__)),
                       '../../../data/hydro/generated/hydro_sto_inflow_time_series_per_'+str(topology_unit)+'_GWh.csv'),
                       sep=';')

    return None


if __name__ == '__main__':

    # # IEA validation data in 2016 (TWh produced per year)
    # hydro_production_dict = {'AT': 14., 'BE': 0.5, 'BG': 5.7, 'HR': 6.4, 'CZ': 1.8, 'FI': 16.8, 'FR': 54.5, 'DE': 19.,
    #                          'EL': 6.1, 'IE': 0.8, 'IT': 45.5, 'LV': 1.9, 'NO': 137.9, 'PL': 1.8, 'PT': 8.9, 'RO': 16.6,
    #                          'SK': 3.9, 'SI': 3.8, 'CH': 36., 'ES': 28.1, 'SE': 75.3, 'UK': 6.3, 'AL': 6., 'BA': 5.,
    #                          'ME': 1.5, 'MK': 1.9, 'RS': 9.7, 'HU':1., 'LT':1.}

    # Inflow multiplier (proportional to the water head) required to reach IEA production levels throughout the year.
    sto_multipliers = {'AT': 35., 'BE': 1., 'BG': 71., 'HR': 27., 'CZ': 38., 'FI': 1., 'FR': 1., 'DE': 1.,
                              'EL': 30., 'IE': 1., 'IT': 21., 'LV': 1., 'NO': 44., 'PL': 60., 'PT': 30., 'RO': 42.,
                              'SK': 110., 'SI': 36., 'CH': 60., 'ES': 59., 'SE': 35., 'UK': 1., 'AL': 48.,
                              'BA': 1., 'ME': 10., 'MK': 48., 'RS': 120., 'HU':1., 'LT': 1.}

    # Eurelectric 2011 study. Assumed 12h for the rest. Optimistic from a existing storage perspective.
    php_durations = {'ES': 340, 'CH': 246, 'FR': 41, 'AT': 42, 'PT': 107, 'LT': 49, 'DE': 6,
                    'UK': 11, 'EL': 30, 'PL': 6, 'BE': 7, 'CZ': 6, 'LU': 6, 'SK': 8, 'IE': 7,
                    'SI': 4, 'BG': 12, 'NO': 12, 'IT': 12, 'SE': 12, 'HR': 12}


    topology_unit = 'NUTS2'
    ror_flood_threshold = 0.8
    start = datetime(2014, 1, 1, 0, 0, 0)
    end = datetime(2018, 12, 31, 23, 0, 0)

    retrieve_hydro_EU_data(topology_unit, php_durations, sto_multipliers, ror_flood_threshold, start, end)