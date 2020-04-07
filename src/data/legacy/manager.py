from os.path import join, dirname, abspath
from typing import List, Tuple, Dict, Union, Any

import numpy as np
import pandas as pd
import scipy.spatial
import xarray as xr
from shapely.geometry import MultiPoint

from src.data.geographics.manager import convert_country_codes, match_points_to_region, get_onshore_shapes
from src.data.land_data.manager import filter_onshore_offshore_points


# TODO: could there be points outside of mainland europe??? -> Yes expl: NL -69.8908, 12.474 in curaçao
# TODO: need to merge the end of the if and else
def read_legacy_capacity_data(tech: str, legacy_min_capacity: float, countries: List[str], spatial_resolution: float,
                              points: List[Tuple[float, float]]) -> Dict[List[Tuple[float, float]], float]:
    """
    Reads dataset of existing RES units in the given area and associated to the closest points. Available for EU only.

    Parameters
    ----------
    tech: str
        Technology for which we want existing capacity
    legacy_min_capacity: float
        Points with an aggregate capacity under this capacity will be removed
    countries: List[str]
        List of ISO codes of countries for which we want data
    points : List[Tuple[float, float]]
        Points to which existing capacity must be associated

    Returns
    -------
    point_capacity_dict : Dict[List[Tuple[float, float]], float]
        Dictionary storing existing capacities per node.
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    path_legacy_data = join(dirname(abspath(__file__)), '../../../data/legacy')

    if tech in ["wind_onshore", "wind_offshore"]:

        data = pd.read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127.xls'), sheet_name='Windfarms',
                             header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
        data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
        data = data[data['Status'] != 'Dismantled']
        data = data[data['ISO code'].isin(countries)]
        # Converting from kW to GW
        data['Total power'] *= 1e-6

        # Keep only onshore or offshore point depending on technology
        if tech == 'wind_onshore':
            data = data[data['Area'] != 'Offshore']
        else:  # wind_offhsore
            data = data[data['Area'] == 'Offshore']

        # Associate each location with legacy capacity to a point in points
        legacy_capacity_locs = np.array(list(zip(data['Longitude'], data['Latitude'])))
        points = np.array(points)
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(np.array(points), legacy_capacity_locs, 'euclidean'), axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['Total power'].agg('sum')

        point_capacity_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity].to_dict()

    elif tech == 'pv_utility':

        data = pd.read_excel(join(path_legacy_data, 'Solarfarms_Europe_20200208.xlsx'), sheet_name='ProjReg_rpt',
                             header=0, usecols=[0, 3, 4, 5, 8])
        data = data[pd.notnull(data['Coords'])]

        data["Location"] = data["Coords"].apply(lambda x: (float(x.split(',')[1]), float(x.split(',')[0])))
        data['Country'] = data['Country'].apply(lambda c: convert_country_codes('alpha_2', name=c))
        data = data[data['Country'].isin(countries)]
        # Converting from MW to GW
        data['MWac'] *= 1e-3

        # Associate each location with legacy capacity to a point in points
        # TODO: make a function of this? use kneighbors?
        points = np.array(points)
        legacy_capacity_locs = np.array(list(data['Location'].values))
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(points, legacy_capacity_locs, 'euclidean'), axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')

        point_capacity_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity].to_dict()

    else: # tech == 'pv_residential'

        data = pd.read_excel(join(path_legacy_data, 'SolarEurope_Residential_deployment.xlsx'), header=0, index_col=0)
        data = data['Capacity [GW]']
        data = data[data.index.isin(countries)]

        # Load population density dataset
        path_pop_data = join(dirname(abspath(__file__)), '../../../data/population_density')
        dataset_population = \
            xr.open_dataset(join(path_pop_data, 'gpw_v4_population_density_rev11_' + str(spatial_resolution) + '.nc'))
        # Rename the only variable to 'data' # TODO: is there not a cleaner way to do this?
        varname = [item for item in dataset_population.data_vars][0]
        dataset_population = dataset_population.rename({varname: 'data'})
        # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
        data_pop = dataset_population.sel(raster=5)

        # Compute population density at intermediate points
        array_pop_density = data_pop['data'].interp(longitude=np.arange(-180, 180, float(spatial_resolution)),
                                                    latitude=np.arange(-89, 91, float(spatial_resolution))[::-1],
                                                    method='linear').fillna(0.)
        array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude')).sel(locations=points)

        codes = [item for item in data.index]
        filter_shape_data = get_onshore_shapes(codes, filterremote=True,
                                               save_file_name=''.join(sorted(codes)) + "_nuts2_on.geojson")

        coords_multipoint = MultiPoint(points)
        df = pd.DataFrame([])

        for country in data.index:

            points_in_country = coords_multipoint.intersection(filter_shape_data.loc[country, 'geometry'])
            points_in_country = [(point.x, point.y) for point in points_in_country]

            unit_capacity = data.loc[country] / array_pop_density.sel(locations=points_in_country).values.sum()

            dict_in_country = {key: None for key in points_in_country}

            for point in points_in_country:
                dict_in_country[point] = unit_capacity * array_pop_density.sel(locations=point).values

            df = pd.concat([df, pd.DataFrame.from_dict(dict_in_country, orient='index')])
        aggregate_capacity_per_node = df.T.squeeze()

        point_capacity_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity].to_dict()

    return point_capacity_dict


def get_legacy_capacity(technologies: List[str], tech_config: Dict[str, Any],
                        regions: List[str], points: List[Tuple[float, float]],
                        spatial_resolution: float) -> Dict[str, Dict[Tuple[float, float], float]]:
    """
    Returns, for each technology and for each point, the existing (legacy) capacity in GW.

    Parameters
    ----------
    technologies: List[str]
        List of technologies for which we want to obtain legacy capacity
    regions: List[str]
        Regions for which we want legacy capacity
    points : List[Tuple[float, float]]
        Points to which existing capacity must be associated
    spatial_resolution: float
        Spatial resolution of the points.

    Returns
    -------
    existing_capacity_dict : Dict[str, Dict[Tuple[float, float], float]]
        Dictionary giving for each technology a dictionary associating each location
        with its legacy capacity for that technology.

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']
    for tech in technologies:
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    existing_capacity_dict = dict.fromkeys(technologies)
    for tech in technologies:
        # Filter coordinates to obtain only the ones on land or offshore
        onshore = False if tech == 'wind_offshore' else True
        land_filtered_points = filter_onshore_offshore_points(onshore, points, spatial_resolution)
        # Get legacy capacity at points in land_filtered_coordinates where legacy capacity exists
        existing_capacity_dict[tech] = read_legacy_capacity_data(tech, tech_config[tech]['legacy_min_capacity'],
                                                                 regions, spatial_resolution, land_filtered_points)

    return existing_capacity_dict


def get_legacy_capacity_in_regions(tech: str, regions: pd.Series, countries: List[str]):
    """
    Returns the total existing capacity (in GW) for the given tech for a set of regions

    Parameters
    ----------
    tech: str
        One of 'wind_onshore', 'wind_offshore' or 'pv_utility'
    regions: pd.Series [Union[Polygon, MultiPolygon]]
        Geographical regions
    countries: List[str]
        List of ISO codes of countries for which we want data

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each region

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    path_legacy_data = join(dirname(abspath(__file__)), '../../../data/legacy')

    if tech in ["wind_onshore", "wind_offshore"]:

        data = pd.read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127.xls'), sheet_name='Windfarms',
                             header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
        data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
        data = data[data['Status'] != 'Dismantled']
        if countries is not None:
            data = data[data['ISO code'].isin(countries)]
        # Converting from kW to GW
        data['Total power'] *= 1e-6
        data["Location"] = data[["Longitude", "Latitude"]].apply(lambda x: (x.Longitude, x.Latitude), axis=1)

        # Keep only onshore or offshore point depending on technology
        if tech == 'wind_onshore':
            data = data[data['Area'] != 'Offshore']
        else:  # wind_offhsore
            data = data[data['Area'] == 'Offshore']

    else:  # pv_utility

        data = pd.read_excel(join(path_legacy_data, 'Solarfarms_Europe_20200208.xlsx'), sheet_name='ProjReg_rpt',
                             header=0, usecols=[0, 4, 8])
        data = data[pd.notnull(data['Coords'])]
        data["Location"] = data["Coords"].apply(lambda x: (float(x.split(',')[1]), float(x.split(',')[0])))
        if countries is not None:
            data['Country'] = data['Country'].apply(lambda c: convert_country_codes('alpha_2', name=c))
            data = data[data['Country'].isin(countries)]
        # Converting from MW to GW
        data['Total power'] = data['MWac']*1e-3

    data = data[["Location", "Total power"]]

    points_region = match_points_to_region(data["Location"].values, regions).dropna()
    capacities = pd.Series(index=regions.index)
    for region in regions.index:
        points_in_region = points_region[points_region == region].index.values
        capacities[region] = data[data["Location"].isin(points_in_region)]["Total power"].sum()

    return capacities