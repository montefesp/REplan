from os.path import join, dirname, abspath
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.spatial
import xarray as xr
from shapely.geometry import MultiPoint

from src.data.geographics.manager import convert_country_codes, match_points_to_regions, get_onshore_shapes
from src.data.land_data.manager import filter_onshore_offshore_points
from src.data.population_density.manager import load_population_density_data


# TODO: could there be points outside of mainland europe??? -> Yes expl: NL -69.8908, 12.474 in curaçao
# TODO: need to merge the end of the if and else
def read_legacy_capacity_data(tech: str, legacy_min_capacity: float, countries: List[str], spatial_resolution: float,
                              points: List[Tuple[float, float]]) -> pd.Series:
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
    point_capacity_ds : pd.Series
        Existing capacities per each node.
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
        else:  # wind_offshore
            data = data[data['Area'] == 'Offshore']

        # Associate each location with legacy capacity to a point in points
        legacy_capacity_locs = np.array(list(zip(data['Longitude'], data['Latitude'])))
        points = np.array(points)
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(np.array(points), legacy_capacity_locs, 'euclidean'),
                              axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['Total power'].agg('sum')

        points_capacity_ds = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity]

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

        points_capacity_ds = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity]

    else:  # tech == 'pv_residential'

        data = pd.read_excel(join(path_legacy_data, 'SolarEurope_Residential_deployment.xlsx'), header=0, index_col=0)
        data = data['Capacity [GW]']
        data = data[data.index.isin(countries)]

        array_pop_density = load_population_density_data(spatial_resolution)

        codes = [item for item in data.index]
        filter_shape_data = get_onshore_shapes(codes, filterremote=True)
        # , save_file_name=f"{''.join(sorted(codes))}_nuts2_on.geojson")

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

        points_capacity_ds = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity]

    return points_capacity_ds


def get_legacy_capacity(technologies: List[str], tech_config: Dict[str, Any],
                        countries: List[str], points: List[Tuple[float, float]],
                        spatial_resolution: float) -> pd.Series:
    """
    Returns, for each technology and for each point, the existing (legacy) capacity in GW.

    Parameters
    ----------
    technologies: List[str]
        List of technologies for which we want to obtain legacy capacity
    countries: List[str]
        Countries for which we want legacy capacity
    points : List[Tuple[float, float]]
        Points to which existing capacity must be associated
    spatial_resolution: float
        Spatial resolution of the points.

    Returns
    -------
    existing_capacity_dict : pd.Series (MultiIndex on technologies and points)
        Gives for each technology and each point its existing capacity (if its not zero)

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']
    for tech in technologies:
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    existing_capacity_ds = pd.Series(index=pd.MultiIndex.from_product([technologies, points]))
    for tech in technologies:
        # Filter coordinates to obtain only the ones on land or offshore
        onshore = False if tech == 'wind_offshore' else True
        land_filtered_points = filter_onshore_offshore_points(onshore, points, spatial_resolution)
        # Get legacy capacity at points in land_filtered_coordinates where legacy capacity exists
        points_capacity_ds = read_legacy_capacity_data(tech, tech_config[tech]['legacy_min_capacity'],
                                                       countries, spatial_resolution, land_filtered_points)
        existing_capacity_ds.loc[tech, points_capacity_ds.index] = points_capacity_ds.values
    existing_capacity_ds = existing_capacity_ds.dropna()

    return existing_capacity_ds


# TODO: need to add legacy capacity for pv residential
def get_legacy_capacity_in_regions(tech: str, regions: pd.Series, countries: List[str]) -> pd.Series:
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
        else:  # wind_offshore
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

    points_region = match_points_to_regions(data["Location"].values, regions).dropna()
    capacities = pd.Series(index=regions.index)
    for region in regions.index:
        points_in_region = points_region[points_region == region].index.values
        capacities[region] = data[data["Location"].isin(points_in_region)]["Total power"].sum()

    return capacities


if __name__ == '__main__':

    import yaml
    tech_config_path = join(dirname(abspath(__file__)), '../../parameters/pv_wind_tech_configs.yml')
    tech_conf = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)
    print(get_legacy_capacity(["wind_onshore", "pv_utility"],
                              tech_conf, ['FR'], [(2.0, 49.0), (2.5, 50.0), (5.0, 69.0)], 0.5))
