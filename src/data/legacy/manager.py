from src.data.land_data.manager import filter_onshore_offshore_points
from os.path import join, dirname, abspath
import pandas as pd
import scipy.spatial

import numpy as np
from typing import *

from src.data.geographics.manager import _get_country


def read_legacy_capacity_data(tech: str, legacy_min_capacity: float, countries: List[str],
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

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility']
    assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

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

    else:

        data = pd.read_excel(join(path_legacy_data, 'Solarfarms_Europe_20200208.xlsx'), sheet_name='ProjReg_rpt',
                             header=0, usecols=[0, 3, 4, 5, 8])
        data = data[pd.notnull(data['Coords'])]
        data['Longitude'] = data['Coords'].str.split(',', 1).str[1]
        data['Latitude'] = data['Coords'].str.split(',', 1).str[0]
        data['Country'] = data['Country'].apply(lambda c: _get_country('alpha_2', name=c))
        data = data[data['Country'].isin(countries)]
        # Converting from MW to GW
        data['MWac'] *= 1e-3

        # Associate each location with legacy capacity to a point in points
        # TODO: make a function of this? use kneighbors?
        legacy_capacity_locs = np.array(list(zip(data['Longitude'], data['Latitude'])))
        points = np.array(points)
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(points, legacy_capacity_locs, 'euclidean'), axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')

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

    accepted_techs = ['wind_onshore', 'wind_offshore', 'pv_utility']
    for tech in technologies:
        assert tech in accepted_techs, "Error: tech {} is not in {}".format(tech, accepted_techs)

    existing_capacity_dict = dict.fromkeys(technologies)
    for tech in technologies:
        # Filter coordinates to obtain only the ones on land or offshore
        onshore = False if tech == 'wind_offshore' else True
        land_filtered_points = filter_onshore_offshore_points(onshore, points, spatial_resolution)
        # Get legacy capacity at points in land_filtered_coordinates where legacy capacity exists
        existing_capacity_dict[tech] = read_legacy_capacity_data(tech, tech_config[tech]['legacy_min_capacity'],
                                                                 regions, land_filtered_points)

    return existing_capacity_dict


def get_legacy_capacity_in_regions():
    pass