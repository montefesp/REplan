from os.path import join, dirname, abspath
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy.spatial
from shapely.geometry import MultiPoint

from pyggrid.data.geographics import convert_country_codes, match_points_to_regions, get_shapes
from pyggrid.data.land_data import filter_onshore_offshore_points
from pyggrid.data.indicators.population import load_population_density_data
from pyggrid.data.technologies import get_config_values


def associated_legacy_to_points(tech: str, points: List[Tuple[float, float]], spatial_resolution: float,
                                countries: List[str], legacy_min_capacity: float) -> pd.Series:
    """
    Read dataset of existing RES units in the given area and associated to the closest points. Available for EU only.

    Parameters
    ----------
    tech: str
        Technology for which we want existing capacity
    points : List[Tuple[float, float]]
        Points to which existing capacity must be associated
    spatial_resolution: float
        Spatial resolution of the points.
    countries: List[str]
        List of ISO codes of countries for which we want data
    legacy_min_capacity: float
        Points with an aggregate capacity under this capacity will be removed

    Returns
    -------
    point_capacity_ds : pd.Series
        Existing capacities per each node.
    """

    path_legacy_data = join(dirname(abspath(__file__)), '../../../../../data/generation/vres/legacy/source/')

    plant, plant_type = get_config_values(tech, ["plant", "type"])
    if (plant, plant_type) in [("Wind", "Onshore"), ("Wind", "Offshore")]:

        data = pd.read_excel(f"{path_legacy_data}Windfarms_Europe_20200127.xls", sheet_name='Windfarms',
                             header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
        data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
        data = data[data['Status'] != 'Dismantled']
        data = data[data['ISO code'].isin(countries)]
        # Converting from kW to GW
        data['Total power'] *= 1e-6

        # Keep only onshore or offshore point depending on technology
        if plant_type == 'Onshore':
            data = data[data['Area'] != 'Offshore']
        else:  # Offshore
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

    elif (plant, plant_type) == ('PV', 'Utility'):

        data = pd.read_excel(f"{path_legacy_data}Solarfarms_Europe_20200208.xlsx", sheet_name='ProjReg_rpt',
                             header=0, usecols=[0, 3, 4, 5, 8])
        data = data[pd.notnull(data['Coords'])]

        data["Location"] = data["Coords"].apply(lambda x: (float(x.split(',')[1]), float(x.split(',')[0])))
        data['Country'] = convert_country_codes(data['Country'].values, 'name', 'alpha_2')
        data = data[data['Country'].isin(countries)]
        # Converting from MW to GW
        data['MWac'] *= 1e-3

        # Associate each location with legacy capacity to a point in points
        # ODO: make a function of this? use kneighbors?
        points = np.array(points)
        legacy_capacity_locs = np.array(list(data['Location'].values))
        associated_points = \
            [(x[0], x[1]) for x in
             points[np.argmin(scipy.spatial.distance.cdist(points, legacy_capacity_locs, 'euclidean'), axis=0)]]

        data['Node'] = associated_points
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')

        points_capacity_ds = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity]

    elif (plant, plant_type) == ('PV', 'Residential'):

        data = pd.read_excel(f"{path_legacy_data}SolarEurope_Residential_deployment.xlsx", header=0, index_col=0)
        data = data['Capacity [GW]']
        data = data[data.index.isin(countries)]

        pop_density_array = load_population_density_data(spatial_resolution)

        codes = [item for item in data.index]
        onshore_shapes = get_shapes(codes, which='onshore', save=True)["geometry"]

        coords_multipoint = MultiPoint(points)
        df = pd.DataFrame([])

        for country in data.index:

            points_in_country = coords_multipoint.intersection(onshore_shapes.loc[country])
            points_in_country = [(point.x, point.y) for point in points_in_country]

            unit_capacity = data.loc[country] / pop_density_array.sel(locations=points_in_country).values.sum()

            dict_in_country = {key: None for key in points_in_country}

            for point in points_in_country:
                dict_in_country[point] = unit_capacity * pop_density_array.sel(locations=point).values

            df = pd.concat([df, pd.DataFrame.from_dict(dict_in_country, orient='index')])
        aggregate_capacity_per_node = df.T.squeeze()

        points_capacity_ds = aggregate_capacity_per_node[aggregate_capacity_per_node > legacy_min_capacity]

    else:
        raise ValueError(f"Error: No legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")

    return points_capacity_ds


def get_legacy_capacity_at_points(technologies: List[str], countries: List[str],
                                  points: List[Tuple[float, float]],
                                  spatial_resolution: float) -> pd.Series:
    """
    Return, for each technology and for each point, the existing (legacy) capacity in GW.

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

    accepted_plants = [("PV", "Utility"), ("PV", "Residential"), ("Wind", "Onshore"), ("Wind", "Offshore")]
    for tech in technologies:
        plant, plant_type = get_config_values(tech, ["plant", "type"])
        assert (plant, plant_type) in accepted_plants, \
            f"Error: No legacy data exists for tech {tech} with plant {plant} and type {plant_type}."

    existing_capacity_ds = pd.Series(index=pd.MultiIndex.from_product([technologies, points]))
    for tech in technologies:
        onshore, legacy_min_capacity = get_config_values(tech, ["onshore", "legacy_min_capacity"])
        # Filter coordinates to obtain only the ones on land or offshore
        land_filtered_points = filter_onshore_offshore_points(onshore, points, spatial_resolution)
        # Associate existing legacy plants to points
        points_capacity_ds = associated_legacy_to_points(tech, land_filtered_points, spatial_resolution,
                                                         countries, legacy_min_capacity)
        existing_capacity_ds.loc[tech, points_capacity_ds.index] = points_capacity_ds.values
    existing_capacity_ds = existing_capacity_ds.dropna()

    return existing_capacity_ds


def get_legacy_capacity_in_countries(tech: str, countries: List[str], raise_error: bool = True) -> pd.Series:
    """
    Return the total existing capacity (in GW) for the given tech for a set of countries.

    If there is not data for a certain country, returns a capacity of 0.

    Parameters
    ----------
    tech: str
        Name of technology for which we want to retrieve legacy data.
    countries: List[str]
        List of ISO codes of countries
    raise_error: bool (default: True)
        Whether to raise an error if no legacy data is available for this technology.

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each country.

    """

    assert len(countries) != 0, "Error: List of countries is empty."

    path_legacy_data = join(dirname(abspath(__file__)), '../../../../../data/generation/vres/legacy/source/')

    capacities = pd.Series(0., index=countries, name="Legacy capacity (GW)", dtype=float)
    plant, plant_type = get_config_values(tech, ["plant", "type"])
    if (plant, plant_type) in [("Wind", "Onshore"), ("Wind", "Offshore")]:

        data = pd.read_excel(f"{path_legacy_data}Windfarms_Europe_20200127.xls", sheet_name='Windfarms',
                             header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
        data = data.dropna(subset=['Total power'])
        data = data[data['Status'] != 'Dismantled']
        if countries is not None:
            data = data[data['ISO code'].isin(countries)]
        # Converting from kW to GW
        data['Total power'] *= 1e-6

        # Keep only onshore or offshore point depending on technology
        if plant_type == 'Onshore':
            data = data[data['Area'] != 'Offshore']
        else:  # Offshore
            data = data[data['Area'] == 'Offshore']

        # Aggregate capacity per country
        # TODO: shitty because takes also values which are not part of mainland...
        data = data[["ISO code", "Total power"]].groupby('ISO code').sum()
        capacities[data.index] = data["Total power"]

    elif (plant, plant_type) == ("PV", "Utility"):

        data = pd.read_excel(f"{path_legacy_data}Solarfarms_Europe_20200208.xlsx", sheet_name='ProjReg_rpt',
                             header=0, usecols=[0, 4, 8])
        if countries is not None:
            data['Country'] = convert_country_codes(data['Country'].values, 'name', 'alpha_2')
            data = data[data['Country'].isin(countries)]
        # Converting from MW to GW
        data['Total power'] = data['MWac']*1e-3

        # Aggregate capacity per country
        data = data[["Country", "Total power"]].groupby('Country').sum()
        capacities[data.index] = data["Total power"]

    elif (plant, plant_type) == ("PV", "Residential"):

        legacy_capacity_fn = f"{path_legacy_data}SolarEurope_Residential_deployment.xlsx"
        data = pd.read_excel(legacy_capacity_fn, header=0, index_col=0, usecols=[0, 4], squeeze=True).sort_index()
        data = data[data.index.isin(countries)]
        capacities[data.index] = data

    else:
        if raise_error:
            raise ValueError(f"Error: no legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")
        else:
            warnings.warn(f"Warning: No legacy data exists for tech {tech}.")
    return capacities


def get_legacy_capacity_in_regions(tech: str, regions_shapes: pd.Series, countries: List[str],
                                   match_distance: float = 50., raise_error: bool = True) -> pd.Series:
    """
    Return the total existing capacity (in GW) for the given tech for a set of geographical regions.

    Parameters
    ----------
    tech: str
        Technology name.
    regions_shapes: pd.Series [Union[Polygon, MultiPolygon]]
        Geographical regions
    countries: List[str]
        List of ISO codes of countries in which the regions are situated
    match_distance: float (default: 50)
        Distance threshold (in km) used when associating points to shape.
    raise_error: bool (default: True)
        Whether to raise an error if no legacy data is available for this technology.

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each region

    """

    path_legacy_data = join(dirname(abspath(__file__)), '../../../../../data/generation/vres/legacy/source/')

    capacities = pd.Series(0., index=regions_shapes.index)
    plant, plant_type = get_config_values(tech, ["plant", "type"])
    if (plant, plant_type) in [("Wind", "Onshore"), ("Wind", "Offshore"), ("PV", "Utility")]:

        if plant == "Wind":

            data = pd.read_excel(f"{path_legacy_data}Windfarms_Europe_20200127.xls", sheet_name='Windfarms',
                                 header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
            data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
            data = data[data['Status'] != 'Dismantled']
            if countries is not None:
                data = data[data['ISO code'].isin(countries)]

            # Converting from kW to GW
            data['Total power'] *= 1e-6
            data["Location"] = data[["Longitude", "Latitude"]].apply(lambda x: (x.Longitude, x.Latitude), axis=1)

            # Keep only onshore or offshore point depending on technology
            if plant_type == 'Onshore':
                data = data[data['Area'] != 'Offshore']
            else:  # Offshore
                data = data[data['Area'] == 'Offshore']

            if len(data) == 0:
                return capacities

        else:  # plant == "PV":

            data = pd.read_excel(f"{path_legacy_data}Solarfarms_Europe_20200208.xlsx", sheet_name='ProjReg_rpt',
                                 header=0, usecols=[0, 4, 8])
            data = data[pd.notnull(data['Coords'])]
            data["Location"] = data["Coords"].apply(lambda x: (float(x.split(',')[1]), float(x.split(',')[0])))
            if countries is not None:
                data['Country'] = convert_country_codes(data['Country'].values, 'name', 'alpha_2')
                data = data[data['Country'].isin(countries)]

            if len(data) == 0:
                return capacities

            # Converting from MW to GW
            data['Total power'] = data['MWac']*1e-3

        data = data[["Location", "Total power"]]

        points_region = match_points_to_regions(data["Location"].values, regions_shapes,
                                                distance_threshold=match_distance).dropna()

        for region in regions_shapes.index:
            points_in_region = points_region[points_region == region].index.values
            capacities[region] = data[data["Location"].isin(points_in_region)]["Total power"].sum()

    elif (plant, plant_type) == ("PV", "Residential"):

        legacy_capacity_fn = join(path_legacy_data, 'SolarEurope_Residential_deployment.xlsx')
        data = pd.read_excel(legacy_capacity_fn, header=0, index_col=0, usecols=[0, 4], squeeze=True).sort_index()
        data = data[data.index.isin(countries)]

        if len(data) == 0:
            return capacities

        # Get countries shapes
        countries_shapes = get_shapes(data.index.values, which='onshore', save=True)["geometry"]

        for region_id, region_shape in regions_shapes.items():
            for country_id, country_shape in countries_shapes.items():
                capacities[region_id] += \
                    (region_shape.intersection(country_shape).area/country_shape.area) * data[country_id]

    else:
        if raise_error:
            raise ValueError(f"Error: No legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")
        else:
            warnings.warn(f"Warning: No legacy data exists for tech {tech}.")

    return capacities
