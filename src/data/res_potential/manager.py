from os.path import join, dirname, abspath
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd

from shapely.ops import cascaded_union
from shapely.geometry import Polygon, MultiPolygon
from shapely.errors import TopologicalError

from src.data.geographics.manager import get_onshore_shapes, get_offshore_shapes, match_points_to_regions
from src.data.population_density.manager import load_population_density_data

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def get_available_regions(region_type: str) -> List[str]:
    """Returns the list of codes of regions for which capacity potential is available"""

    accepted_types = ["nuts2", "nuts0", "eez"]
    assert region_type in accepted_types, f"Error: region_type {region_type} is not in {accepted_types}"

    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/generated/')

    # Onshore, return NUTS (0 or 2) capacity potentials
    if region_type in ["nuts2", "nuts0"]:
        return list(pd.read_csv(f"{path_potential_data}{region_type}_capacity_potentials_GW.csv", index_col=0).index)
    # Offshore, return EEZ capacity potentials
    else:
        return list(pd.read_csv(f"{path_potential_data}eez_capacity_potentials_GW.csv", index_col=0))


def read_capacity_potential(tech: str, nuts_type: str) -> pd.Series:
    """
    Returns for each NUTS2 (or NUTS0) region or EEZ (depending on technology) its capacity potential in GW

    Parameters
    ----------
    tech: str
        Technology name among 'wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility' and 'pv_residential'
    nuts_type: str
        If equal to 'nuts0', returns capacity per NUTS0 region, if 'nuts2', returns per NUTS2 region.

    Returns
    -------
    pd.Series:
        Gives for each NUTS2 region or EEZ its capacity potential in GW

    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"
    accepted_nuts = ["nuts2", "nuts0"]
    assert nuts_type in accepted_nuts, f"Error: nuts_type {nuts_type} is not in {accepted_nuts}"

    path_potential_data = join(dirname(abspath(__file__)), '../../../data/res_potential/generated/')

    # Onshore, return NUTS (0 or 2) capacity potentials
    if tech in ['wind_onshore', 'pv_utility', 'pv_residential']:
        return pd.read_csv(f"{path_potential_data}{nuts_type}_capacity_potentials_GW.csv", index_col=0)[tech]
    # Offshore, return EEZ capacity potentials
    else:
        return pd.read_csv(f"{path_potential_data}eez_capacity_potentials_GW.csv", index_col=0)[tech]


def get_capacity_potential_at_points(tech_points_dict: Dict[str, List[Tuple[float, float]]],
                                     spatial_resolution: float, countries: List[str],
                                     existing_capacity_ds: pd.Series = None) -> pd.Series:
    """
    Computes the potential capacity at a series of points for different technologies.

    Parameters
    ----------
    tech_points_dict : Dict[str, Dict[str, List[Tuple[float, float]]]
        Dictionary associating to each tech a list of points.
    spatial_resolution : float
        Spatial resolution of the points.
    countries: List[str]
        List of ISO codes of countries in which the points are situated
    existing_capacity_ds: pd.Series (default: None)
        Data series given for each tuple of (tech, point) the existing capacity.

    Returns
    -------
    capacity_potential_ds : pd.Series
        Gives for each pair of technology - point the associated capacity potential in GW
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in tech_points_dict.keys():
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    assert spatial_resolution in [0.5, 1.0], \
        f"Error: Accepted resolution are 0.5 or 1.0, received {spatial_resolution}"

    # Create a modified copy of regions to deal with UK and EL
    iso_to_nuts0 = {"GB": "UK", "GR": "EL"}
    nuts0_regions = [iso_to_nuts0[c] if c in iso_to_nuts0 else c for c in countries]

    pop_density_array = load_population_density_data(spatial_resolution)

    # Get EEZ shapes
    region_shapes_dict = {"nuts2": None, "eez": None}
    onshore_shapes = get_onshore_shapes(countries, filterremote=True,
                                        save_file_name=f"{''.join(sorted(countries))}_regions_on.geojson")
    onshore_shapes_union = cascaded_union(onshore_shapes["geometry"].values)
    region_shapes_dict["eez"] = \
        get_offshore_shapes(countries, onshore_shape=onshore_shapes_union, filterremote=True,
                            save_file_name=f"{''.join(sorted(countries))}_regions_off.geojson")
    region_shapes_dict["eez"].index = [f"EZ{code}" for code in region_shapes_dict["eez"].index]

    # Get NUTS2 shapes
    nuts2_regions_list = get_available_regions("nuts2")
    codes = [code for code in nuts2_regions_list if code[:2] in nuts0_regions]
    region_shapes_dict["nuts2"] = get_onshore_shapes(codes, filterremote=True,
                                                     save_file_name=f"{''.join(sorted(countries))}_nuts2_on.geojson")

    tech_points_tuples = sorted([(tech, point) for tech, points in tech_points_dict.items() for point in points])
    capacity_potential_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_points_tuples))

    # Check that if existing capacity is defined for every point
    if existing_capacity_ds is not None:
        missing_existing_points = set(existing_capacity_ds.index) - set(capacity_potential_ds.index)
        assert not missing_existing_points, \
            f"Error: Missing following points in existing capacity series: {missing_existing_points}"

    for tech, points in tech_points_dict.items():

        # Check that points are at the right resolution
        assert all(map(lambda point: int(point[0]/spatial_resolution) == point[0]/spatial_resolution
                   and int(point[1]/spatial_resolution) == point[1]/spatial_resolution, points)), \
                f"Error: Some points do not have the correct resolution {spatial_resolution}"

        # Compute potential for each NUTS2 or EEZ
        potential_per_region_ds = read_capacity_potential(tech, nuts_type='nuts2')

        # TODO: delete when we are sure everything works
        """
        # Get NUTS2 and EEZ shapes
        # TODO: this is shit -> not generic enough, e.g.: would probably not work for us states
        #  would need to get this out of the loop
        if tech in ['wind_offshore', 'wind_floating']:
            onshore_shapes = get_onshore_shapes(countries, filterremote=True,
                                                save_file_name=f"{''.join(sorted(countries))}_regions_on.geojson")
            onshore_shapes_union = cascaded_union(onshore_shapes["geometry"].values)
            filter_shape_data = get_offshore_shapes(countries, onshore_shape=onshore_shapes_union,
                                                    filterremote=True,
                                                    save_file_name=f"{''.join(sorted(countries))}_regions_off.geojson")
            filter_shape_data.index = [f"EZ{code}" for code in filter_shape_data.index]
        else:
            codes = [code for code in potential_per_region_ds.index if code[:2] in nuts0_regions]
            filter_shape_data = get_onshore_shapes(codes, filterremote=True,
                                                   save_file_name=f"{''.join(sorted(countries))}_nuts2_on.geojson")
        """

        # Find the geographical region code associated to each point
        if tech in ['wind_offshore', 'wind_floating']:
            region_shapes = region_shapes_dict["eez"]
        else:
            region_shapes = region_shapes_dict["nuts2"]
        point_regions_ds = match_points_to_regions(points, region_shapes["geometry"]).dropna()
        points = list(point_regions_ds.index)
        points_info_df = pd.DataFrame(point_regions_ds.values, point_regions_ds.index, columns=["region"])

        if tech in ['wind_offshore', 'wind_floating']:

            # For offshore sites, divide the total potential of the region by the number of points
            # associated to that region

            # Get how many points we have in each region and the potential capacity of those regions
            region_freq_ds = points_info_df.groupby(['region'])['region'].count()
            regions = region_freq_ds.index
            region_cap_pot_ds = potential_per_region_ds[regions]
            region_info_df = pd.concat([region_freq_ds, region_cap_pot_ds], axis=1)
            region_info_df.columns = ["freq", "cap_pot"]

            # Assign these values to each points depending on which region they fall in
            points_info_df = \
                points_info_df.merge(region_info_df, left_on='region', right_on='region', right_index=True)

            # Compute potential of each point by dividing the region potential by the number of points it contains
            cap_pot_per_point = points_info_df["cap_pot"]/points_info_df["freq"]

        else:  # tech in ['wind_onshore', 'pv_utility', 'pv_residential']:

            # For onshore sites, divide the total anti-proportionally (or proportionally for residential PV)
            # to population
            # Here were actually using population density, which is proportional to population because we consider
            # that each point is associated to an equivalent area.
            points_info_df['pop_dens'] = np.clip(pop_density_array.sel(locations=points).values, a_min=1., a_max=None)
            if tech in ['wind_onshore', 'pv_utility']:
                points_info_df['pop_dens'] = 1./points_info_df['pop_dens']

            # Aggregate per region and get capacity potential for regions in which the points fall
            regions_info_df = points_info_df.groupby(['region']).sum()
            regions_info_df["cap_pot"] = potential_per_region_ds[regions_info_df.index]
            regions_info_df.columns = ['sum_pop_dens', 'cap_pot']

            # Assign these values to each points depending on which region they fall in
            points_info_df = points_info_df.merge(regions_info_df, left_on='region', right_on='region',
                                                  right_index=True)
            # Compute potential
            cap_pot_per_point = points_info_df['pop_dens'] * points_info_df['cap_pot'] / points_info_df['sum_pop_dens']

        capacity_potential_ds.loc[tech, cap_pot_per_point.index] = cap_pot_per_point.values

    # Update capacity potential with existing potential if present
    if existing_capacity_ds is not None:
        underestimated_capacity = existing_capacity_ds[capacity_potential_ds.index] > capacity_potential_ds
        capacity_potential_ds[underestimated_capacity] = existing_capacity_ds[underestimated_capacity]

    return capacity_potential_ds


def get_capacity_potential_for_regions(tech_regions_dict: Dict[str, List[Union[Polygon, MultiPolygon]]]) -> pd.Series:
    """
    Get capacity potential (in GW) for a series of technology for associated geographical regions

    Parameters
    ----------
    tech_regions_dict: Dict[str, List[Union[Polygon, MultiPolygon]]]
        Dictionary giving for each technology for which region we want to obtain potential capacity

    Returns
    -------
    capacity_potential_ds: pd.Series
        Gives for each pair of technology and region the associated potential capacity in GW

    """
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in tech_regions_dict.keys():
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    tech_regions_tuples = [(tech, i) for tech, points in tech_regions_dict.items() for i in range(len(points))]
    capacity_potential_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_regions_tuples))

    for tech, regions in tech_regions_dict.items():

        # Compute potential for each NUTS2 or EEZ
        potential_per_subregion_ds = read_capacity_potential(tech, nuts_type='nuts2')

        # Get NUTS2 or EEZ shapes
        #  TODO: would need to get this out of the loop
        if tech in ['wind_offshore', 'wind_floating']:
            codes = [code[2:4] for code in potential_per_subregion_ds.index.values]
            onshore_shapes_union = cascaded_union(get_onshore_shapes(codes, filterremote=True)["geometry"].values)
                                                  #, save_file_name="cap_potential_regions_on.geojson"
            shapes = get_offshore_shapes(codes, onshore_shape=onshore_shapes_union,
                                         filterremote=True)#, save_file_name="cap_potential_regions_off.geojson")
            shapes.index = [f"EZ{code}" for code in shapes.index]
        else:
            shapes = get_onshore_shapes(potential_per_subregion_ds.index.values, filterremote=True)
                                        # save_file_name="cap_potential_regions_on.geojson")
            # TODO: problem BA00 does not exists in shapes -> rename by BA?

        # Compute capacity potential for the regions given as argument
        for i, region in enumerate(regions):
            cap_pot = 0
            for index, shape in shapes.iterrows():
                try:
                    intersection = region.intersection(shape["geometry"])
                except TopologicalError:
                    logger.info(f"Warning: Problem with shape for code {index}")
                    continue
                if intersection.is_empty or intersection.area == 0.:
                    continue
                cap_pot += potential_per_subregion_ds[index]*intersection.area/shape["geometry"].area
                try:
                    region = region.difference(intersection)
                except TopologicalError:
                    logger.info(f"Warning: Problem with shape for code {index}")
                if region.is_empty or region.area == 0.:
                    break
            capacity_potential_ds.loc[tech, i] = cap_pot

    return capacity_potential_ds


def get_capacity_potential_for_countries(tech: str, countries: List[str]) -> pd.Series:
    """
    Get capacity potential (in GW) for a given technology for all countries for which it is available.
    If data is not available for one of the given countries, there will be no entry for that country
     in the returned series.

    Parameters
    ----------
    tech: str
        One of ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    countries: List[str]
        List of ISO codes of countries

    Returns
    -------
    capacity_potential_ds: pd.Series
        Gives for each pair of technology and region the associated potential capacity in GW

    """
    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    capacity_potential_ds = read_capacity_potential(tech, nuts_type='nuts0')

    # Convert EEZ names to country names
    if tech in ['wind_offshore', 'wind_floating']:
        capacity_potential_ds.index = [code[2:] for code in capacity_potential_ds.index]

    # Change 'UK' to 'GB' and 'EL' to 'GR'
    capacity_potential_ds.rename(index={'UK': 'GB', 'EL': 'GR'}, inplace=True)

    # Extract only countries for which data is available
    countries = sorted(list(set(countries) & set(capacity_potential_ds.index)))
    capacity_potential_ds = capacity_potential_ds.loc[countries]

    return capacity_potential_ds


if __name__ == '__main__':
    print(get_capacity_potential_for_countries('wind_offshore', ["CH", "BE", "IE", "GB"]))
