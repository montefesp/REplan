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
        offshore_potential = pd.read_csv(f"{path_potential_data}eez_capacity_potentials_GW.csv", index_col=0)[tech]
        return offshore_potential


def get_capacity_potential_at_points(tech_points_dict: Dict[str, List[Tuple[float, float]]],
                                     spatial_resolution: float, regions: List[str],
                                     existing_capacity_ds: pd.Series = None) -> pd.Series:
    """
    Computes the capacity that can potentially be deployed at a series of points for different technologies

    Parameters
    ----------
    tech_points_dict : Dict[str, Dict[str, List[Tuple[float, float]]]
        Dictionary associating to each tech a list of points.
    spatial_resolution : float
        Spatial resolution of the points.
    regions: List[str]
        Codes of geographical regions in which the points are situated
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

    # Create a modified copy of regions to deal with UK and EL
    nuts0_problems = {"GB": "UK", "GR": "EL"}
    nuts0_regions = [nuts0_problems[r] if r in nuts0_problems else r for r in regions]

    array_pop_density = load_population_density_data(spatial_resolution)

    tech_coords_tuples = sorted([(tech, point) for tech, points in tech_points_dict.items() for point in points])
    capacity_potential_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_coords_tuples))

    for tech, coords in tech_points_dict.items():

        # Compute potential for each NUTS2 or EEZ
        potential_per_region_ds = read_capacity_potential(tech, nuts_type='nuts2')

        # Get NUTS2 and EEZ shapes
        # TODO: this is shit -> not generic enough, e.g.: would probably not work for us states
        #  would need to get this out of the loop
        if tech in ['wind_offshore', 'wind_floating']:
            onshore_shapes_union = \
                cascaded_union(get_onshore_shapes(regions, filterremote=True,
                                                  save_file_name=f"{''.join(sorted(regions))}_regions_on.geojson")
                               ["geometry"].values)
            filter_shape_data = get_offshore_shapes(regions, onshore_shape=onshore_shapes_union,
                                                    filterremote=True,
                                                    save_file_name=f"{''.join(sorted(regions))}_regions_off.geojson")
            filter_shape_data.index = [f"EZ{code}" for code in filter_shape_data.index]
        else:
            codes = [code for code in potential_per_region_ds.index if code[:2] in nuts0_regions]
            filter_shape_data = get_onshore_shapes(codes, filterremote=True,
                                                   save_file_name=f"{''.join(sorted(regions))}_nuts2_on.geojson")

        # Find the geographical region code associated to each coordinate
        coords_regions_ds = match_points_to_regions(coords, filter_shape_data["geometry"]).dropna()
        coords = list(coords_regions_ds.index)
        coords_regions_df = pd.DataFrame(coords_regions_ds.values, coords_regions_ds.index,
                                         columns=["region"])

        if tech in ['wind_offshore', 'wind_floating']:

            # For offshore sites, divide the total potential of the region by the number of coordinates
            # associated to that region
            # TODO: change variable names
            region_freq_ds = coords_regions_df.groupby(['region'])['region'].count()
            region_freq_df = pd.DataFrame(region_freq_ds.values, index=region_freq_ds.index, columns=['freq'])
            region_freq_df["cap_pot"] = potential_per_region_ds[region_freq_df.index]
            coords_regions_df = \
                coords_regions_df.merge(region_freq_df, left_on='region', right_on='region', right_index=True)
            capacity_potential = coords_regions_df["cap_pot"]/coords_regions_df["freq"]
            capacity_potential_ds.loc[tech, capacity_potential.index] = capacity_potential.values

        elif tech in ['wind_onshore', 'pv_utility', 'pv_residential']:

            # TODO: change variable names
            coords_regions_df['pop_dens'] = \
                 np.clip(array_pop_density.sel(locations=coords).values, a_min=1., a_max=None)

            if tech in ['wind_onshore', 'pv_utility']:
                coords_regions_df['pop_dens'] = 1./coords_regions_df['pop_dens']

            # Keep only the potential of regions in which points fall
            coords_to_regions_df_sum = coords_regions_df.groupby(['region']).sum()
            coords_to_regions_df_sum["cap_pot"] = potential_per_region_ds[coords_to_regions_df_sum.index]
            coords_to_regions_df_sum.columns = ['sum_per_region', 'cap_pot']
            coords_to_regions_df_merge = \
                coords_regions_df.merge(coords_to_regions_df_sum,
                                        left_on='region', right_on='region', right_index=True)

            capacity_potential_per_coord = coords_to_regions_df_merge['pop_dens'] * \
                coords_to_regions_df_merge['cap_pot']/coords_to_regions_df_merge['sum_per_region']
            capacity_potential_ds.loc[tech, capacity_potential_per_coord.index] = capacity_potential_per_coord.values

    # Update capacity potential with existing potential if present
    if existing_capacity_ds is not None:
        underestimated_capacity = existing_capacity_ds > capacity_potential_ds
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
        # TODO: this is shit -> not generic enough, e.g.: would probably not work for us states
        #  would need to get this out of the loop
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
            # TODO: problem BA00 does not exists in shapes

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

    print(capacity_potential_ds)
    # Change 'UK' to 'GB' and 'EL' to 'GR'
    capacity_potential_ds.rename(index={'UK': 'GB', 'EL': 'GR'}, inplace=True)

    # Extract only countries for which data is available
    countries = sorted(list(set(countries) & set(capacity_potential_ds.index)))
    capacity_potential_ds = capacity_potential_ds.loc[countries]

    return capacity_potential_ds


if __name__ == '__main__':
    print(get_capacity_potential_for_countries('wind_offshore', ["CH", "BE", "IE", "GB"]))
