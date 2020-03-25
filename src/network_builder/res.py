from os.path import join, dirname, abspath
import pickle
from typing import *

import numpy as np
import pandas as pd
from datetime import datetime
from shapely.geometry import Point, Polygon, MultiPolygon

import pypsa

from src.data.resource.manager import compute_capacity_factors
from src.data.geographics.manager import is_onshore, match_points_to_region
from src.data.res_potential.manager import get_capacity_potential_for_regions, get_potential_ehighway
from src.data.legacy.manager import get_legacy_capacity_in_regions
from src.resite.resite import Resite
from src.parameters.costs import get_cost

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: this function only allows to repeat previous results, will probably disappear
def add_generators_from_file(network: pypsa.Network, onshore_region_shape, strategy: str, site_nb: int,
                             area_per_site: int, cap_dens_dict: Dict[str, float]) \
        -> pypsa.Network:
    """Adds wind and pv generator that where selected via a certain siting method to a Network class.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with regions
    onshore_region_shape: Polygon
        Sum of all the onshore regions associated to the buses in network
    strategy: str
        "comp" or "max, strategy used to select the sites
    site_nb: int
        Number of generation sites to add
    area_per_site: int
        Area per site in km2
    cap_dens_dict: Dict[str, float]
        Dictionary giving per technology the max capacity per km2

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    resite_data_fn = join(dirname(abspath(__file__)),
                          "../../data/resite/generated/" + strategy + "_site_data_" + str(site_nb) + ".p")
    selected_points = pickle.load(open(resite_data_fn, "rb"))

    # regions = network.buses.region.values
    onshore_buses = network.buses[network.buses.onshore]
    # onshore_bus_positions = [Point(x, y) for x, y in zip(onshore_buses.x, onshore_buses.y)]
    # offshore_buses = network.buses[network.buses.onshore == False]
    # offshore_bus_positions = [Point(x, y) for x, y in zip(offshore_buses.x, offshore_buses.y)]

    for tech, points_dict in selected_points.items():

        # Get the real tech
        tech = tech.split("_")[0]
        if tech == "solar":
            tech = "pv_utility"
        elif tech == "wind":
            tech = "wind_onshore"

        # Detect to which bus the node should be associated
        points_bus_ds = match_points_to_region(list(points_dict.keys()), onshore_buses.region)

        # if len(offshore_bus_positions) != 0:
        #     bus_ids = [(onshore_buses.index[
        #            np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
        #            if is_onshore(Point(point), onshore_region_shape)
        #            else offshore_buses.index[
        #            np.argmin([bus_pos.distance(Point(point[0], point[1])) for bus_pos in offshore_bus_positions])])
        #            for point in points_dict]
        # else:
        #     bus_ids = [onshore_buses.index[
        #                     np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
        #                for point in points_dict]

        # Get capacities for each bus
        bus_capacities_per_km = get_potential_ehighway(onshore_buses.index, tech).values
        bus_capacity_per_km_dict = dict.fromkeys(onshore_buses.index)
        for i, key in enumerate(onshore_buses.index):
            bus_capacity_per_km_dict[key] = bus_capacities_per_km[i]

        capital_cost, marginal_cost = get_cost(tech, len(network.snapshots))

        for i, point in enumerate(points_dict):

            bus_id = points_bus_ds.iloc[i]
            # Define the capacities per km from parameters if existing
            capacity_per_km = cap_dens_dict[tech]
            if capacity_per_km == "":
                capacity_per_km = bus_capacity_per_km_dict[bus_id]

            # TODO: this is super shitty
            cap_factor_series = points_dict[point][0:len(network.snapshots)]

            network.add("Generator", "Gen " + tech + " " + str(point[0]) + "-" + str(point[1]),
                        bus=bus_id,
                        p_nom_extendable=True,
                        p_nom_max=capacity_per_km * area_per_site,
                        p_max_pu=cap_factor_series,
                        type=tech,
                        x=point[0],
                        y=point[1],
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost)

    return network


def add_generators(network: pypsa.Network, params: Dict[str, Any], tech_config: Dict[str, Any], region: str,
                   output_dir = None) \
        -> pypsa.Network:
    """
    This function will add generators for different technologies at a series of location selected via an optimization
    mechanism.

    Parameters
    ----------
    network: pypsa.Network
        A network with region associated to each buses.
    region: str
        Region over which the network is defined
    output_dir: str
        Absolute path to directory where resite output should be stored

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    logger.info('Setting up resite.')
    resite = Resite([region], params["technologies"], tech_config, params["timeslice"], params["spatial_resolution"],
                    params["keep_files"])

    resite.build_input_data(params['use_ex_cap'], params['filtering_layers'])

    logger.info('resite model being built.')
    resite.build_model(params["modelling"], params['formulation'], params['deployment_vector'], params['write_lp'])

    logger.info('Sending resite to solver.')
    resite.solve_model(params['solver'], params['solver_options'][params['solver']], params['write_log'])

    logger.info('Retrieving resite results.')
    tech_location_dict = resite.retrieve_solution()
    existing_cap_ds, cap_potential_ds, cap_factor_df = resite.retrieve_sites_data()

    resite.save(params, output_dir)

    if not resite.timestamps.equals(network.snapshots):
        # If network snapshots is a subset of resite snapshots just crop the data
        # TODO: probably need a better condition
        if network.snapshots[0] in resite.timestamps and network.snapshots[-1] in resite.timestamps:
            cap_factor_df = cap_factor_df.loc[network.snapshots]
        else:
            # In other case, need to recompute capacity factors
            # TODO: to be implemented
            pass

    for tech, points in tech_location_dict.items():

        if tech in ['wind_offshore', 'wind_floating']:
            offshore_buses = network.buses[network.buses.onshore == False]
            associated_buses = match_points_to_region(points, offshore_buses.region)
        else:
            onshore_buses = network.buses[network.buses.onshore]
            associated_buses = match_points_to_region(points, onshore_buses.region)

        existing_cap = 0
        if params['use_ex_cap']:
            existing_cap = existing_cap_ds[tech][points].values

        capital_cost, marginal_cost = get_cost(tech, len(network.snapshots))

        network.madd("Generator",
                     "Gen " + tech + " " + pd.Index([str(x) for x, _ in points]) + "-" +
                     pd.Index([str(y) for _, y in points]),
                     bus=associated_buses.values,
                     p_nom_extendable=True,
                     p_nom_max=cap_potential_ds[tech][points].values,
                     p_nom=existing_cap,
                     p_nom_min=existing_cap,
                     p_max_pu=cap_factor_df[tech][points].values,
                     type=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


# TODO: add existing capacity
def add_generators_at_resolution(network: pypsa.Network, regions: List[str], technologies: List[str],
                                 tech_config: Dict[str, Any], spatial_resolution: float,
                                 filtering_layers: Dict[str, bool], use_ex_cap: bool) \
        -> pypsa.Network:
    """
    Creates pv and wind generators for every coordinate at a resolution of 0.5 inside the region associate to each bus
    and attach them to the corresponding bus.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Generate input data using resite
    resite = Resite(regions, technologies, tech_config, [network.snapshots[0], network.snapshots[-1]],
                    spatial_resolution, False)
    resite.build_input_data(use_ex_cap, filtering_layers)

    for tech in technologies:

        points = resite.tech_points_dict[tech]

        if tech in ['wind_offshore', 'wind_floating']:
            offshore_buses = network.buses[network.buses.onshore == False]
            associated_buses = match_points_to_region(points, offshore_buses.region)
        else:
            onshore_buses = network.buses[network.buses.onshore]
            associated_buses = match_points_to_region(points, onshore_buses.region)

        capital_cost, marginal_cost = get_cost(tech, len(network.snapshots))

        network.madd("Generator",
                     "Gen " + tech + " " + pd.Index([str(x) for x, _ in points]) + "-" +
                     pd.Index([str(y) for _, y in points]),
                     bus=associated_buses.values,
                     p_nom_extendable=True,
                     p_nom_max=resite.cap_potential_ds[tech][points].values,
                     p_max_pu=resite.cap_factor_df[tech][points].values,
                     type=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


def add_generators_per_bus(network: pypsa.Network, technologies: List[str], countries: List[str],
                           tech_config: Dict[str, Any], use_ex_cap: bool = True) -> pypsa.Network:
    """
    Adds pv and wind generators to each bus of a PyPSA Network, each bus being associated to a geographical region.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    technologies: List[str]
        Technologies to each bus
    tech_config:
        # TODO: comment
    use_ex_cap: bool (default: True)
        Whether to take into account existing capacity

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Compute capacity potential and capacity factors
    spatial_res = 0.5
    tech_regions_dict = dict.fromkeys(technologies)
    tech_points_dict = dict.fromkeys(technologies)
    for tech in technologies:

        is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
        buses = network.buses[network.buses.onshore == is_onshore]
        tech_points_dict[tech] = [(round(x/spatial_res)*spatial_res, round(y/spatial_res)*spatial_res)
                      for x, y in buses[["x", "y"]].values]
        tech_regions_dict[tech] = buses.region.values

    cap_pot_ds = get_capacity_potential_for_regions(tech_regions_dict)
    cap_factor_df = compute_capacity_factors(tech_points_dict, tech_config, spatial_res, network.snapshots)

    for tech in technologies:

        is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
        buses = network.buses[network.buses.onshore == is_onshore]

        # Get legacy capacity
        legacy_capacities = 0
        if use_ex_cap and tech in ['wind_onshore', 'wind_offshore', 'pv_utility']:
            legacy_capacities = get_legacy_capacity_in_regions(tech, buses.region, countries).values

        capital_cost, marginal_cost = get_cost(tech, len(network.snapshots))

        cap_potential = cap_pot_ds[tech].values
        for i in range(len(cap_potential)):
            if cap_potential[i] < legacy_capacities[i]:
                cap_potential[i] = legacy_capacities[i]

        # Adding to the network
        network.madd("Generator", f"Gen {tech} " + buses.index,
                     bus=buses.index,
                     p_nom_extendable=True,
                     p_nom=legacy_capacities,
                     p_nom_min=legacy_capacities,
                     p_nom_max=cap_potential,
                     p_max_pu=cap_factor_df[tech].values,
                     type=tech,
                     x=buses.x.values,
                     y=buses.y.values,
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network
