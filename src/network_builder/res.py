from os.path import join, dirname, abspath
import pickle
from typing import *

import numpy as np
import pandas as pd
from datetime import datetime
from shapely.geometry import Point, Polygon, MultiPolygon

import pypsa

from src.data.resource.manager import get_cap_factor_for_regions, get_cap_factor_at_points, \
    read_resource_database, compute_capacity_factors
from src.data.land_data.manager import filter_points
from src.data.geographics.manager import is_onshore, get_nuts_area, match_points_to_region, return_points_in_shape
from src.data.res_potential.manager import get_capacity_potential, get_potential_ehighway, \
    get_capacity_potential_for_regions
# TODO: this shoulnd't be here normally
from src.data.topologies.ehighway import get_ehighway_clusters
from src.resite.resite import Resite
from src.tech_parameters.costs import get_cost

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: this will not work anymore because now we have costs for different type of wind and pv
# def add_generators_from_file(network: pypsa.Network, onshore_region_shape, gen_costs: Dict[str, Dict[str, float]],
#                          strategy: str, site_nb: int, area_per_site: int, cap_dens_dict: Dict[str, float]) \
#         -> pypsa.Network:
#     """Adds wind and pv generator that where selected via a certain siting method to a Network class.
#
#     Parameters
#     ----------
#     network: pypsa.Network
#         A Network instance with regions
#     onshore_region_shape: Polygon
#         Sum of all the onshore regions associated to the buses in network
#     gen_costs: Dict[str, Dict[str, float]]
#         Dictionary containing opex and capex for solar and wind generators
#     strategy: str
#         "comp" or "max, strategy used to select the sites
#     site_nb: int
#         Number of generation sites to add
#     area_per_site: int
#         Area per site in km2
#     cap_dens_dict: Dict[str, float]
#         Dictionary giving per technology the max capacity per km2
#
#     Returns
#     -------
#     network: pypsa.Network
#         Updated network
#     """
#
#     resite_data_fn = join(dirname(abspath(__file__)),
#                           "../../../data/resite/generated/" + strategy + "_site_data_" + str(site_nb) + ".p")
#     selected_points = pickle.load(open(resite_data_fn, "rb"))
#
#     # regions = network.buses.region.values
#     onshore_buses = network.buses[network.buses.onshore]
#     # onshore_bus_positions = [Point(x, y) for x, y in zip(onshore_buses.x, onshore_buses.y)]
#     offshore_buses = network.buses[network.buses.onshore == False]
#     offshore_bus_positions = [Point(x, y) for x, y in zip(offshore_buses.x, offshore_buses.y)]
#
#     # TODO: would probably be nice to invert the dictionary
#     #  so that for each point we have the technology(ies) we need to install there
#     for tech in selected_points:
#
#         # Get the real tech
#         tech_1 = tech.split("_")[0]
#         if tech_1 == "solar":
#             tech_1 = "pv"
#         points_dict = selected_points[tech]
#
#         # Detect to which bus the node should be associated
#         if len(offshore_bus_positions) != 0:
#             bus_ids = [(onshore_buses.index[
#                    np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
#                    if is_onshore(Point(point), onshore_region_shape)
#                    else offshore_buses.index[
#                    np.argmin([bus_pos.distance(Point(point[0], point[1])) for bus_pos in offshore_bus_positions])])
#                    for point in points_dict]
#         else:
#             bus_ids = [onshore_buses.index[
#                             np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
#                        for point in points_dict]
#
#         # Get capacities for each bus
#         # TODO: should add a parameter to differentiate between the two cases
#         bus_ids_unique = list(set(bus_ids))
#         bus_capacities_per_km = get_potential_ehighway(bus_ids_unique, tech_1).values*1000.0
#         bus_capacity_per_km_dict = dict.fromkeys(bus_ids_unique)
#         for i, key in enumerate(bus_ids_unique):
#             bus_capacity_per_km_dict[key] = bus_capacities_per_km[i]
#
#         for i, point in enumerate(points_dict):
#
#             bus_id = bus_ids[i]
#             # Define the capacities per km from tech_parameters if existing
#             capacity_per_km = cap_dens_dict[tech_1]
#             if capacity_per_km == "":
#                 capacity_per_km = bus_capacity_per_km_dict[bus_id]
#
#             cap_factor_series = points_dict[point][0:len(network.snapshots)]
#             network.add("Generator", "Gen " + tech_1 + " " + str(point[0]) + "-" + str(point[1]),
#                         bus=bus_id,
#                         p_nom_extendable=True,
#                         p_nom_max=capacity_per_km * area_per_site*1000,
#                         p_max_pu=cap_factor_series,
#                         type=tech_1,
#                         carrier=tech_1,
#                         x=point[0],
#                         y=point[1],
#                         marginal_cost=gen_costs[tech_1]["opex"]/1000.0,
#                         capital_cost=gen_costs[tech_1]["capex"]*len(network.snapshots)/(8760*1000.0))
#
#     return network


def add_generators(network: pypsa.Network, params: Dict[str, Any], tech_config: Dict[str, Any], region: str) \
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

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    logger.info('Building class.')
    resite = Resite([region], params["technologies"], tech_config, params["timeslice"], params["spatial_resolution"],
                    params["keep_files"])

    logger.info('Reading input.')
    resite.build_input_data(params['use_ex_cap'], params['filtering_layers'])

    logger.info('Model being built.')
    resite.build_model(params["modelling"], params['formulation'], params['deployment_vector'], params['write_lp'])

    logger.info('Sending model to solver.')
    resite.solve_model(params['solver'], params['solver_options'][params['solver']])

    logger.info('Retrieving results.')
    tech_location_dict = resite.retrieve_solution()  # TODO: parametrize?
    existing_cap_ds, cap_potential_ds, cap_factor_df = resite.retrieve_sites_data()

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
                     p_nom_min=existing_cap,
                     p_max_pu=cap_factor_df[tech][points].values,
                     type=tech,
                     carrier=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


def add_generators_at_resolution(network: pypsa.Network, total_shape: Union[Polygon, MultiPolygon], regions: List[str],
                                 technologies: List[str], tech_config: Dict[str, Any], spatial_resolution: float,
                                 filtering_layers: Dict[str, bool]) \
        -> pypsa.Network:
    """
    Creates pv and wind generators for every coordinate at a resolution of 0.5 inside the region associate to each bus
    and attach them to the corresponding bus.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    total_shape: Polygon
        Sum of all the regions associated to the buses in network

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # TODO: add existing capacity

    # Obtain the list of point in the geographical region
    # TODO: Need to remove the first init_points by downloading new data
    path_resource_data = join(dirname(abspath(__file__)), '../../../data/resource/' + str(spatial_resolution))
    database = read_resource_database(path_resource_data)
    init_points = list(zip(database.longitude.values, database.latitude.values))
    init_points = return_points_in_shape(total_shape, spatial_resolution, init_points)

    tech_points_dict = filter_points(technologies, tech_config, init_points, spatial_resolution, filtering_layers)

    # Get capacity factors for all points
    cap_factor_df = compute_capacity_factors(tech_points_dict, tech_config, spatial_resolution, network.snapshots)

    # Get capacity per point
    cap_potential_ds = get_capacity_potential(tech_points_dict, spatial_resolution, regions)

    for tech in technologies:

        points = tech_points_dict[tech]

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
                     p_nom_max=cap_potential_ds[tech][points].values*1000,
                     # p_nom_min=existing_cap,
                     p_max_pu=cap_factor_df[tech][points].values,
                     type=tech,
                     carrier=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


def add_generators_per_bus(network: pypsa.Network, technologies: List[str], tech_config: Dict[str, Any]) -> pypsa.Network:
    """
    Adds pv and wind generators to each bus of a PyPSA Network, each bus being associated to a geographical region.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    technologies: List[str]
        Technologies to each bus

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Compute capacity potential for each bus region
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

    # Compute capacity factors
    cap_factor_df = compute_capacity_factors(tech_points_dict, tech_config, spatial_res, network.snapshots)

    for tech in technologies:

        is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
        buses = network.buses[network.buses.onshore == is_onshore]

        capital_cost, marginal_cost = get_cost(tech, len(network.snapshots))

        # Adding to the network
        network.madd("Generator", f"Gen {tech} " + buses.index,
                     bus=buses.index,
                     p_nom_extendable=True,
                     p_nom_max=cap_pot_ds[tech].values*1000,  # Convert to MW
                     p_max_pu=cap_factor_df[tech].values,
                     type=tech,
                     carrier=tech,
                     x=buses.x,
                     y=buses.y,
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network

# missing_wind_dict = {
#     "AL": ["BG"],
#     "BA": ["HR"],
#     "ME": ["HR"],
#     "RS": ["BG"]
# }


# TODO: this probably needs to disappear, not sure
# TODO: the problem when including offshore in this function is what area to consider?
# def add_generators_without_siting(network: pypsa.Network, technologies: List[str], gen_costs: Dict[str, Any]) \
#         -> pypsa.Network:
#     """Adds wind and pv generator at each node of a Network instance with limited capacities.
#
#     Parameters
#     ----------
#     network: pypsa.Network
#         A PyPSA Network instance with buses associated to regions
#     technologies: List[str]
#         Technologies to add
#     gen_costs: Dict[str, Any]
#         Dictionary containing opex and capex for generation technologies
#
#     Returns
#     -------
#     network: pypsa.Network
#         Updated network
#     """
#
#     profiles_fn = join(dirname(abspath(__file__)),
#                        "../../../data/res_potential/source/ninja_pv_wind_profiles_singleindex.csv")
#     profiles = pd.read_csv(profiles_fn)
#     profiles["time"] = profiles["time"].apply(lambda x: np.datetime64(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")))
#     profiles = profiles.set_index("time")
#     profiles = profiles.loc[network.snapshots]
#
#     eh_clusters = get_ehighway_clusters()
#
#     areas = get_nuts_area()
#     areas.index.name = 'code'
#
#     # TODO: too slow
#     capacity_per_km_pv = get_potential_ehighway(network.buses.index.values, "pv").values
#     capacity_per_km_wind = get_potential_ehighway(network.buses.index.values, "wind").values
#
#     for tech in technologies:
#         for i, bus_id in enumerate(network.buses.index):
#
#             # Get region area
#             area = np.sum(areas.loc[eh_clusters.loc[bus_id]["codes"].split(",")]["2015"])
#
#             # PV
#             country_pv_profile = profiles[eh_clusters.loc[bus_id].country + "_pv_national_current"]
#
#             # Add a pv generator
#             capacity_per_km = capacity_per_km_pv[i]
#             network.add("generator", "Gen " + bus_id + " pv",
#                         bus=bus_id,
#                         p_nom_extendable=True, # consider that the tech can be deployed on 50*50 km2
#                         p_nom_max=capacity_per_km * area * 1000,
#                         p_max_pu=country_pv_profile.values,
#                         type="pv",
#                         carrier="pv",
#                         marginal_cost=gen_costs[tech]["opex"]/1000.0,
#                         capital_cost=gen_costs[tech]["capex"]*len(network.snapshots)/(8760*1000.0))
#
#             # Wind
#             replacing_country = eh_clusters.loc[bus_id].country
#             if eh_clusters.loc[bus_id].country in missing_wind_dict:
#                 replacing_country = missing_wind_dict[replacing_country][0]
#             if replacing_country + "_wind_onshore_current" in profiles.keys():
#                 country_wind_profile = profiles[replacing_country + "_wind_onshore_current"]
#             else:
#                 country_wind_profile = profiles[replacing_country + "_wind_national_current"]
#
#             # Add a wind generator
#             capacity_per_km = capacity_per_km_wind[i]
#             network.add("generator", "Gen " + bus_id + " wind",
#                         bus=bus_id,
#                         p_nom_extendable=True, # consider that the tech can be deployed on 50*50 km2
#                         p_nom_max=capacity_per_km * area * 1000,
#                         p_max_pu=country_wind_profile.values,
#                         type="wind",
#                         carrier="wind",
#                         marginal_cost=gen_costs[tech]["opex"]/1000.0,
#                         capital_cost=gen_costs[tech]["capex"]*len(network.snapshots)/(8760*1000.0))
#
#     return network
