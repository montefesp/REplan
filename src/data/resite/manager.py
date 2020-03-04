import numpy as np
from shapely.geometry import Point, Polygon
import shapely
from os.path import join, dirname, abspath
import pickle
from src.data.res_potential.manager import get_potential_ehighway
from typing import *
import pypsa
from src.data.geographics.manager import is_onshore, match_points_to_region
from src.resite.resite import Resite
import pandas as pd
import yaml
from itertools import product

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators_pypsa(network: pypsa.Network, onshore_region_shape, gen_costs: Dict[str, Dict[str, float]],
                         strategy: str, site_nb: int, area_per_site: int, cap_dens_dict: Dict[str, float]) \
        -> pypsa.Network:
    """Adds wind and pv generator that where selected via a certain siting method to a Network class.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with regions
    onshore_region_shape: Polygon
        Sum of all the onshore regions associated to the buses in network
    gen_costs: Dict[str, Dict[str, float]]
        Dictionary containing opex and capex for solar and wind generators
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
                          "../../../data/resite/generated/" + strategy + "_site_data_" + str(site_nb) + ".p")
    selected_points = pickle.load(open(resite_data_fn, "rb"))

    # regions = network.buses.region.values
    onshore_buses = network.buses[network.buses.onshore]
    # onshore_bus_positions = [Point(x, y) for x, y in zip(onshore_buses.x, onshore_buses.y)]
    offshore_buses = network.buses[network.buses.onshore == False]
    offshore_bus_positions = [Point(x, y) for x, y in zip(offshore_buses.x, offshore_buses.y)]

    # TODO: would probably be nice to invert the dictionary
    #  so that for each point we have the technology(ies) we need to install there
    for tech in selected_points:

        # Get the real tech
        tech_1 = tech.split("_")[0]
        if tech_1 == "solar":
            tech_1 = "pv"
        points_dict = selected_points[tech]

        # Detect to which bus the node should be associated
        if len(offshore_bus_positions) != 0:
            bus_ids = [(onshore_buses.index[
                   np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
                   if is_onshore(Point(point), onshore_region_shape)
                   else offshore_buses.index[
                   np.argmin([bus_pos.distance(Point(point[0], point[1])) for bus_pos in offshore_bus_positions])])
                   for point in points_dict]
        else:
            bus_ids = [onshore_buses.index[
                            np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
                       for point in points_dict]

        # Get capacities for each bus
        # TODO: should add a parameter to differentiate between the two cases
        bus_ids_unique = list(set(bus_ids))
        bus_capacities_per_km = get_potential_ehighway(bus_ids_unique, tech_1).values*1000.0
        bus_capacity_per_km_dict = dict.fromkeys(bus_ids_unique)
        for i, key in enumerate(bus_ids_unique):
            bus_capacity_per_km_dict[key] = bus_capacities_per_km[i]

        for i, point in enumerate(points_dict):

            bus_id = bus_ids[i]
            # Define the capacities per km from parameters if existing
            capacity_per_km = cap_dens_dict[tech_1]
            if capacity_per_km == "":
                capacity_per_km = bus_capacity_per_km_dict[bus_id]

            cap_factor_series = points_dict[point][0:len(network.snapshots)]
            network.add("Generator", "Gen " + tech_1 + " " + str(point[0]) + "-" + str(point[1]),
                        bus=bus_id,
                        p_nom_extendable=True,
                        p_nom_max=capacity_per_km * area_per_site*1000,
                        p_max_pu=cap_factor_series,
                        type=tech_1,
                        carrier=tech_1,
                        x=point[0],
                        y=point[1],
                        marginal_cost=gen_costs[tech_1]["opex"]/1000.0,
                        capital_cost=gen_costs[tech_1]["capex"]*len(network.snapshots)/(8760*1000.0))

    return network


def add_generators(network: pypsa.Network, region: str, gen_costs: Dict[str, Any], use_ex_cap: bool):
    """
    This function will add generators for different technologies at a series of location selected via an optimization
    mechanism.

    Parameters
    ----------
    network: pypsa.Network
        A network with region associated to each buses.
    region: str
        Region over which the network is defined
    gen_costs: Dict[str, Any]
        Dictionary containing the prices of technologies
    use_ex_cap: bool
        Whether to use existing capacity

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # TODO: remove parameter region but keep the rest?
    params_fn = join(dirname(abspath(__file__)), "../../resite/config_model.yml")
    params = yaml.load(open(params_fn), Loader=yaml.FullLoader)

    params["regions"] = [region] # TODO: not sure that is very optimal

    logger.info('Building class.')
    resite = Resite(params)

    logger.info('Reading input.')
    resite.build_input_data(params['filtering_layers'])

    logger.info('Model being built.')
    resite.build_model(params["modelling"], params['formulation'], params['deployment_vector'], params['write_lp'])

    logger.info('Sending model to solver.')
    resite.solve_model(params['solver'], params['solver_options'][params['solver']])

    logger.info('Retrieving results.')
    tech_location_dict = resite.retrieve_sites(save_file=True)  # TODO: parametrize?
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
            # TODO: this is shit change
            offshore_buses_regions = pd.DataFrame(offshore_buses.region.values, index=offshore_buses.index,
                                                  columns=["geometry"])
            associated_buses = match_points_to_region(points, offshore_buses_regions)
        else:
            onshore_buses = network.buses[network.buses.onshore]
            # TODO: this is shit change
            onshore_buses_regions = pd.DataFrame(onshore_buses.region.values, index=onshore_buses.index,
                                                 columns=["geometry"])
            associated_buses = match_points_to_region(points, onshore_buses_regions)

        existing_cap = 0
        if use_ex_cap:
            existing_cap = existing_cap_ds[tech][points].values

        network.madd("Generator",
                     "Gen " + tech + " " + pd.Index([str(x) for x, _ in points]) + "-" +
                     pd.Index([str(y) for _, y in points]),
                     bus=associated_buses["subregion"].values,
                     p_nom_extendable=True,
                     p_nom_max=cap_potential_ds[tech][points].values,
                     p_nom_min=existing_cap,
                     p_max_pu=cap_factor_df[tech][points].values,
                     type=tech,
                     carrier=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=gen_costs[tech.split("_")[0]]["opex"] / 1000.0,
                     capital_cost=gen_costs[tech.split("_")[0]]["capex"] * len(network.snapshots) / (8760 * 1000.0))

    return network
