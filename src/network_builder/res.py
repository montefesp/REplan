from os.path import join, dirname, abspath
import pickle
from typing import List, Dict, Any, Union

import pandas as pd

import pypsa

from src.data.vres_profiles import compute_capacity_factors, get_cap_factor_for_countries
from src.data.geographics import match_points_to_regions, match_points_to_countries, get_shapes
from src.data.vres_potential import get_capacity_potential_for_countries, get_capacity_potential_at_points, \
    get_capacity_potential_for_regions
from src.data.legacy import get_legacy_capacity_in_regions, get_legacy_capacity_in_countries
from src.resite.resite import Resite
from src.data.technologies import get_costs

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: invert path and strategy
# TODO: too many arguments here too.
def add_generators_from_file(network: pypsa.Network, technologies: List[str], strategy: str, path: str,
                             area_per_site: int, spatial_resolution: float, countries: List[str],
                             topology_type: str = "countries", cap_dens_dict: Dict[str, float] = None,
                             offshore_buses: bool = True) -> pypsa.Network:
    """
    Add wind and PV generators based on sites that where selected via a certain siting method to a Network class.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with regions
    technologies: List[str]
        Which technologies we want to add
    path: str
        Path to sites directory
    strategy: str
        "comp" or "max, strategy used to select the sites
    area_per_site: int
        Area per site in km2
    spatial_resolution: float
        Spatial resolution at which the points are defined
    countries: List[str]
        List of ISO codes of countries over which the network is built
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    cap_dens_dict: Dict[str, float] (default: None)
        Dictionary giving per technology the max capacity per km2
    offshore_buses: bool (default: True)
        Whether the network contains offshore buses

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in technologies:
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    assert topology_type == "countries" or "region" in network.buses.columns, \
        "Error: If you are not using a one-node-per-country topology, you must associate regions to buses."

    # Load site data
    resite_data_path = join(dirname(abspath(__file__)), f"../../data/resite/generated/{path}/")
    resite_data_fn = join(resite_data_path, f"{strategy}_site_data.p")
    tech_points_cap_factor_df = pickle.load(open(resite_data_fn, "rb"))

    missing_timestamps = set(network.snapshots) - set(tech_points_cap_factor_df.index)
    assert not missing_timestamps, f"Error: Following timestamps are not part of capacity factors {missing_timestamps}"

    for tech in technologies:

        if tech not in set(tech_points_cap_factor_df.columns.get_level_values(0)):
            print(f"Warning: Technology {tech} is not part of RES data from files. Therefore it was not added.")
            continue

        points = sorted(list(tech_points_cap_factor_df[tech].columns), key=lambda x: x[0])

        # If there are no offshore buses, add all generators to onshore buses
        # If there are offshore buses, add onshore techs to onshore buses and offshore techs to offshore buses
        buses = network.buses.copy()
        if offshore_buses:
            is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
            buses = buses[buses.onshore == is_onshore]

        # Detect to which bus each site should be associated
        if not offshore_buses and topology_type == "countries":
            points_bus_ds = match_points_to_countries(points, list(buses.index)).dropna()
        else:
            points_bus_ds = match_points_to_regions(points, buses.region).dropna()
        points = list(points_bus_ds.index)

        logger.info(f"Adding {tech} in {list(set(points_bus_ds))}.")

        # Compute capacity potential
        if cap_dens_dict is None or tech not in cap_dens_dict:

            # Compute capacity potential for all initial points in region and then keep the one only for selected sites
            init_points_fn = join(resite_data_path, "init_coordinates_dict.p")
            init_points_list = pickle.load(open(init_points_fn, "rb"))[tech]

            # TODO: some weird behaviour happening for offshore, duplicate locations occurring.
            #  To be further checked, ideally this filtering disappears..
            init_points_list = list(set(init_points_list))

            capacity_potential_per_node_full = \
                get_capacity_potential_at_points({tech: init_points_list}, spatial_resolution, countries)[tech]
            points_capacity_potential = list(capacity_potential_per_node_full.loc[points].values)

        else:

            # Use predefined per km capacity multiplied by are_per_site
            bus_capacity_potential_per_km = pd.Series(cap_dens_dict[tech], index=buses.index)
            points_capacity_potential = \
                [bus_capacity_potential_per_km[points_bus_ds[point]]*area_per_site/1e3 for point in points]

        # Get capacity factors
        cap_factor_series = tech_points_cap_factor_df.loc[network.snapshots][tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(network.snapshots))

        network.madd("Generator",
                     pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                     bus=points_bus_ds.values,
                     p_nom_extendable=True,
                     p_nom_max=points_capacity_potential,
                     p_min_pu=0.,
                     p_max_pu=cap_factor_series,
                     type=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


def add_generators_using_siting(network: pypsa.Network, technologies: List[str],
                                params: Dict[str, Any], tech_config: Dict[str, Any], region: str,
                                topology_type: str = 'countries', offshore_buses: bool = True,
                                output_dir: str = None) -> pypsa.Network:
    """
    Add generators for different technologies at a series of location selected via an optimization mechanism.

    Parameters
    ----------
    network: pypsa.Network
        A network with defined buses.
    technologies: List[str]
        Which technologies to add using this methodology
    params
        # TODO: the way this is given to the function and then pass to resite is shitty, depends on resite update
    tech_config
        # TODO: see related issue
    region: str
        Region over which the network is defined
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    offshore_buses: bool (default: True)
        Whether the network contains offshore buses
    output_dir: str
        Absolute path to directory where resite output should be stored

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in technologies:
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    assert topology_type == "countries" or "region" in network.buses.columns, \
        "Error: If you are not using a one-node-per-country topology, you must associate regions to buses."

    logger.info('Setting up resite.')
    resite = Resite([region], technologies, tech_config, params["timeslice"], params["spatial_resolution"])

    resite.build_input_data(params['use_ex_cap'], params['filtering_layers'])

    logger.info('resite model being built.')
    resite.build_model(params["modelling"], params['formulation'], params['formulation_params'],
                       params['write_lp'], output_dir)

    logger.info('Sending resite to solver.')
    resite.solve_model()

    logger.info('Retrieving resite results.')
    tech_location_dict = resite.retrieve_solution()
    existing_cap_ds, cap_potential_ds, cap_factor_df = resite.retrieve_sites_data()

    logger.info("Saving resite results")
    resite.save(output_dir)

    if not resite.timestamps.equals(network.snapshots):
        # If network snapshots is a subset of resite snapshots just crop the data
        missing_timestamps = set(network.snapshots) - set(resite.timestamps)
        if not missing_timestamps:
            cap_factor_df = cap_factor_df.loc[network.snapshots]
        else:
            # In other case, need to recompute capacity factors
            # TODO: to be implemented
            pass

    for tech, points in tech_location_dict.items():

        buses = network.buses.copy()
        if offshore_buses or tech in ["pv_residential", "pv_utility", "wind_onshore"]:
            is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
            buses = buses[buses.onshore == is_onshore]
            associated_buses = match_points_to_regions(points, buses.region).dropna()
        elif topology_type == 'countries':
            associated_buses = match_points_to_countries(points, list(buses.index)).dropna()
        else:
            raise ValueError("If you are not using a one-node-per-country topology, you must define offshore buses.")

        points = list(associated_buses.index)

        existing_cap = 0
        if params['use_ex_cap']:
            existing_cap = existing_cap_ds[tech][points].values

        p_nom_max = 'inf'
        if params['limit_max_cap']:
            p_nom_max = cap_potential_ds[tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(network.snapshots))

        network.madd("Generator",
                     pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                     bus=associated_buses.values,
                     p_nom_extendable=True,
                     p_nom_max=p_nom_max,
                     p_nom=existing_cap,
                     p_nom_min=existing_cap,
                     p_min_pu=0.,
                     p_max_pu=cap_factor_df[tech][points].values,
                     type=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


# TODO: I am not sure how to deal with the inputs here either
def add_generators_at_resolution(network: pypsa.Network, technologies: List[str], regions: List[str],
                                 tech_config: Dict[str, Any], spatial_resolution: float,
                                 filtering_layers: Dict[str, bool], use_ex_cap: bool, limit_max_cap: bool = False,
                                 topology_type: str = 'countries', offshore_buses: bool = True) -> pypsa.Network:
    """
    Create PV and wind generators for every coordinate at a given resolution inside the region associate to each bus
    and attach them to the corresponding bus.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    technologies: List[str]
        Which technologies to add
    # TODO: shouldn't we pass directly the list of countries? - need to change that in resite too
    regions: List[str]
        List of regions code defined in data/region_definition.csv over which the network is defined
    tech_config
        # TODO: comment
    spatial_resolution
    filtering_layers
    use_ex_cap
    limit_max_cap
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    offshore_buses: bool (default: True)
        Whether the network contains offshore buses

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in technologies:
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    assert topology_type == "countries" or "region" in network.buses.columns, \
        "Error: If you are not using a one-node-per-country topology, you must associate regions to buses."

    # Generate input data using resite
    resite = Resite(regions, technologies, tech_config, [network.snapshots[0], network.snapshots[-1]],
                    spatial_resolution)
    resite.build_input_data(use_ex_cap, filtering_layers)

    for tech in technologies:

        points = resite.tech_points_dict[tech]

        buses = network.buses.copy()
        if offshore_buses or tech in ["pv_residential", "pv_utility", "wind_onshore"]:
            is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
            buses = buses[buses.onshore == is_onshore]
            associated_buses = match_points_to_regions(points, buses.region).dropna()
        elif topology_type == 'countries':
            associated_buses = match_points_to_countries(points, list(buses.index)).dropna()
        else:
            raise ValueError("If you are not using a one-node-per-country topology, you must define offshore buses.")
        points = list(associated_buses.index)

        existing_cap = 0
        if use_ex_cap:
            existing_cap = resite.existing_capacity_ds[tech][points].values

        p_nom_max = 'inf'
        if limit_max_cap:
            p_nom_max = resite.cap_potential_ds[tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(network.snapshots))

        network.madd("Generator",
                     pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                     bus=associated_buses.values,
                     p_nom_extendable=True,
                     p_nom_max=p_nom_max,
                     p_nom=existing_cap,
                     p_nom_min=existing_cap,
                     p_min_pu=0.,
                     p_max_pu=resite.cap_factor_df[tech][points].values,
                     type=tech,
                     x=[x for x, _ in points],
                     y=[y for _, y in points],
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


def add_generators_per_bus(network: pypsa.Network, technologies: List[str],
                           converters: Dict[str, Union[Dict[str, str], str]],
                           countries: List[str], use_ex_cap: bool = True,
                           topology_type: str = 'countries', offshore_buses: bool = True) -> pypsa.Network:
    """
    Add PV and wind generators to each bus of a PyPSA Network, each bus being associated to a geographical region.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    technologies: List[str]
        Technologies to each bus
    converters: Dict[str, Union[Dict[str, str], str]]
        Dictionary indicating for each technology which converter(s) to use.
        For each technology in the dictionary:
            - if it is pv-based, the name of the converter must be specified as a string
            - if it is wind, a dictionary must be defined associated for the four wind regimes
            defined below (I, II, III, IV), the name of the converter as a string
    countries: List[str]
      List of ISO codes of countries over which the network is defined
    use_ex_cap: bool (default: True)
        Whether to take into account existing capacity
    topology_type: str
        Can currently be countries (for one node per country topologies)
        or ehighway (for topologies based on ehighway)
    offshore_buses: bool (default: True)
        Whether the network contains offshore buses


    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in technologies:
        assert tech in accepted_techs, f"Error: tech {tech} is not in {accepted_techs}"

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    spatial_res = 0.5
    for tech in technologies:

        # If there are no offshore buses, add all generators to onshore buses
        # If there are offshore buses, add onshore techs to onshore buses and offshore techs to offshore buses
        buses = network.buses.copy()
        if offshore_buses:
            is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
            buses = buses[buses.onshore == is_onshore]

        # Get the shapes of regions associated to each bus
        # TODO: maybe add a condition where we check if regions are defined per bus
        if not offshore_buses and tech in ['wind_offshore', 'wind_floating']:
            offshore_shapes = get_shapes(list(buses.index), which='offshore', save_file_str='countries')
            buses = buses.loc[offshore_shapes.index]
            buses_regions_shapes_ds = offshore_shapes["geometry"]
        else:
            buses_regions_shapes_ds = buses.region

        # Compute capacity potential at each bus
        if topology_type == "countries":
            cap_pot_ds = get_capacity_potential_for_countries(tech, buses.index)
        else:  # topology_type == "ehighway"
            cap_pot_ds = get_capacity_potential_for_regions({tech: buses_regions_shapes_ds.values})[tech]
            cap_pot_ds.index = buses.index

        # Get one capacity factor time series per bus
        if topology_type == 'countries':
            cap_factor_df = get_cap_factor_for_countries(tech, buses.index, network.snapshots)
        else:
            # Compute capacity factors at buses position
            if not offshore_buses and tech in ['wind_offshore', 'wind_floating']:
                points = [(round(shape.centroid.x/spatial_res) * spatial_res,
                           round(shape.centroid.y/spatial_res) * spatial_res)
                          for shape in buses_regions_shapes_ds.values]
            else:
                points = [(round(x/spatial_res)*spatial_res,
                           round(y/spatial_res)*spatial_res)
                          for x, y in buses[["x", "y"]].values]
            cap_factor_df = compute_capacity_factors({tech: points}, spatial_res, network.snapshots, converters)[tech]
            cap_factor_df.columns = buses.index

        legacy_capacities = pd.Series(0., index=buses.index)
        # Compute legacy capacity (not available for wind_floating)
        if use_ex_cap and tech != "wind_floating":
            if topology_type == 'countries':
                legacy_capacities = get_legacy_capacity_in_countries(tech, buses.index)
            else:
                legacy_capacities = get_legacy_capacity_in_regions(tech, buses_regions_shapes_ds, countries)

        # Update capacity potentials if legacy capacity is bigger
        for bus in buses.index:
            if cap_pot_ds.loc[bus] < legacy_capacities.loc[bus]:
                cap_pot_ds.loc[bus] = legacy_capacities.loc[bus]

        # Get costs
        capital_cost, marginal_cost = get_costs(tech, len(network.snapshots))

        # Adding to the network
        network.madd("Generator",
                     f"Gen {tech} " + buses.index,
                     bus=buses.index,
                     p_nom_extendable=True,
                     p_nom=legacy_capacities.values,
                     p_nom_min=legacy_capacities.values,
                     p_nom_max=cap_pot_ds.values,
                     p_min_pu=0.,
                     p_max_pu=cap_factor_df.values,
                     type=tech,
                     x=buses.x.values,
                     y=buses.y.values,
                     marginal_cost=marginal_cost,
                     capital_cost=capital_cost)

    return network


# TODO: to be removed
# def add_generators_at_bus_test(network: pypsa.Network, params: Dict[str, Any], tech_config: Dict[str, Any],
#                                region: str, output_dir: str = None) -> pypsa.Network:
#     """
#     Add generators for different technologies at a series of location selected via an optimization mechanism.
#
#     Parameters
#     ----------
#     network: pypsa.Network
#         A network with region associated to each buses.
#     tech_config
#         # ODO comment
#     params
#         # ODO comment
#     region: str
#         Region over which the network is defined
#     output_dir: str
#         Absolute path to directory where resite output should be stored
#
#     Returns
#     -------
#     network: pypsa.Network
#         Updated network
#     """
#
#     logger.info('Setting up resite.')
#     resite = Resite([region], params["technologies"], tech_config, params["timeslice"], params["spatial_resolution"],
#                     params["keep_files"])
#
#     resite.build_input_data(params['use_ex_cap'], params['filtering_layers'])
#
#     logger.info('resite model being built.')
#     resite.build_model(params["modelling"], params['formulation'], params['formulation_params'], params['write_lp'])
#
#     logger.info('Sending resite to solver.')
#     resite.solve_model()
#
#     logger.info('Retrieving resite results.')
#     tech_location_dict = resite.retrieve_solution()
#     existing_cap_ds, cap_potential_ds, cap_factor_df = resite.retrieve_sites_data()
#
#     logger.info("Saving resite results")
#     resite.save(params, output_dir)
#
#     if not resite.timestamps.equals(network.snapshots):
#         # If network snapshots is a subset of resite snapshots just crop the data
#         missing_timestamps = set(network.snapshots) - set(resite.timestamps)
#         if not missing_timestamps:
#             cap_factor_df = cap_factor_df.loc[network.snapshots]
#         else:
#             # In other case, need to recompute capacity factors
#             # ODO: to be implemented
#             pass
#
#     for tech, points in tech_location_dict.items():
#
#         if tech in ['wind_offshore', 'wind_floating']:
#             offshore_buses = network.buses[~network.buses.onshore]
#             associated_buses = match_points_to_regions(points, offshore_buses.region).dropna()
#         else:
#             onshore_buses = network.buses[network.buses.onshore]
#             associated_buses = match_points_to_regions(points, onshore_buses.region).dropna()
#
#         # Get only one point per bus
#         buses_points_df = pd.DataFrame(list(associated_buses.index), index=associated_buses.values)
#         buses_points_df["Locations"] = buses_points_df[[0, 1]].apply(lambda x: [(x[0], (x[1]))], axis=1)
#         buses_points_df = buses_points_df.drop([0, 1], axis=1)
#         one_point_per_bus = buses_points_df.groupby(buses_points_df.index).sum().apply(lambda x: x[0][0], axis=1)
#         points = one_point_per_bus.values
#
#         existing_cap = 0
#         if params['use_ex_cap']:
#             existing_cap = existing_cap_ds[tech][points].values
#
#         p_nom_max = 'inf'
#         if params['limit_max_cap']:
#             p_nom_max = cap_potential_ds[tech][points].values
#
#         capital_cost, marginal_cost = get_costs(tech, len(network.snapshots))
#
#         network.madd("Generator",
#                      pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
#                      bus=one_point_per_bus.index,
#                      p_nom_extendable=True,
#                      p_nom_max=p_nom_max,
#                      p_nom=existing_cap,
#                      p_nom_min=existing_cap,
#                      p_max_pu=cap_factor_df[tech][points].values,
#                      type=tech,
#                      x=[x for x, _ in points],
#                      y=[y for _, y in points],
#                      marginal_cost=marginal_cost,
#                      capital_cost=capital_cost)
#
#     return network
