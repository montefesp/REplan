from os.path import join
import pickle
from typing import List, Dict, Any

import pandas as pd

import pypsa

from pyggrid.data.geographics import match_points_to_regions, match_points_to_countries, get_shapes,\
    remove_landlocked_countries, get_area_per_site
from pyggrid.data.generation.vres.legacy import get_legacy_capacity_in_regions, get_legacy_capacity_in_countries
# from pyggrid.data.potentials import get_capacity_potential_at_points
from pyggrid.data.generation.vres.potentials.enspreso import get_capacity_potential_for_countries,\
    get_capacity_potential_for_regions
from pyggrid.data.generation.vres.potentials.glaes import get_capacity_potential_for_shapes
from pyggrid.data.generation.vres.profiles import compute_capacity_factors, get_cap_factor_for_countries
from pyggrid.data.technologies import get_costs, get_config_values, get_config_dict
from pyggrid.resite.resite import Resite

from pyggrid.data import data_path

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators_from_file(net: pypsa.Network, topology_type: str, technologies: List[str],
                             sites_dir: str, sites_fn: str, spatial_resolution: float,
                             power_density: float) -> pypsa.Network:
    """
    Add wind and PV generators based on sites that where selected via a certain siting method to a Network class.

    Parameters
    ----------
    net: pypsa.Network
        A Network instance with regions
    topology_type: str
        Can currently be countries (for one node per country topologies)
        or regions (for topologies based on arbitrary regions)
    technologies: List[str]
        Which technologies we want to add
    sites_dir: str
        Relative to directory where sites files are kept.
    sites_fn: str
        Name of file containing sites.
    spatial_resolution: float
        Spatial resolution at which the points are defined.
    power_density: float
        Power density of a given technology.

    Returns
    -------
    net: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "regions"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    assert topology_type == "countries" or "region" in net.buses.columns, \
        "Error: If you are not using a country-based topology, you must associate regions to buses."

    # assert not use_default_capacity or area_per_site is not None, \
    #     "Error: area_per_site must be defined if use_default_capacity is True."

    # Get countries over which the network is defined
    countries = list(net.buses.country.dropna())

    # Load site data
    resite_data_path = f"{data_path}resite/generated/{sites_dir}/"
    resite_data_fn = join(resite_data_path, sites_fn)
    tech_points_cap_factor_df = pickle.load(open(resite_data_fn, "rb"))

    missing_timestamps = set(net.snapshots) - set(tech_points_cap_factor_df.index)
    assert not missing_timestamps, f"Error: Following timestamps are not part of capacity factors {missing_timestamps}"

    # Determine if the network contains offshore buses
    offshore_buses = True if hasattr(net.buses, 'onshore') and sum(~net.buses.onshore) != 0 else False

    for tech in technologies:

        if tech not in set(tech_points_cap_factor_df.columns.get_level_values(0)):
            print(f"Warning: Technology {tech} is not part of RES data from files. Therefore it was not added.")
            continue

        points = sorted(list(tech_points_cap_factor_df[tech].columns), key=lambda x: x[0])

        # If there are no offshore buses, add all generators to onshore buses
        # If there are offshore buses, add onshore techs to onshore buses and offshore techs to offshore buses
        buses = net.buses.copy()
        if offshore_buses:
            is_onshore = False if tech in ['wind_offshore', 'wind_floating'] else True
            buses = buses[buses.onshore == is_onshore]

        # Detect to which bus each site should be associated
        if not offshore_buses and topology_type == "countries":
            points_bus_ds = match_points_to_countries(points, countries).dropna()
        else:
            points_bus_ds = match_points_to_regions(points, buses.region).dropna()
        points = list(points_bus_ds.index)

        logger.info(f"Adding {tech} in {list(set(points_bus_ds))}.")

        # Compute capacity potential
        # TODO: what do we do with the commented section below?
        # if not use_default_capacity:
        #
        #     # Compute capacity potential for all initial points in region and then keep the one only for selected
        #     sites
        #     init_points_fn = join(resite_data_path, "init_coordinates_dict.p")
        #     init_points_list = pickle.load(open(init_points_fn, "rb"))[tech]
        #
        #     # ODO: some weird behaviour happening for offshore, duplicate locations occurring.
        #     #  To be further checked, ideally this filtering disappears..
        #     init_points_list = list(set(init_points_list))
        #
        #     capacity_potential_per_node_full = \
        #         get_capacity_potential_at_points({tech: init_points_list}, spatial_resolution, countries)[tech]
        #     points_capacity_potential = list(capacity_potential_per_node_full.loc[points].values)
        #
        # else:
        #
        #     # Use predefined per km capacity multiplied by grid cell area.
        #     bus_capacity_potential_per_km = pd.Series(get_config_values(tech, ['power_density']), index=buses.index)
        #     points_capacity_potential = \
        #         [bus_capacity_potential_per_km[points_bus_ds[point]] *
        #          get_area_per_site(point, spatial_resolution) / 1e3 for point in points]

        # Use predefined per km capacity multiplied by grid cell area.
        bus_capacity_potential_per_km = pd.Series(power_density, index=buses.index)
        points_capacity_potential = \
            [bus_capacity_potential_per_km[points_bus_ds[point]] *
             get_area_per_site(point, spatial_resolution) / 1e3 for point in points]

        # Get capacity factors
        cap_factor_series = tech_points_cap_factor_df.loc[net.snapshots][tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

        net.madd("Generator",
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

    return net


def add_generators_using_siting(net: pypsa.Network, topology_type: str, technologies: List[str],
                                region: str, siting_params: Dict[str, Any],
                                use_ex_cap: bool = True, limit_max_cap: bool = True,
                                output_dir: str = None) -> pypsa.Network:
    """
    Add generators for different technologies at a series of location selected via an optimization mechanism.

    Parameters
    ----------
    net: pypsa.Network
        A network with defined buses.
    topology_type: str
        Can currently be countries (for one node per country topologies)
        or regions (for topologies based on arbitrary regions)
    technologies: List[str]
        Which technologies to add using this methodology
    siting_params: Dict[str, Any]
        Set of parameters necessary for siting.
    region: str
        Region over which the network is defined
    use_ex_cap: bool (default: True)
        Whether to take into account existing capacity.
    limit_max_cap: bool (default: True)
        Whether to limit capacity expansion at each grid cell to a certain capacity potential.
    output_dir: str
        Absolute path to directory where resite output should be stored

    Returns
    -------
    net: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "regions"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    assert topology_type == "countries" or "region" in net.buses.columns, \
        "Error: If you are using a region-based topology, you must associate regions to buses."

    assert topology_type == "regions" or "country" in net.buses.columns, \
        "Error: If you are using a country-based topology, you must associate ISO codes to buses."

    for param in ["timeslice", "spatial_resolution", "modelling", "formulation", "formulation_params", "write_lp"]:
        assert param in siting_params, f"Error: Missing parameter {param} for siting."

    logger.info('Setting up resite.')
    resite = Resite([region], technologies, siting_params["timeslice"], siting_params["spatial_resolution"],
                    siting_params["min_cap_if_selected"])
    resite.build_data(use_ex_cap)

    logger.info('resite model being built.')
    resite.build_model(siting_params["modelling"], siting_params['formulation'], siting_params['formulation_params'],
                       siting_params['write_lp'], output_dir)

    logger.info('Sending resite to solver.')
    resite.solve_model(solver_options=siting_params['solver_options'], solver=siting_params['solver'])

    logger.info('Retrieving resite results.')
    resite.retrieve_selected_sites_data()
    tech_location_dict = resite.sel_tech_points_dict
    existing_cap_ds = resite.sel_data_dict["existing_cap_ds"]
    cap_potential_ds = resite.sel_data_dict["cap_potential_ds"]
    cap_factor_df = resite.sel_data_dict["cap_factor_df"]

    logger.info("Saving resite results")
    resite.save(output_dir)

    if not resite.timestamps.equals(net.snapshots):
        # If network snapshots is a subset of resite snapshots just crop the data
        missing_timestamps = set(net.snapshots) - set(resite.timestamps)
        if not missing_timestamps:
            cap_factor_df = cap_factor_df.loc[net.snapshots]
        else:
            # In other case, need to recompute capacity factors
            raise NotImplementedError("Error: Network snapshots must currently be a subset of resite snapshots.")

    # Determine if the network contains offshore buses
    has_offshore_buses = True if hasattr(net.buses, 'onshore') and sum(~net.buses.onshore) != 0 else False

    for tech, points in tech_location_dict.items():

        onshore_tech = get_config_values(tech, ['onshore'])
        if topology_type == "regions" and not has_offshore_buses and not onshore_tech:
            raise ValueError(f"Offshore-based technology {tech} can only be added to region-based topology if"
                             f" offshore buses are defined.")

        # Associate sites to buses
        buses = net.buses.copy()
        if topology_type == "countries":
            countries = list(buses.country.dropna())
            associated_buses = match_points_to_countries(points, countries).dropna()
            associated_buses = associated_buses.apply(lambda c: net.buses[net.buses.country == c].index[0])
        else:  # topology_type == "regions"
            buses = buses[buses.onshore == onshore_tech]
            associated_buses = match_points_to_regions(points, buses.region).dropna()
        points = list(associated_buses.index)

        p_nom_max = 'inf'
        if limit_max_cap:
            p_nom_max = cap_potential_ds[tech][points].values
        p_nom = existing_cap_ds[tech][points].values
        p_max_pu = cap_factor_df[tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

        net.madd("Generator",
                 pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                 bus=associated_buses.values,
                 p_nom_extendable=True,
                 p_nom_max=p_nom_max,
                 p_nom=p_nom,
                 p_nom_min=p_nom,
                 p_min_pu=0.,
                 p_max_pu=p_max_pu,
                 type=tech,
                 x=[x for x, _ in points],
                 y=[y for _, y in points],
                 marginal_cost=marginal_cost,
                 capital_cost=capital_cost)

    return net


def add_generators_in_grid_cells(net: pypsa.Network, topology_type: str, technologies: List[str],
                                 region: str, spatial_resolution: float,
                                 use_ex_cap: bool = True, limit_max_cap: bool = True,
                                 min_cap_pot: List[float] = None) -> pypsa.Network:
    """
    Create VRES generators in every grid cells obtained from dividing a certain number of regions.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    topology_type: str
        Can currently be countries (for one node per country topologies)
        or regions (for topologies based on arbitrary regions)
    technologies: List[str]
        Which technologies to add.
    region: str
        Region code defined in 'data_path'/geographics/region_definition.csv over which the network is defined.
    spatial_resolution: float
        Spatial resolution at which to define grid cells.
    use_ex_cap: bool (default: True)
        Whether to take into account existing capacity.
    limit_max_cap: bool (default: True)
        Whether to limit capacity expansion at each grid cell to a certain capacity potential.
    min_cap_pot: List[float] (default: None)
        List of thresholds per technology. Points with capacity potential under this threshold will be removed.


    Returns
    -------
    net: pypsa.Network
        Updated network

    Notes
    -----
    Each bus must contain 'region' and 'country' attributes.


    If the network does not contain any offshore buses, but one of the technology to be added is offshore-based,
    the behavior of this function differs whether the topology is 'country' or 'region' based.

    - For country-based topology, the corresponding offshore generators will be
        associated to the onshore bus for country that have offshore territories.

    - For region-based topology, as it is not possible to assign offshore
        territories to onshore buses, an error will be raised.
    """

    accepted_topologies = ["countries", "regions"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    assert topology_type == "countries" or "region" in net.buses.columns, \
        "Error: If you are using a region-based topology, you must associate regions to buses."

    assert topology_type == "regions" or "country" in net.buses.columns, \
        "Error: If you are using a country-based topology, you must associate ISO codes to buses."

    # Determine if the network contains offshore buses
    has_offshore_buses = True if hasattr(net.buses, 'onshore') and sum(~net.buses.onshore) != 0 else False

    # Generate deployment sites using resite
    resite = Resite([region], technologies, [net.snapshots[0], net.snapshots[-1]], spatial_resolution)
    resite.build_data(use_ex_cap, min_cap_pot)

    for tech in technologies:

        points = resite.tech_points_dict[tech]

        onshore_tech = get_config_values(tech, ['onshore'])
        if topology_type == "regions" and not has_offshore_buses and not onshore_tech:
            raise ValueError(f"Offshore-based technology {tech} can only be added to region-based topology if"
                             f" offshore buses are defined.")

        # Associate sites to buses
        buses = net.buses.copy()
        if topology_type == "countries":
            countries = list(buses.country.dropna())
            associated_buses = match_points_to_countries(points, countries).dropna()
            associated_buses = associated_buses.apply(lambda c: net.buses[net.buses.country == c].index[0])
        else:  # topology_type == "regions"
            buses = buses[buses.onshore == onshore_tech]
            associated_buses = match_points_to_regions(points, buses.region).dropna()
        points = list(associated_buses.index)

        p_nom_max = 'inf'
        if limit_max_cap:
            p_nom_max = resite.data_dict["cap_potential_ds"][tech][points].values
        p_nom = resite.data_dict["existing_cap_ds"][tech][points].values
        p_max_pu = resite.data_dict["cap_factor_df"][tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

        net.madd("Generator",
                 pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                 bus=associated_buses.values,
                 p_nom_extendable=True,
                 p_nom_max=p_nom_max,
                 p_nom=p_nom,
                 p_nom_min=p_nom,
                 p_min_pu=0.,
                 p_max_pu=p_max_pu,
                 type=tech,
                 x=[x for x, _ in points],
                 y=[y for _, y in points],
                 marginal_cost=marginal_cost,
                 capital_cost=capital_cost)

    return net


def add_generators_per_bus(net: pypsa.Network, topology_type: str,
                           technologies: List[str], use_ex_cap: bool = True,
                           bus_ids: List[str] = None) -> pypsa.Network:
    """
    Add VRES generators to each bus of a PyPSA Network, each bus being associated to a geographical region.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies)
        or regions (for topologies based on arbitrary regions).
    technologies: List[str]
        Names of VRES technologies to be added.
    use_ex_cap: bool (default: True)
        Whether to take into account existing capacity.
    bus_ids: List[str]
        Subset of buses to which the generators must be added.

    Returns
    -------
    net: pypsa.Network
        Updated network

    Notes
    -----
    Each bus must contain 'x', 'y', 'region' and 'country' attributes.


    If the network does not contain any offshore buses, but one of the technology to be added is offshore-based,
    the behavior of this function differs whether the topology is 'country' or 'region' based.

    - For country-based topology, the corresponding offshore generators will be
        associated to the onshore bus for country that have offshore territories.

    - For region-based topology, as it is not possible to assign offshore
        territories to onshore buses, an error will be raised.

    """

    accepted_topologies = ["countries", "regions"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    # Filter out buses
    # TODO: to be tested
    all_buses = net.buses.copy()
    if bus_ids is not None:
        all_buses = all_buses.loc[bus_ids]

    for attr in ["x", "y", "region", "country"]:
        assert hasattr(all_buses, attr), f"Error: Buses must contain a '{attr}' attribute."

    # Determine if the network contains offshore buses
    has_offshore_buses = True if (hasattr(all_buses, 'onshore') and sum(~all_buses.onshore) != 0) else False

    tech_config_dict = get_config_dict(technologies, ["filters", "power_density", "onshore"])
    for tech in technologies:

        # Detect if technology is onshore(/offshore) based
        onshore_tech = tech_config_dict[tech]["onshore"]
        if topology_type == "regions" and not has_offshore_buses and not onshore_tech:
            raise ValueError(f"Offshore-based technology {tech} can only be added to region-based topology if"
                             f" offshore buses are defined.")

        # If there are only onshore buses, we add all technologies (including offshore-based ones) to those buses.
        buses = all_buses.copy()
        # If there are offshore buses, we add onshore-based technologies to onshore buses and
        # offshore-based technologies to offshore buses.
        if has_offshore_buses:
            buses = buses[buses.onshore == onshore_tech]

        # Get countries over which the buses are defined.
        countries = list(set(buses.country.dropna()))
        # If no offshore buses and adding an offshore tech, remove countries which are landlocked and associated buses
        if not has_offshore_buses and not onshore_tech:
            countries = remove_landlocked_countries(countries)
            buses = buses[buses.country.isin(countries)]

        # Get the shapes of regions associated to each bus
        # For offshore technologies, if we don't have any offshore buses, compute the offshore shapes
        # (note: this can only happen in the case of a country-based topology)
        if not has_offshore_buses and not onshore_tech:
            countries_shapes = get_shapes(countries, which='offshore', save=True)["geometry"]
            buses_regions_shapes_ds = pd.Series(index=buses.index)
            buses_regions_shapes_ds[:] = countries_shapes.loc[buses.country]
        # For all other cases, we get the shapes directly associated to the buses.
        else:
            buses_regions_shapes_ds = buses.region

        # Compute capacity potential at each bus
        # TODO: WARNING: first part of if-else to be removed
        enspreso = False
        if enspreso:
            logger.warning("Capacity potentials computed using ENSPRESO data.")
            if topology_type == "countries" and len(countries) != 0:
                cap_pot_country_ds = get_capacity_potential_for_countries(tech, countries)
                cap_pot_ds = pd.Series(index=buses.index)
                cap_pot_ds[:] = cap_pot_country_ds.loc[buses.country]
            else:  # topology_type == "regions"
                cap_pot_ds = get_capacity_potential_for_regions({tech: buses_regions_shapes_ds.values})[tech]
                cap_pot_ds.index = buses.index
        else:
            # Using GLAES
            filters = tech_config_dict[tech]["filters"]
            power_density = tech_config_dict[tech]["power_density"]
            cap_pot_ds = pd.Series(index=buses.index)
            cap_pot_ds[:] = get_capacity_potential_for_shapes(buses_regions_shapes_ds.values, filters, power_density)

        # Get one capacity factor time series per bus
        if topology_type == 'countries' and len(countries) != 0:
            # For country-based topologies, use aggregated series obtained from Renewables.ninja
            cap_factor_countries_df = get_cap_factor_for_countries(tech, countries, net.snapshots, False)
            cap_factor_df = pd.DataFrame(index=net.snapshots, columns=buses.index)
            cap_factor_df[:] = cap_factor_countries_df[buses.country]
        else:
            # For region-based topology, compute capacity factors at (rounded) buses position
            spatial_res = 0.5
            if not has_offshore_buses and not onshore_tech:
                points = [(round(shape.centroid.x/spatial_res) * spatial_res,
                           round(shape.centroid.y/spatial_res) * spatial_res)
                          for shape in buses_regions_shapes_ds.values]
            else:
                points = [(round(x/spatial_res)*spatial_res,
                           round(y/spatial_res)*spatial_res)
                          for x, y in buses[["x", "y"]].values]
            cap_factor_df = compute_capacity_factors({tech: points}, spatial_res, net.snapshots)[tech]
            cap_factor_df.columns = buses.index

        # Compute legacy capacity (not available for wind_floating)
        legacy_cap_ds = pd.Series(0., index=buses.index)
        if use_ex_cap and tech != "wind_floating":
            if topology_type == 'countries' and len(countries) != 0:
                legacy_cap_countries = get_legacy_capacity_in_countries(tech, countries)
                legacy_cap_ds[:] = legacy_cap_countries.loc[buses.country]
            else:
                legacy_cap_ds = get_legacy_capacity_in_regions(tech, buses_regions_shapes_ds, countries)

        # Update capacity potentials if legacy capacity is bigger
        for bus in buses.index:
            if cap_pot_ds.loc[bus] < legacy_cap_ds.loc[bus]:
                cap_pot_ds.loc[bus] = legacy_cap_ds.loc[bus]

        # Get costs
        capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

        # Adding to the network
        net.madd("Generator",
                 buses.index,
                 suffix=f" Gen {tech}",
                 bus=buses.index,
                 p_nom_extendable=True,
                 p_nom=legacy_cap_ds,
                 p_nom_min=legacy_cap_ds,
                 p_nom_max=cap_pot_ds,
                 p_min_pu=0.,
                 p_max_pu=cap_factor_df,
                 type=tech,
                 x=buses.x.values,
                 y=buses.y.values,
                 marginal_cost=marginal_cost,
                 capital_cost=capital_cost)

    return net
