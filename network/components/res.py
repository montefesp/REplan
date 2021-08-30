from typing import List, Dict, Any
from os.path import join
import pickle

import pandas as pd

import pypsa

from iepy import data_path
from iepy.geographics import match_points_to_regions_eez, match_points_to_regions, get_offshore_shapes
from iepy.geographics.codes import replace_iso2_codes, revert_iso2_codes
from iepy.generation.vres.legacy import get_legacy_capacity_in_regions, get_legacy_capacity_in_countries
# from iepy.potentials import get_capacity_potential_at_points
from iepy.generation.vres.potentials.enspreso import get_capacity_potential_for_countries,\
    get_capacity_potential_for_regions
from iepy.generation.vres.potentials.glaes import get_capacity_potential_for_shapes
from iepy.generation.vres.profiles import compute_capacity_factors, get_cap_factor_for_countries
from iepy.technologies import get_costs, get_config_values, get_config_dict
from iepy.geographics import get_area_per_site
from iepy.generation.vres.legacy import get_legacy_capacity_at_points


import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators_from_file(net: pypsa.Network, technologies: List[str], use_ex_cap: bool,
                             sites_dir: str, sites_fn: str, spatial_resolution: float,
                             tech_config: dict) -> pypsa.Network:
    """
    Add wind and PV generators based on sites that where selected via a certain siting method to a Network class.
    Parameters
    ----------
    net: pypsa.Network
        A Network instance with regions
    technologies: List[str]
        Which technologies we want to add
    use_ex_cap: bool
        Whether to use legacy capacity or not.
    sites_dir: str
        Relative to directory where sites files are kept.
    sites_fn: str
        Name of file containing sites.
    spatial_resolution: float
        Spatial resolution at which the points are defined.
    tech_config: dict
    Returns
    -------
    net: pypsa.Network
        Updated network
    """

    # Load site data
    resite_data_path = f"{data_path}../../resite_ip/output/{sites_dir}/"
    resite_data_fn = join(resite_data_path, sites_fn)
    tech_points_cap_factor_df = pickle.load(open(resite_data_fn, "rb"))

    eez_shapes = get_offshore_shapes(replace_iso2_codes(list(net.buses.index.str[:2].unique())))
    eez_shapes.index = revert_iso2_codes(eez_shapes.index)

    tech_points_cap_factor_df.index = pd.to_datetime(tech_points_cap_factor_df.index)
    sampling_rate = tech_points_cap_factor_df.index.hour[1] - tech_points_cap_factor_df.index.hour[0]
    if sampling_rate != 1:
        tech_points_cap_factor_df = tech_points_cap_factor_df.reindex(net.snapshots, method='ffill')

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
        # if not offshore_buses:
        #     points_bus_ds = match_points_to_countries(points, countries).dropna()
        # else:
        if tech in ['wind_offshore', 'wind_floating']:
            points_bus_ds = match_points_to_regions_eez(points, buses['onshore_region'], eez_shapes,
                                                        distance_threshold=1e4).dropna()
        else:
            points_bus_ds = match_points_to_regions(points, buses['onshore_region']).dropna()
        points = list(points_bus_ds.index)

        logger.info(f"Adding {tech} in {list(set(points_bus_ds))}.")

        # Use predefined per km capacity multiplied by grid cell area.
        bus_capacity_potential_per_km = pd.Series(tech_config[tech]['power_density'], index=buses.index)
        points_capacity_potential = \
            [bus_capacity_potential_per_km[points_bus_ds[point]] *
             get_area_per_site(point, spatial_resolution) / 1e3 for point in points]
        points_capacity_potential_ds = pd.Series(data=points_capacity_potential,
                                                 index=pd.MultiIndex.from_tuples(points)).round(3)

        # Get capacity factors
        cap_factor_series = tech_points_cap_factor_df.loc[net.snapshots][tech][points]

        # Compute legacy capacity
        legacy_cap_ds = pd.Series(0., index=points, dtype=float)
        if use_ex_cap and tech != "wind_floating":
            legacy_cap_ds = get_legacy_capacity_at_points(tech, points)

        # Update capacity potentials if legacy capacity is bigger
        for p in points_capacity_potential_ds.index:
            if points_capacity_potential_ds.loc[p] < legacy_cap_ds.loc[p]:
                points_capacity_potential_ds.loc[p] = legacy_cap_ds.loc[p]

        capital_cost, marginal_cost, _ = get_costs(tech, len(net.snapshots))

        net.madd("Generator",
                 pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                 bus=points_bus_ds.values,
                 p_nom_extendable=True,
                 p_nom=legacy_cap_ds.values,
                 p_nom_max=points_capacity_potential_ds.values,
                 p_nom_min=legacy_cap_ds.values,
                 p_min_pu=0.,
                 p_max_pu=cap_factor_series.values,
                 type=tech,
                 x=[x for x, _ in points],
                 y=[y for _, y in points],
                 marginal_cost=marginal_cost,
                 capital_cost=capital_cost)

    return net


def add_generators_using_siting(net: pypsa.Network, technologies: List[str],
                                region: str, siting_params: Dict[str, Any],
                                use_ex_cap: bool = True, limit_max_cap: bool = True,
                                output_dir: str = None) -> pypsa.Network:
    """
    Add generators for different technologies at a series of location selected via an optimization mechanism.

    Parameters
    ----------
    net: pypsa.Network
        A network with defined buses.
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

    Notes
    -----
    net.buses must have a 'region_onshore' if adding onshore technologies and a 'region_offshore' attribute
    if adding offshore technologies.
    """

    for param in ["timeslice", "spatial_resolution", "modelling", "formulation", "formulation_params", "write_lp"]:
        assert param in siting_params, f"Error: Missing parameter {param} for siting."

    from resite.resite import Resite

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

    for tech, points in tech_location_dict.items():

        onshore_tech = get_config_values(tech, ['onshore'])

        # Associate sites to buses (using the associated shapes)
        buses = net.buses.copy()
        region_type = 'onshore_region' if onshore_tech else 'offshore_region'
        buses = buses.dropna(subset=[region_type])
        associated_buses = match_points_to_regions(points, buses[region_type]).dropna()
        points = list(associated_buses.index)

        p_nom_max = 'inf'
        if limit_max_cap:
            p_nom_max = cap_potential_ds[tech][points].values
        p_nom = existing_cap_ds[tech][points].values
        p_max_pu = cap_factor_df[tech][points].values

        capital_cost, marginal_cost, _ = get_costs(tech, len(net.snapshots))

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


def add_generators_in_grid_cells(net: pypsa.Network, technologies: List[str],
                                 region: str, spatial_resolution: float,
                                 use_ex_cap: bool = True, limit_max_cap: bool = True,
                                 min_cap_pot: List[float] = None) -> pypsa.Network:
    """
    Create VRES generators in every grid cells obtained from dividing a certain number of regions.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
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
    net.buses must have a 'region_onshore' if adding onshore technologies and a 'region_offshore' attribute
    if adding offshore technologies.
    """

    from resite.resite import Resite

    # Generate deployment sites using resite
    resite = Resite([region], technologies, [net.snapshots[0], net.snapshots[-1]], spatial_resolution)
    resite.build_data(use_ex_cap, min_cap_pot)

    for tech in technologies:

        points = resite.tech_points_dict[tech]
        onshore_tech = get_config_values(tech, ['onshore'])

        # Associate sites to buses (using the associated shapes)
        buses = net.buses.copy()
        region_type = 'onshore_region' if onshore_tech else 'offshore_region'
        buses = buses.dropna(subset=[region_type])
        associated_buses = match_points_to_regions(points, buses[region_type]).dropna()
        points = list(associated_buses.index)

        p_nom_max = 'inf'
        if limit_max_cap:
            p_nom_max = resite.data_dict["cap_potential_ds"][tech][points].values
        p_nom = resite.data_dict["existing_cap_ds"][tech][points].values
        p_max_pu = resite.data_dict["cap_factor_df"][tech][points].values

        capital_cost, marginal_cost, _ = get_costs(tech, len(net.snapshots))

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


def add_generators_per_bus(net: pypsa.Network, technologies: List[str],
                           use_ex_cap: bool = True, bus_ids: List[str] = None,
                           precision: int = 3) -> pypsa.Network:
    """
    Add VRES generators to each bus of a PyPSA Network, each bus being associated to a geographical region.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions.
    technologies: List[str]
        Names of VRES technologies to be added.
    use_ex_cap: bool (default: True)
        Whether to take into account existing capacity.
    bus_ids: List[str]
        Subset of buses to which the generators must be added.
    precision: int (default: 3)
        Indicates at which decimal values should be rounded

    Returns
    -------
    net: pypsa.Network
        Updated network

    Notes
    -----
    Each bus must contain 'x', 'y' attributes.
    In addition, each bus must have a 'region_onshore' and/or 'region_offshore' attributes.
    Finally, if the topology has one bus per country (and no offshore buses), all buses can be associated
    to an ISO code under the attribute 'country' to fasten some computations.

    """

    # Filter out buses
    all_buses = net.buses.copy()
    all_buses = all_buses[all_buses['onshore_region'].notna()]
    if bus_ids is not None:
        all_buses = all_buses.loc[bus_ids]

    for attr in ["x", "y"]:
        assert hasattr(all_buses, attr), f"Error: Buses must contain a '{attr}' attribute."
    # assert all([len(bus[["onshore_region", "offshore_region"]].dropna()) != 0 for idx, bus in all_buses.iterrows()]), \
    #     "Error: Each bus must be associated to an 'onshore_region' and/or 'offshore_region' attribute."

    one_bus_per_country = False
    # if hasattr(all_buses, 'country'):
    #     # Check every bus has a value for this attribute
    #     complete = len(all_buses["country"].dropna()) == len(all_buses)
    #     # Check the values are unique
    #     unique = len(all_buses["country"].unique()) == len(all_buses)
    #     one_bus_per_country = complete & unique

    tech_config_dict = get_config_dict(technologies, ["filters", "power_density", "onshore"])
    for tech in technologies:

        # Detect if technology is onshore(/offshore) based
        onshore_tech = tech_config_dict[tech]["onshore"]

        # Get buses which are associated to an onshore/offshore region
        region_type = "onshore_region" if onshore_tech else 'offshore_region'
        buses = all_buses.dropna(subset=[region_type], axis=0)
        countries = list(buses.index.str[:2].unique())

        # Get the shapes of regions associated to each bus
        buses_regions_shapes_ds = buses[region_type]

        # Compute capacity potential at each bus
        logger.warning("Capacity potentials computed using ENSPRESO data.")
        if one_bus_per_country:
            cap_pot_country_ds = get_capacity_potential_for_countries(tech, countries)
            cap_pot_ds = pd.Series(index=buses.index)
            cap_pot_ds[:] = cap_pot_country_ds.loc[buses.country]
        else:  # topology_type == "regions"
            cap_pot_ds = get_capacity_potential_for_regions(list(buses.index), {tech: buses_regions_shapes_ds.values})[tech]
            cap_pot_ds.index = buses.index

        # Get one capacity factor time series per bus
        if one_bus_per_country:
            # For country-based topologies, use aggregated series obtained from Renewables.ninja
            cap_factor_countries_df = get_cap_factor_for_countries(tech, countries, net.snapshots, precision, False)
            cap_factor_df = pd.DataFrame(index=net.snapshots, columns=buses.index)
            cap_factor_df[:] = cap_factor_countries_df[buses.country]
        else:
            # For region-based topology, compute capacity factors at (rounded) buses position
            spatial_res = 0.25
            points = [(round(shape.centroid.x/spatial_res) * spatial_res,
                       round(shape.centroid.y/spatial_res) * spatial_res)
                      for shape in buses_regions_shapes_ds.values]
            cap_factor_df = compute_capacity_factors({tech: points}, spatial_res, net.snapshots, precision)[tech]
            cap_factor_df.columns = buses.index

        # Compute legacy capacity (not available for wind_floating)
        legacy_cap_ds = pd.Series(0., index=buses.index)
        if use_ex_cap and tech != "wind_floating":
            if one_bus_per_country and len(countries) != 0:
                legacy_cap_countries = get_legacy_capacity_in_countries(tech, countries)
                legacy_cap_ds[:] = legacy_cap_countries.loc[buses.country]
            else:
                legacy_cap_ds = get_legacy_capacity_in_regions(tech, buses_regions_shapes_ds, countries)

        # Update capacity potentials if legacy capacity is bigger
        for bus in buses.index:
            if cap_pot_ds.loc[bus] < legacy_cap_ds.loc[bus]:
                cap_pot_ds.loc[bus] = legacy_cap_ds.loc[bus]

        # Remove generators if capacity potential is 0
        non_zero_potential_gens_index = cap_pot_ds[cap_pot_ds > 0].index
        cap_pot_ds = cap_pot_ds.loc[non_zero_potential_gens_index]
        legacy_cap_ds = legacy_cap_ds.loc[non_zero_potential_gens_index]
        cap_factor_df = cap_factor_df[non_zero_potential_gens_index]
        buses = buses.loc[non_zero_potential_gens_index]

        # Get costs
        capital_cost, marginal_cost, _ = get_costs(tech, len(net.snapshots))

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
