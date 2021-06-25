from os.path import join
import pickle
from typing import List

import pandas as pd

import pypsa

from iepy import data_path

from iepy.geographics import match_points_to_regions, get_area_per_site, match_points_to_countries
from iepy.generation.vres.legacy import get_legacy_capacity_at_points, get_legacy_capacity_in_countries
from iepy.generation.vres.potentials.enspreso import get_capacity_potential_for_countries,\
    get_capacity_potential_for_regions
from iepy.generation.vres.profiles import compute_capacity_factors, get_cap_factor_for_countries
from iepy.technologies import get_costs, get_config_dict


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

    # Get countries over which the network is defined
    countries = list(net.buses.country.dropna())

    # Load site data
    resite_data_path = f"{data_path}../../resite_ip/output/{sites_dir}/"
    resite_data_fn = join(resite_data_path, sites_fn)
    tech_points_cap_factor_df = pickle.load(open(resite_data_fn, "rb"))

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
        if not offshore_buses:
            points_bus_ds = match_points_to_countries(points, countries).dropna()
        else:
            points_bus_ds = match_points_to_regions(points, buses.region).dropna()
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

        capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

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


def add_generators_per_bus(net: pypsa.Network, technologies: List[str],
                           use_ex_cap: bool = True, extendable: bool = True,
                           bus_ids: List[str] = None) -> pypsa.Network:
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
    all_buses = all_buses[all_buses['country'].notna()]
    if bus_ids is not None:
        all_buses = all_buses.loc[bus_ids]

    for attr in ["x", "y"]:
        assert hasattr(all_buses, attr), f"Error: Buses must contain a '{attr}' attribute."
    assert all([len(bus[["onshore_region", "offshore_region"]].dropna()) != 0 for idx, bus in all_buses.iterrows()]), \
        "Error: Each bus must be associated to an 'onshore_region' and/or 'offshore_region' attribute."

    one_bus_per_country = False
    if hasattr(all_buses, 'country'):
        # Check every bus has a value for this attribute
        complete = len(all_buses["country"].dropna()) == len(all_buses)
        # Check the values are unique
        unique = len(all_buses["country"].unique()) == len(all_buses)
        one_bus_per_country = complete & unique

    tech_config_dict = get_config_dict(technologies, ["filters", "power_density", "onshore"])

    for tech in technologies:

        # Detect if technology is onshore(/offshore) based
        onshore_tech = tech_config_dict[tech]["onshore"]

        # Get buses which are associated to an onshore/offshore region
        region_type = "onshore_region" if onshore_tech else 'offshore_region'
        buses = all_buses.dropna(subset=[region_type], axis=0)
        countries = list(buses["country"].unique())

        # Get the shapes of regions associated to each bus
        buses_regions_shapes_ds = buses[region_type]

        # Compute capacity potential at each bus
        logger.warning(f"Capacity potentials of {tech} computed using ENSPRESO data.")
        if one_bus_per_country:
            cap_pot_country_ds = get_capacity_potential_for_countries(tech, countries)
            cap_pot_ds = pd.Series(index=buses.index)
            cap_pot_ds[:] = cap_pot_country_ds.loc[buses.country]
        else:  # topology_type == "regions"
            cap_pot_ds = get_capacity_potential_for_regions({tech: buses_regions_shapes_ds.values})[tech]
            cap_pot_ds.index = buses.index

        # Get one capacity factor time series per bus
        if one_bus_per_country:
            # For country-based topologies, use aggregated series obtained from Renewables.ninja
            cap_factor_countries_df = get_cap_factor_for_countries(tech, countries, net.snapshots, False)
            cap_factor_df = pd.DataFrame(index=net.snapshots, columns=buses.index)
            cap_factor_df[:] = cap_factor_countries_df[buses.country]
        else:
            # For region-based topology, compute capacity factors at (rounded) buses position
            spatial_res = 0.5
            points = [(round(shape.centroid.x/spatial_res) * spatial_res,
                       round(shape.centroid.y/spatial_res) * spatial_res)
                      for shape in buses_regions_shapes_ds.values]
            cap_factor_df = compute_capacity_factors({tech: points}, spatial_res, net.snapshots)[tech]
            cap_factor_df.columns = buses.index

        # Compute legacy capacity (not available for wind_floating)
        legacy_cap_ds = pd.Series(0., index=buses.index)
        if use_ex_cap and tech != "wind_floating":
            legacy_cap_countries = get_legacy_capacity_in_countries(tech, countries)
            legacy_cap_ds[:] = legacy_cap_countries.loc[buses.country]

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
                 p_nom_extendable=extendable,
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
