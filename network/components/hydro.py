import logging

import pypsa

from iepy.generation.hydro import *
from iepy.technologies import get_costs, get_tech_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_phs_plants(net: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """
    Add pumped-hydro storage units to a PyPSA Network instance.

    Parameters
    ----------
    net: pypsa.Network
        A Network instance.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    net: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    comp_attrs = ["x", "y", "onshore_region"]
    comp_attrs += ["country"] if topology_type == "countries" else []
    for attr in comp_attrs:
        assert hasattr(net.buses, attr), f"Error: Buses must contain a '{attr}' attribute."

    # Hydro generators can only be added onshore
    # buses_onshore = net.buses[net.buses.onshore]
    buses_onshore = net.buses.dropna(subset=["onshore_region"], axis=0)

    # Load capacities
    aggr_level = "countries" if topology_type == "countries" else "NUTS3"
    pow_cap, en_cap = get_phs_capacities(aggr_level)

    if topology_type == 'countries':
        # Extract only countries for which data is available
        countries_with_capacity = sorted(list(set(buses_onshore.country) & set(pow_cap.index)))
        buses_with_capacity_indexes = net.buses[net.buses.country.isin(countries_with_capacity)].index
        bus_pow_cap = pow_cap.loc[countries_with_capacity]
        bus_pow_cap.index = buses_with_capacity_indexes
        bus_en_cap = en_cap.loc[countries_with_capacity]
        bus_en_cap.index = buses_with_capacity_indexes
    else:  # topology_type == 'ehighway
        bus_pow_cap, bus_en_cap = phs_inputs_nuts_to_ehighway(buses_onshore.index, pow_cap, en_cap)
        countries_with_capacity = set(bus_pow_cap.index.str[2:])

    logger.info(f"Adding {bus_pow_cap.sum():.3f} GW of PHS hydro "
                f"with {bus_en_cap.sum():.3f} GWh of storage in {countries_with_capacity}.")

    max_hours = bus_en_cap / bus_pow_cap

    # Get cost and efficiencies
    capital_cost, marginal_cost = get_costs('phs', len(net.snapshots))
    efficiency_dispatch, efficiency_store, self_discharge = \
        get_tech_info('phs', ["efficiency_ds", "efficiency_ch", "efficiency_sd"])
    self_discharge = round(1 - self_discharge, 4)

    net.madd("StorageUnit",
             bus_pow_cap.index,
             suffix=" Storage PHS",
             bus=bus_pow_cap.index,
             type='phs',
             p_nom=bus_pow_cap,
             p_nom_min=bus_pow_cap,
             p_nom_extendable=extendable,
             max_hours=max_hours.values,
             capital_cost=capital_cost,
             marginal_cost=marginal_cost,
             efficiency_store=efficiency_store,
             efficiency_dispatch=efficiency_dispatch,
             self_discharge=self_discharge,
             cyclic_state_of_charge=cyclic_sof,
             x=buses_onshore.loc[bus_pow_cap.index].x,
             y=buses_onshore.loc[bus_pow_cap.index].y)

    return net


def add_ror_plants(net: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False) -> pypsa.Network:
    """
    Add run-of-river generators to a Network instance.

    Parameters
    ----------
    net: pypsa.Network
        A Network instance.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable

    Returns
    -------
    net: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    comp_attrs = ["x", "y", "onshore_region"]
    comp_attrs += ["country"] if topology_type == "countries" else []
    for attr in comp_attrs:
        assert hasattr(net.buses, attr), f"Error: Buses must contain a '{attr}' attribute."

    # Hydro generators can only be added onshore
    # buses_onshore = net.buses[net.buses.onshore]
    buses_onshore = net.buses.dropna(subset=["onshore_region"], axis=0)

    # Load capacities and inflows
    aggr_level = "countries" if topology_type == "countries" else "NUTS3"
    pow_cap = get_ror_capacities(aggr_level)
    inflows = get_ror_inflows(aggr_level, net.snapshots)

    if topology_type == 'countries':
        # Extract only countries for which data is available
        countries_with_capacity = sorted(list(set(buses_onshore.country) & set(pow_cap.index)))
        buses_with_capacity_indexes = net.buses[net.buses.country.isin(countries_with_capacity)].index
        bus_pow_cap = pow_cap.loc[countries_with_capacity]
        bus_pow_cap.index = buses_with_capacity_indexes
        bus_inflows = inflows[countries_with_capacity]
        bus_inflows.columns = buses_with_capacity_indexes
    else:  # topology_type == 'ehighway'
        bus_pow_cap, bus_inflows = \
            ror_inputs_nuts_to_ehighway(buses_onshore.index, pow_cap, inflows)
        countries_with_capacity = set(bus_pow_cap.index.str[2:])

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of ROR hydro in {countries_with_capacity}.")

    bus_inflows = bus_inflows.dropna().round(3)

    # Get cost and efficiencies
    capital_cost, marginal_cost = get_costs('ror', len(net.snapshots))
    efficiency = get_tech_info('ror', ["efficiency_ds"])["efficiency_ds"]

    net.madd("Generator",
             bus_pow_cap.index,
             suffix=" Generator ror",
             bus=bus_pow_cap.index,
             type='ror',
             p_nom=bus_pow_cap,
             p_nom_min=bus_pow_cap,
             p_nom_extendable=extendable,
             capital_cost=capital_cost,
             marginal_cost=marginal_cost,
             efficiency=efficiency,
             p_min_pu=0.,
             p_max_pu=bus_inflows,
             x=buses_onshore.loc[bus_pow_cap.index].x,
             y=buses_onshore.loc[bus_pow_cap.index].y)

    return net


def add_sto_plants(net: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """
    Add run-of-river generators to a Network instance

    Parameters
    ----------
    net: pypsa.Network
        A Network instance.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    net: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    comp_attrs = ["x", "y", "onshore_region"]
    comp_attrs += ["country"] if topology_type == "countries" else []
    for attr in comp_attrs:
        assert hasattr(net.buses, attr), f"Error: Buses must contain a '{attr}' attribute."

    # Hydro generators can only be added onshore
    # buses_onshore = net.buses[net.buses.onshore]
    buses_onshore = net.buses.dropna(subset=["onshore_region"], axis=0)

    # Load capacities and inflows
    aggr_level = "countries" if topology_type == "countries" else "NUTS3"
    pow_cap, en_cap = get_sto_capacities(aggr_level)
    inflows = get_sto_inflows(aggr_level, net.snapshots)

    if topology_type == 'countries':
        # Extract only countries for which data is available
        countries_with_capacity = sorted(list(set(buses_onshore.country) & set(pow_cap.index)))
        buses_with_capacity_indexes = net.buses[net.buses.country.isin(countries_with_capacity)].index
        bus_pow_cap = pow_cap.loc[countries_with_capacity]
        bus_pow_cap.index = buses_with_capacity_indexes
        bus_en_cap = en_cap.loc[countries_with_capacity]
        bus_en_cap.index = buses_with_capacity_indexes
        bus_inflows = inflows[countries_with_capacity]
        bus_inflows.columns = buses_with_capacity_indexes
    else:  # topology_type == 'ehighway'
        bus_pow_cap, bus_en_cap, bus_inflows = \
            sto_inputs_nuts_to_ehighway(buses_onshore.index, pow_cap, en_cap, inflows)
        countries_with_capacity = set(bus_pow_cap.index.str[2:])

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of STO hydro "
                f"with {bus_en_cap.sum() * 1e-3:.2f} TWh of storage in {countries_with_capacity}.")
    bus_inflows = bus_inflows.round(3)

    max_hours = bus_en_cap / bus_pow_cap

    capital_cost, marginal_cost = get_costs('sto', len(net.snapshots))

    # Get efficiencies
    efficiency_dispatch = get_tech_info('sto', ['efficiency_ds'])["efficiency_ds"]

    net.madd("StorageUnit",
             bus_pow_cap.index,
             suffix=" Storage reservoir",
             bus=bus_pow_cap.index,
             type='sto',
             p_nom=bus_pow_cap,
             p_nom_min=bus_pow_cap,
             p_min_pu=0.,
             p_nom_extendable=extendable,
             capital_cost=capital_cost,
             marginal_cost=marginal_cost,
             efficiency_store=0.,
             efficiency_dispatch=efficiency_dispatch,
             cyclic_state_of_charge=cyclic_sof,
             max_hours=max_hours,
             inflow=bus_inflows,
             x=buses_onshore.loc[bus_pow_cap.index.values].x,
             y=buses_onshore.loc[bus_pow_cap.index.values].y)

    return net
