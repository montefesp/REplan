import logging

import pypsa

from src.data.hydro import *
from src.data.technologies import get_costs, get_plant_type

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_phs_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """
    Add pumped-hydro storage units to a PyPSA Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    buses_onshore = network.buses[network.buses.onshore]

    # Load capacities
    aggr_level = "countries" if topology_type == "countries" else "NUTS3"
    pow_cap, en_cap = get_phs_capacities(aggr_level)

    if topology_type == 'countries':
        # Extract only countries for which data is available
        nodes_with_capacity = sorted(list(set(buses_onshore.index) & set(pow_cap.index)))
        bus_pow_cap = pow_cap.loc[nodes_with_capacity]
        bus_en_cap = en_cap.loc[nodes_with_capacity]
    else:  # topology_type == 'ehighway
        bus_pow_cap, bus_en_cap, nodes_with_capacity = phs_nuts_to_ehighway(buses_onshore.index, pow_cap, en_cap)

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of PHS hydro "
                f"with {bus_en_cap.sum():.2f} GWh of storage in {nodes_with_capacity}.")

    max_hours = bus_en_cap / bus_pow_cap

    capital_cost, marginal_cost = get_costs('phs', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../../data/technologies/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency_dispatch, efficiency_store, self_discharge = \
        tech_info.loc[get_plant_type('phs')][["efficiency_ds", "efficiency_ch", "efficiency_sd"]]
    self_discharge = round(1 - self_discharge, 4)

    network.madd("StorageUnit",
                 "Storage PHS " + bus_pow_cap.index,
                 bus=bus_pow_cap.index,
                 type='phs',
                 p_nom=bus_pow_cap.values,
                 p_nom_min=bus_pow_cap.values,
                 p_nom_extendable=extendable,
                 max_hours=max_hours.values,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency_store=efficiency_store,
                 efficiency_dispatch=efficiency_dispatch,
                 self_discharge=self_discharge,
                 cyclic_state_of_charge=cyclic_sof,
                 x=buses_onshore.loc[bus_pow_cap.index].x.values,
                 y=buses_onshore.loc[bus_pow_cap.index].y.values)

    return network


def add_ror_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False) -> pypsa.Network:
    """
    Add run-of-river generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    buses_onshore = network.buses[network.buses.onshore]

    # Load capacities and inflows
    aggr_level = "countries" if topology_type == "countries" else "NUTS3"
    pow_cap = get_ror_capacities(aggr_level)
    inflows = get_ror_inflows(aggr_level, network.snapshots)

    if topology_type == 'countries':
        # Extract only countries for which data is available
        nodes_with_capacity = sorted(list(set(buses_onshore.index) & set(pow_cap.index)))
        bus_pow_cap = pow_cap.loc[nodes_with_capacity]
        bus_inflows = inflows[nodes_with_capacity]
    else:  # topology_type == 'ehighway'
        bus_pow_cap, bus_inflows, nodes_with_capacity = \
            ror_inputs_nuts_to_ehighway(buses_onshore.index, pow_cap, inflows)

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of ROR hydro in {nodes_with_capacity}.")

    bus_inflows = bus_inflows.round(3)

    capital_cost, marginal_cost = get_costs('ror', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../../data/technologies/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency = tech_info.loc[get_plant_type('ror')]["efficiency_ds"]

    network.madd("Generator",
                 "Generator ror " + bus_pow_cap.index,
                 bus=bus_pow_cap.index.values,
                 type='ror',
                 p_nom=bus_pow_cap.values,
                 p_nom_min=bus_pow_cap.values,
                 p_nom_extendable=extendable,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency=efficiency,
                 p_min_pu=0.,
                 p_max_pu=bus_inflows.values,
                 x=buses_onshore.loc[bus_pow_cap.index].x.values,
                 y=buses_onshore.loc[bus_pow_cap.index].y.values)

    return network


def add_sto_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """
    Add run-of-river generators to a Network instance

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    buses_onshore = network.buses[network.buses.onshore]

    # Load capacities and inflows
    aggr_level = "countries" if topology_type == "countries" else "NUTS3"
    pow_cap, en_cap = get_sto_capacities(aggr_level)
    inflows = get_sto_inflows(aggr_level, network.snapshots)

    if topology_type == 'countries':
        # Extract only countries for which data is available
        nodes_with_capacity = sorted(list(set(buses_onshore.index) & set(pow_cap.index)))
        bus_pow_cap = pow_cap.loc[nodes_with_capacity]
        bus_en_cap = en_cap.loc[nodes_with_capacity]
        bus_inflows = inflows[nodes_with_capacity]
    else:  # topology_type == 'ehighway'
        bus_pow_cap, bus_en_cap, bus_inflows, nodes_with_capacity = \
            sto_inputs_nuts_to_ehighway(buses_onshore.index, pow_cap, en_cap, inflows)

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of STO hydro "
                f"with {bus_en_cap.sum() * 1e-3:.2f} TWh of storage in {nodes_with_capacity}.")
    bus_inflows = bus_inflows.round(3)

    max_hours = bus_en_cap / bus_pow_cap

    capital_cost, marginal_cost = get_costs('sto', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../../data/technologies/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency_dispatch = tech_info.loc[get_plant_type('sto')]["efficiency_ds"]

    network.madd("StorageUnit",
                 "Storage reservoir " + bus_pow_cap.index,
                 bus=bus_pow_cap.index.values,
                 type='sto',
                 p_nom=bus_pow_cap.values,
                 p_nom_min=bus_pow_cap.values,
                 p_min_pu=0.,
                 p_nom_extendable=extendable,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency_store=0.,
                 efficiency_dispatch=efficiency_dispatch,
                 cyclic_state_of_charge=cyclic_sof,
                 max_hours=max_hours.values,
                 inflow=bus_inflows.values,
                 x=buses_onshore.loc[bus_pow_cap.index].x.values,
                 y=buses_onshore.loc[bus_pow_cap.index].y.values)

    return network
