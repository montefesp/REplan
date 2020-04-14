import pypsa

from src.parameters.costs import get_cost, get_plant_type
from src.data.hydro.manager import *

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_phs_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """Adds pumped-hydro storage units to a PyPSA Network instance

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
    nuts_type = "NUTS0" if topology_type == "countries" else "NUTS2"
    nuts_pow_cap, nuts_en_cap = get_phs_capacities_per_nuts(nuts_type)

    # Convert them to bus capacities
    if topology_type == "ehighway":

        bus_pow_cap, bus_en_cap = phs_inputs_nuts_to_ehighway(buses_onshore.index, nuts_pow_cap, nuts_en_cap)
        countries = list(set([i[2:] for i in bus_pow_cap.index]))

    else:  # topology_type == "countries":

        bus_pow_cap, bus_en_cap = phs_inputs_nuts_to_countries(buses_onshore.index, nuts_pow_cap, nuts_en_cap)
        countries = bus_pow_cap.index.values

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of PHS hydro "
                f"with {bus_en_cap.sum():.2f} GWh of storage in {countries}.")

    max_hours = bus_en_cap/bus_pow_cap

    capital_cost, marginal_cost = get_cost('phs', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
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
    """Adds run-of-river generators to a Network instance.

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
    nuts_type = "NUTS0" if topology_type == "countries" else "NUTS2"
    nuts_cap = get_ror_capacities_per_nuts(nuts_type)
    nuts_inflows = get_ror_inflows_per_nuts(nuts_type, network.snapshots)

    if topology_type == "ehighway":

        bus_cap, bus_inflows = ror_inputs_nuts_to_ehighway(buses_onshore.index, nuts_cap, nuts_inflows)
        countries = list(set([i[2:] for i in bus_cap.index]))

    else:  # topology_type == "countries"

        bus_cap, bus_inflows = ror_inputs_nuts_to_countries(buses_onshore.index, nuts_cap, nuts_inflows)
        countries = bus_cap.index.values

    logger.info(f"Adding {bus_cap.sum():.2f} GW of ROR hydro in {countries}.")

    bus_inflows = bus_inflows.round(2)

    capital_cost, marginal_cost = get_cost('ror', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency = tech_info.loc[get_plant_type('ror')]["efficiency_ds"]

    network.madd("Generator",
                 "Generator ror " + bus_cap.index,
                 bus=bus_cap.index.values,
                 type='ror',
                 p_nom=bus_cap.values,
                 p_nom_min=bus_cap.values,
                 p_nom_extendable=extendable,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency=efficiency,
                 p_max_pu=bus_inflows.values,
                 x=buses_onshore.loc[bus_cap.index].x.values,
                 y=buses_onshore.loc[bus_cap.index].y.values)

    return network


def add_sto_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance

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
    nuts_type = "NUTS0" if topology_type == "countries" else "NUTS2"
    nuts_pow_cap, nuts_en_cap = get_sto_capacities_per_nuts(nuts_type)
    nuts_inflows = get_sto_inflows_per_nuts(nuts_type, network.snapshots)

    if topology_type == "ehighway":

        bus_pow_cap, bus_en_cap, bus_inflows = \
            sto_inputs_nuts_to_ehighway(buses_onshore.index, nuts_pow_cap, nuts_en_cap, nuts_inflows)
        countries = list(set([i[2:] for i in bus_pow_cap.index]))

    else:  # topology_type == countries

        bus_pow_cap, bus_en_cap, bus_inflows = \
            sto_inputs_nuts_to_countries(buses_onshore.index, nuts_pow_cap, nuts_pow_cap, nuts_inflows)
        countries = bus_pow_cap.index.values

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of STO hydro "
                f"with {bus_en_cap.sum()*1e-3:.2f} TWh of storage in {countries}.")
    bus_inflows = bus_inflows.round(3)

    max_hours = bus_en_cap/bus_pow_cap

    capital_cost, marginal_cost = get_cost('sto', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency_dispatch = tech_info.loc[get_plant_type('sto')]["efficiency_ds"]

    network.madd("StorageUnit",
                 "Storage reservoir " + bus_pow_cap.index,
                 bus=bus_pow_cap.index.values,
                 type='sto',
                 p_nom=bus_pow_cap.values,
                 p_nom_min=bus_pow_cap.values,
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
