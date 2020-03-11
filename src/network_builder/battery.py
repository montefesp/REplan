from os.path import join, dirname, abspath

import pandas as pd

import pypsa

from src.tech_parameters.costs import get_cost, get_plant_type


def add_batteries(network: pypsa.Network, carrier: str, max_hours: float):
    """
    Adds a battery at each node of the network

    Parameters
    ----------
    network: pypsa.Network
        Pypsa network
    carrier: str
        Type of battery to add
    max_hours: float
        Maximum state of charge capacity in terms of hours at full output capacity

    Returns
    -------
    network: pypsa.Network
        Updated network

    """

    onshore_bus_indexes = pd.Index([bus_id for bus_id in network.buses.index if network.buses.loc[bus_id].onshore])

    # Get costs
    capital_cost, marginal_cost = get_cost(carrier, len(network.snapshots))

    # Get efficiencies
    # TODO: maybe move this to another function
    tech_info_fn = join(dirname(abspath(__file__)), "../tech_parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency_dispatch, efficiency_store, self_discharge = \
        tech_info.loc[get_plant_type(carrier)][["efficiency_ds", "efficiency_ch", "efficiency_sd"]]
    self_discharge = round(1 - self_discharge, 4)

    network.madd("StorageUnit",
                 [f"StorageUnit {carrier} " + str(bus_id) for bus_id in onshore_bus_indexes],
                 carrier=carrier,
                 bus=onshore_bus_indexes,
                 p_nom_extendable=True,
                 max_hours=max_hours,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency_dispatch=efficiency_dispatch,
                 efficiency_store=efficiency_store,
                 self_discharge=self_discharge)

    return network
