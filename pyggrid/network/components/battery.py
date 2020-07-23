import pandas as pd

import pypsa

from pyggrid.data.technologies import get_costs, get_config_values, get_tech_info

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def add_batteries(network: pypsa.Network, battery_type: str) -> pypsa.Network:
    """
    Add a battery at each node of the network.

    Parameters
    ----------
    network: pypsa.Network
        Pypsa network
    battery_type: str
        Type of battery to add

    Returns
    -------
    network: pypsa.Network
        Updated network

    """
    logger.info(f"Adding {battery_type} storage.")

    onshore_bus_indexes = pd.Index([bus_id for bus_id in network.buses.index if network.buses.loc[bus_id].onshore])

    # Get costs and efficiencies
    capital_cost, marginal_cost = get_costs(battery_type, len(network.snapshots))
    efficiency_dispatch, efficiency_store, self_discharge = \
        get_tech_info(battery_type, ["efficiency_ds", "efficiency_ch", "efficiency_sd"])
    self_discharge = round(1 - self_discharge, 4)

    # Get max number of hours of storage
    max_hours = get_config_values(battery_type, ["max_hours"])

    network.madd("StorageUnit",
                 onshore_bus_indexes,
                 suffix=f" StorageUnit {battery_type}",
                 type=battery_type,
                 bus=onshore_bus_indexes,
                 p_nom_extendable=True,
                 max_hours=max_hours,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency_dispatch=efficiency_dispatch,
                 efficiency_store=efficiency_store,
                 self_discharge=self_discharge)

    return network
