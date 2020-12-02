from typing import List

import pypsa

from network.globals.replace_su_with_store import replace_su_closed_loop
from iepy.technologies import get_costs, get_config_values, get_tech_info

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def add_batteries(network: pypsa.Network, battery_type: str, buses_ids: List[str] = None,
                  fixed_duration: bool = False) -> pypsa.Network:
    """
    Add a battery at each node of the network.

    Parameters
    ----------
    network: pypsa.Network
        PyPSA network
    battery_type: str
        Type of battery to add
    buses_ids: List[str]
        IDs of the buses at which we want to add batteries.
    fixed_duration: bool
        Whether the battery storage is modelled with fixed duration.

    Returns
    -------
    network: pypsa.Network
        Updated network

    """
    logger.info(f"Adding {battery_type} storage.")

    buses = network.buses
    if buses_ids is not None:
        buses = buses.loc[buses_ids]

    # buses = network.buses[network.buses.onshore]
    # onshore_bus_indexes = pd.Index([bus_id for bus_id in buses.index if buses.loc[bus_id].onshore])
    onshore_buses = buses.dropna(subset=["onshore_region"], axis=0)

    # Get costs and efficiencies
    if fixed_duration:

        capital_cost, marginal_cost = get_costs(battery_type, len(network.snapshots))
        efficiency_dispatch, efficiency_store, self_discharge = \
            get_tech_info(battery_type, ["efficiency_ds", "efficiency_ch", "efficiency_sd"])
        self_discharge = round(1 - self_discharge, 4)

        # Get max number of hours of storage
        max_hours = get_config_values(battery_type, ["max_hours"])

        network.madd("StorageUnit",
                     onshore_buses.index,
                     suffix=f" StorageUnit {battery_type}",
                     type=battery_type,
                     bus=onshore_buses.index,
                     p_nom_extendable=True,
                     max_hours=max_hours,
                     capital_cost=capital_cost,
                     marginal_cost=marginal_cost,
                     efficiency_dispatch=efficiency_dispatch,
                     efficiency_store=efficiency_store,
                     standing_loss=self_discharge)

    else:

        battery_type_power = battery_type+'_p'
        battery_type_energy = battery_type+'_e'

        capital_cost, marginal_cost = get_costs(battery_type_power, len(network.snapshots))
        capital_cost_e, marginal_cost_e = get_costs(battery_type_energy, len(network.snapshots))
        efficiency_dispatch, efficiency_store = get_tech_info(battery_type_power, ["efficiency_ds", "efficiency_ch"])
        self_discharge = get_tech_info(battery_type_energy, ["efficiency_sd"]).astype(float)
        self_discharge = round(1 - self_discharge.values[0], 4)

        network.madd("StorageUnit",
                     onshore_buses.index,
                     suffix=f" StorageUnit {battery_type}",
                     type=battery_type,
                     bus=onshore_buses.index,
                     p_nom_extendable=True,
                     capital_cost=capital_cost,
                     marginal_cost=marginal_cost,
                     capital_cost_e=capital_cost_e,
                     marginal_cost_e=marginal_cost_e,
                     efficiency_dispatch=efficiency_dispatch,
                     efficiency_store=efficiency_store,
                     standing_loss=self_discharge)

        storages = network.storage_units.index[network.storage_units.type == battery_type]
        for storage_to_replace in storages:
            replace_su_closed_loop(network, storage_to_replace)

    return network
