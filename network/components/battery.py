from typing import List

import pypsa

from iepy.technologies import get_costs, get_config_values, get_tech_info

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def replace_su_closed_loop(network: pypsa.Network, su_to_replace: str) -> None:
    """
    Convert a PyPSA Storage Unit to a Store with additional components
     so that power and energy can be sized independently.

    Parameters
    ----------
    network: pypsa.Network
        PyPSA network
    su_to_replace: str
        Name of the storage unit to be replaced

    """

    su = network.storage_units.loc[su_to_replace]

    su_short_name = su_to_replace.split(' ')[-1]
    bus_name = f"{su['bus']} {su_short_name}"
    link_1_name = f"{su['bus']} link {su_short_name} to AC"
    link_2_name = f"{su['bus']} link AC to {su_short_name}"
    store_name = f"{bus_name} store"

    # add bus
    network.add("Bus", bus_name)

    # add discharge link
    network.add("Link",
                link_1_name,
                bus0=bus_name,
                bus1=su["bus"],
                capital_cost=su["capital_cost"],
                marginal_cost=su["marginal_cost"],
                p_nom_extendable=su["p_nom_extendable"],
                p_nom=su["p_nom"]/su["efficiency_dispatch"],
                p_nom_min=su["p_nom_min"]/su["efficiency_dispatch"],
                p_nom_max=su["p_nom_max"]/su["efficiency_dispatch"],
                p_max_pu=su["p_max_pu"],
                efficiency=su["efficiency_dispatch"])

    # add charge link
    network.add("Link",
                link_2_name,
                bus1=bus_name,
                bus0=su["bus"],
                p_nom=su["p_nom"],
                p_nom_extendable=su["p_nom_extendable"],
                p_nom_min=su["p_nom_min"],
                p_nom_max=su["p_nom_max"],
                p_max_pu=-su["p_min_pu"],
                efficiency=su["efficiency_store"])

    # add store
    network.add("Store",
                store_name,
                bus=bus_name,
                capital_cost=su["capital_cost_e"],
                marginal_cost=su["marginal_cost_e"],
                e_nom=su["p_nom"]*su["max_hours"],
                e_nom_min=su["p_nom_min"]/su["efficiency_dispatch"]*su["max_hours"],
                e_nom_max=su["p_nom_max"]/su["efficiency_dispatch"]*su["max_hours"],
                e_nom_extendable=su["p_nom_extendable"],
                e_max_pu=1.,
                e_min_pu=0.,
                standing_loss=su["standing_loss"],
                e_cyclic=su['cyclic_state_of_charge'])

    network.remove("StorageUnit", su_to_replace)


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

    # Add batteries with fixed energy-power ratio
    if fixed_duration:

        capital_cost, marginal_cost = get_costs(battery_type, sum(network.snapshot_weightings['objective']))
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

    # Add batteries where energy and power are sized independently
    else:

        battery_type_power = battery_type+'_p'
        battery_type_energy = battery_type+'_e'

        capital_cost, marginal_cost = get_costs(battery_type_power, sum(network.snapshot_weightings['objective']))
        capital_cost_e, marginal_cost_e = get_costs(battery_type_energy, sum(network.snapshot_weightings['objective']))
        efficiency_dispatch, efficiency_store = get_tech_info(battery_type_power, ["efficiency_ds", "efficiency_ch"])
        self_discharge = get_tech_info(battery_type_energy, ["efficiency_sd"]).astype(float)
        self_discharge = round(1 - self_discharge.values[0], 4)
        ctd_ratio = get_config_values(battery_type_power, ["ctd_ratio"])

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
                     standing_loss=self_discharge,
                     ctd_ratio=ctd_ratio)

        storages = network.storage_units.index[network.storage_units.type == battery_type]
        for storage_to_replace in storages:
            replace_su_closed_loop(network, storage_to_replace)

    return network
