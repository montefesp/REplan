from typing import Dict, Any

import pandas as pd

import pypsa


def add_batteries(network: pypsa.Network, max_hours: float, costs: Dict[str, Any]):

    onshore_bus_indexes = pd.Index([bus_id for bus_id in network.buses.index if network.buses.loc[bus_id].onshore])

    network.madd("StorageUnit",
                 ["Battery store " + str(bus_id) for bus_id in onshore_bus_indexes],
                 carrier="battery",
                 bus=onshore_bus_indexes,
                 p_nom_extendable=True,
                 max_hours=max_hours,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0))

    return network