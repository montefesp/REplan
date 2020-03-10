import pandas as pd

import pypsa

from src.tech_parameters.costs import get_cost


def add_batteries(network: pypsa.Network, max_hours: float):

    onshore_bus_indexes = pd.Index([bus_id for bus_id in network.buses.index if network.buses.loc[bus_id].onshore])

    capital_cost, marginal_cost = get_cost('battery', len(network.snapshots))

    network.madd("StorageUnit",
                 ["Battery store " + str(bus_id) for bus_id in onshore_bus_indexes],
                 carrier="battery",
                 bus=onshore_bus_indexes,
                 p_nom_extendable=True,
                 max_hours=max_hours,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost)

    return network
