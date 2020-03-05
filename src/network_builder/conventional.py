from typing import Dict

import pypsa


def add_generators(network: pypsa.Network, tech: str, efficiency: float, costs: Dict[str, float]) -> pypsa.Network:
    """Adds conventional generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with nodes associated to regions.
    tech: str
        Type of conventional generator (ccgt or ocgt)
    efficiency: float
     Efficiency of the technology
    costs: Dict[str, float]
        Contains capex and opex

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Filter to keep only onshore buses
    buses = network.buses[network.buses.onshore]

    network.madd("Generator", "Gen " + tech + " " + buses.index,
                 bus=buses.index,
                 p_nom_extendable=True,
                 type=tech,
                 carrier=tech,
                 efficiency=efficiency,
                 marginal_cost=costs["opex"]/1000.0,
                 capital_cost=costs["capex"]*len(network.snapshots)/(8760*1000.0),
                 x=buses.x.values,
                 y=buses.y.values)

    return network
