from os.path import join, dirname, abspath
from typing import Dict

import pandas as pd

import pypsa

from src.parameters.costs import get_cost, get_plant_type


def add_generators(network: pypsa.Network, tech: str) -> pypsa.Network:
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

    capital_cost, marginal_cost = get_cost(tech, len(network.snapshots))

    # Get fuel type and efficiency
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    fuel, efficiency = tech_info.loc[get_plant_type(tech)][["fuel", "efficiency_ds"]]

    # TODO: add efficiencies

    network.madd("Generator", "Gen " + tech + " " + buses.index,
                 bus=buses.index,
                 p_nom_extendable=True,
                 type=tech,
                 carrier=fuel,
                 efficiency=efficiency,
                 marginal_cost=marginal_cost,
                 capital_cost=capital_cost,
                 x=buses.x.values,
                 y=buses.y.values)

    return network
