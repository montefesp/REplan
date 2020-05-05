from os.path import join, dirname, abspath

import pandas as pd

import pypsa

from src.data.technologies import get_costs, get_plant_type

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators(network: pypsa.Network, tech: str) -> pypsa.Network:
    """
    Add conventional generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with nodes associated to regions.
    tech: str
        Type of conventional generator (ccgt or ocgt)

    Returns
    -------
    network: pypsa.Network
        Updated network
    """
    logger.info(f"Adding {tech} generation.")

    # Filter to keep only onshore buses
    buses = network.buses[network.buses.onshore]

    capital_cost, marginal_cost = get_costs(tech, len(network.snapshots))

    # Get fuel type and efficiency
    tech_info_fn = join(dirname(abspath(__file__)), "../../data/technologies/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    fuel, efficiency = tech_info.loc[get_plant_type(tech)][["fuel", "efficiency_ds"]]

    network.madd("Generator",
                 f"Gen {tech} " + buses.index,
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
