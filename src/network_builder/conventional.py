import pypsa

from src.data.technologies import get_costs, get_tech_info

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


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
    fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])

    network.madd("Generator",
                 buses.index,
                 suffix=f" Gen {tech}",
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
