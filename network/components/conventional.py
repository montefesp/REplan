import pypsa

from iepy.technologies import get_costs, get_tech_info

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

    assert hasattr(network.buses, "onshore_region"), "Some buses must be associated to an onshore region to add" \
                                                     "conventional generators."

    # Filter to keep only onshore buses
    # buses = network.buses[network.buses.onshore]
    buses = network.buses.dropna(subset=["onshore_region"], axis=0)

    capital_cost, marginal_cost, start_up_cost = get_costs(tech, sum(network.snapshot_weightings['objective']))

    # Get fuel type and efficiency
    fuel, efficiency, ramp_rate, min_up_time, min_down_time = \
        get_tech_info(tech, ["fuel", "efficiency_ds", "ramp_rate", "min_up_time", "min_down_time"])

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
                 start_up_cost=start_up_cost,
                 ramp_limit_up=ramp_rate,
                 ramp_limit_down=ramp_rate,
                 min_up_time=min_up_time,
                 min_down_time=min_down_time,
                 p_min_pu=0.,
                 x=buses.x.values,
                 y=buses.y.values)

    return network
