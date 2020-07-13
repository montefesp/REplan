from typing import List

import pypsa

from src.data.generation import get_powerplants, match_powerplants_to_regions
from src.data.technologies import get_costs, get_info

from warnings import warn
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators(net: pypsa.Network, countries: List[str],
                   use_ex_cap: bool = True, extendable: bool = False) -> pypsa.Network:
    """
    Add nuclear generators to a PyPsa Network instance.

    Parameters
    ----------
    net: pypsa.Network
        A Network instance with nodes associated to parameters: 'onshore' and 'region'.
    countries: List[str]
        Codes of countries over which the network is built
    use_ex_cap: bool (default: True)
        Whether to consider existing capacity or not
    extendable: bool (default: False)
        Whether generators are extendable

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    for attr in ["onshore", "region"]:
        assert hasattr(net.buses, attr), f"Error: Buses must contain a '{attr}' attribute."

    # Nuclear plants can only be added onshore
    onshore_buses = net.buses[net.buses.onshore]
    if len(onshore_buses) == 0:
        warn("Warning: Trying to add nuclear to network without onshore buses.")
        return net

    # TODO: why distance threshold of 50? --> test more in depth
    gens = get_powerplants('nuclear', countries)
    buses_countries = list(onshore_buses.country) if hasattr(onshore_buses, 'country') else None
    gens["bus_id"] = match_powerplants_to_regions(gens, onshore_buses.region,
                                                  shapes_countries=buses_countries, dist_threshold=50.0)

    # If no plants in the chosen countries, return directly the network
    if len(gens) == 0:
        return net

    logger.info(f"Adding {gens['Capacity'].sum() * 1e-3:.2f} GW of nuclear capacity "
                f"in {sorted(gens['ISO2'].unique())}.")

    if not use_ex_cap:
        gens.Capacity = 0.
    gens.Capacity /= 1000.  # Convert MW to GW

    capital_cost, marginal_cost = get_costs('nuclear', len(net.snapshots))

    # Get fuel type, efficiency and ramp rates
    fuel, efficiency, ramp_rate, base_level = get_info('nuclear', ["fuel", "efficiency_ds", "ramp_rate", "base_level"])

    net.madd("Generator",
             "Gen nuclear " + gens.Name + " " + gens.bus_id,
             bus=gens.bus_id.values,
             p_nom=gens.Capacity.values,
             p_nom_min=gens.Capacity.values,
             p_nom_extendable=extendable,
             type='nuclear',
             carrier=fuel,
             efficiency=efficiency,
             marginal_cost=marginal_cost,
             capital_cost=capital_cost,
             ramp_limit_up=ramp_rate,
             ramp_limit_down=ramp_rate,
             p_min_pu=base_level,
             x=gens.lon.values,
             y=gens.lat.values)

    return net
