from os.path import join, dirname, abspath
from typing import List

import pandas as pd

import pypsa

from src.data.generation import get_powerplant_df
from src.data.technologies import get_costs, get_plant_type

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators(network: pypsa.Network, countries: List[str], use_ex_cap: bool, extendable: bool) -> pypsa.Network:
    """
    Add nuclear generators to a PyPsa Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    countries: List[str]
        Codes of countries over which the network is built
    use_ex_cap: bool
        Whether to consider existing capacity or not
    extendable: bool
        Whether generators are extendable

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Load existing nuclear plants
    onshore_buses = network.buses[network.buses.onshore]

    gens = get_powerplant_df('nuclear', countries, onshore_buses.region)
    gens.rename(columns={'region_code': 'bus_id'}, inplace=True)

    # If no plants in the chosen countries, return directly the network
    if len(gens) == 0:
        return network

    logger.info(f"Adding {gens['Capacity'].sum() * 1e-3:.2f} GW of nuclear capacity in {gens['Country'].unique()}.")

    if not use_ex_cap:
        gens.Capacity = 0.
    gens.Capacity /= 1000.  # Convert MW to GW

    capital_cost, marginal_cost = get_costs('nuclear', len(network.snapshots))

    # Get fuel type, efficiency and ramp rates
    tech_info_fn = join(dirname(abspath(__file__)), "../../data/technologies/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    fuel, efficiency, ramp_rate, base_level = \
        tech_info.loc[get_plant_type('nuclear')][["fuel", "efficiency_ds", "ramp_rate", "base_level"]]

    network.madd("Generator",
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

    return network
