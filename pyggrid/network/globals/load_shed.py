from os.path import join, dirname, abspath

import pypsa

import pandas as pd

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_load_shedding(network: pypsa.Network, load_df: pd.DataFrame) -> pypsa.Network:
    """
    Adding dummy-generators for load shedding.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with regions
    load_df: pd.DataFrame
        Frame containing load data.

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    tech_dir = join(dirname(abspath(__file__)), "../../../data/technologies/")
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    onshore_bus_indexes = network.buses[network.buses.onshore].index

    # Get peak load and normalized load profile
    loads_max = load_df.max(axis=0)
    loads_pu = load_df.apply(lambda x: x/ x.max(), axis=0)
    # Add generators for load shedding (prevents the model from being infeasible)

    network.madd("Generator",
                 "Load shed " + onshore_bus_indexes,
                 bus=onshore_bus_indexes,
                 type="load",
                 p_nom=loads_max.values,
                 p_max_pu=loads_pu.values,
                 x=network.buses.loc[onshore_bus_indexes].x.values,
                 y=network.buses.loc[onshore_bus_indexes].y.values,
                 marginal_cost=fuel_info.loc["load", "cost"])

    return network
