from os.path import join

import pypsa

import pandas as pd

from iepy import data_path

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_load_shedding(net: pypsa.Network, load_df: pd.DataFrame) -> pypsa.Network:
    """
    Adding dummy-generators for load shedding.

    Parameters
    ----------
    net: pypsa.Network
        A Network instance with regions
    load_df: pd.DataFrame
        Frame containing load data.

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    tech_dir = f"{data_path}technologies/"
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    onshore_buses = net.buses.dropna(subset=["onshore_region"], axis=0)

    # Get peak load and normalized load profile
    loads_max = load_df.max(axis=0)
    loads_pu = load_df.apply(lambda x: x/x.max(), axis=0)
    # Add generators for load shedding (prevents the model from being infeasible)

    net.madd("Generator",
             "Load shed " + onshore_buses.index,
             bus=onshore_buses.index,
             type="load",
             p_nom=loads_max.values,
             p_max_pu=loads_pu.values,
             x=net.buses.loc[onshore_buses.index].x.values,
             y=net.buses.loc[onshore_buses.index].y.values,
             marginal_cost=fuel_info.loc["load", "cost"])

    return net
