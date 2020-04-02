from os.path import join, dirname, abspath
from typing import List

import pandas as pd
import numpy as np

import pypsa

from src.data.geographics.manager import nuts3_to_nuts2, get_nuts_area
from src.data.topologies.ehighways import get_ehighway_clusters
from src.parameters.costs import get_cost, get_plant_type

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: need to revise all these functions, either deleting some of them or removing the dependencies with regard
#  to e-highway, moving them in other files

# ----- PHS ----- #

def phs_inputs_nuts_to_countries(countries: List[str], nuts2_pow_cap: pd.Series, nuts2_en_cap: pd.Series) \
        -> (pd.Series, pd.Series):
    """
    This function takes in inputs for PHS plants at the nuts2 levels and computes equivalent inputs at e-highway
    bus levels.

    Parameters
    ----------
    countries: List[str]
        List of ISO codes of countries
    nuts2_pow_cap: pd.Series (index: nuts2 regions)
        STO power capacity (GW) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity (GWh) for each NUTS2 region (if no capacity for a region, value=NaN)

    Returns
    -------
    bus_pow_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity (GW) for each bus for which there is installed capacity
    bus_en_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity (GWh) for each bus for which there is installed capacity
    """

    nuts2_pow_cap = nuts2_pow_cap.fillna(0)
    nuts2_en_cap = nuts2_en_cap.fillna(0)

    bus_pow_cap = pd.Series(index=countries)
    bus_en_cap = pd.Series(index=countries)

    for country in countries:

        def correct_countries(c):
            if c == "GB":
                return "UK"
            if c == "GR":
                return "EL"
            return c

        # If the code corresponds to a country, get the correspond list of NUTS2
        nuts2_codes = [code for code in nuts2_pow_cap.index if code[0:2] == correct_countries(country)]
        # Pow cap (GW)
        bus_pow_cap.loc[country] = np.sum(nuts2_pow_cap.loc[nuts2_codes])
        # En cap (GWh)
        bus_en_cap.loc[country] = np.sum(nuts2_en_cap.loc[nuts2_codes])

    # Remove buses with no capacity
    bus_pow_cap = bus_pow_cap.drop(bus_pow_cap.loc[bus_pow_cap == 0].index)
    bus_en_cap = bus_en_cap.drop(bus_en_cap.loc[bus_en_cap == 0].index)

    return bus_pow_cap, bus_en_cap


def phs_inputs_nuts_to_ehighway(bus_ids: List[str], nuts2_pow_cap: pd.Series, nuts2_en_cap: pd.Series) \
        -> (pd.Series, pd.Series):
    """
    This function takes in inputs for PHS plants at the nuts2 levels and computes equivalent inputs at e-highway
    bus levels.
    All inputs are rescaled based on regions areas.

    Parameters
    ----------
    bus_ids: List[str]
    nuts2_pow_cap: pd.Series (index: nuts2 regions)
        STO power capacity (GW) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity (GWh) for each NUTS2 region (if no capacity for a region, value=NaN)

    Returns
    -------
    bus_pow_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity (GW) for each bus for which there is installed capacity
    bus_en_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity (GWh) for each bus for which there is installed capacity
    """

    nuts2_pow_cap = nuts2_pow_cap.fillna(0)
    nuts2_en_cap = nuts2_en_cap.fillna(0)

    bus_pow_cap = pd.Series(index=bus_ids)
    bus_en_cap = pd.Series(index=bus_ids)

    area_df = get_nuts_area()
    eh_clusters = get_ehighway_clusters()

    for i, bus_id in enumerate(bus_ids):

        # Get e-highway clusters codes (NUTS3 or countries)
        codes = eh_clusters.loc[bus_id].codes.split(",")

        # If the codes are NUTS3
        if len(codes[0]) != 2:

            nuts3_codes = [code for code in codes if code[:4] in nuts2_pow_cap.index]
            # Obtain corresponding NUTS2 codes
            nuts2_codes = [code[:4] for code in nuts3_codes]

            nuts2_area = np.array([area_df.loc[code]["2016"] for code in nuts2_codes])
            nuts3_area = np.array([area_df.loc[code]["2016"] for code in nuts3_codes])
            nuts_area_prop = nuts3_area/nuts2_area

            # Power cap (GW)
            bus_pow_cap.loc[bus_id] = nuts2_pow_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()
            # Energy cap (GWh)
            bus_en_cap.loc[bus_id] = nuts2_en_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()

        else:
            # If the code corresponds to a country, get the correspond list of NUTS2
            nuts2_codes = [code for code in nuts2_pow_cap.index if code[0:2] == codes[0]]
            # Pow cap (GW)
            bus_pow_cap.loc[bus_id] = np.sum(nuts2_pow_cap.loc[nuts2_codes])
            # En cap (GWh)
            bus_en_cap.loc[bus_id] = np.sum(nuts2_en_cap.loc[nuts2_codes])

    # Remove buses with no capacity
    bus_pow_cap = bus_pow_cap.drop(bus_pow_cap.loc[bus_pow_cap == 0].index)
    bus_en_cap = bus_en_cap.drop(bus_en_cap.loc[bus_en_cap == 0].index)

    return bus_pow_cap, bus_en_cap


def add_phs_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """Adds pumped-hydro storage units to a PyPSA Network instance

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    hydro_capacities_fn = join(dirname(abspath(__file__)), "../../data/hydro/generated/hydro_capacities_per_nuts.csv")
    hydro_capacities = pd.read_csv(hydro_capacities_fn, index_col=0, delimiter=";", usecols=[0, 4, 5])

    buses_onshore = network.buses[network.buses.onshore]
    if topology_type == "ehiwghay":
        psp_pow_cap, psp_en_cap = phs_inputs_nuts_to_ehighway(buses_onshore.index,
                                                              hydro_capacities["PSP_CAP [GW]"],
                                                              hydro_capacities["PSP_EN_CAP [GWh]"])
        countries = list(set([i[2:] for i in psp_pow_cap.index]))
    else:  # topology_type == "countries"
        psp_pow_cap, psp_en_cap = phs_inputs_nuts_to_countries(buses_onshore.index,
                                                               hydro_capacities["PSP_CAP [GW]"],
                                                               hydro_capacities["PSP_EN_CAP [GWh]"])
        countries = psp_pow_cap.index.values

    logger.info(f"Adding {psp_pow_cap.sum():.2f} GW of PHS hydro "
                f"with {psp_en_cap.sum():.2f} GWh of storage "
                f"in {countries}.")

    max_hours = psp_en_cap/psp_pow_cap

    capital_cost, marginal_cost = get_cost('phs', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency_dispatch, efficiency_store, self_discharge = \
        tech_info.loc[get_plant_type('phs')][["efficiency_ds", "efficiency_ch", "efficiency_sd"]]
    self_discharge = round(1 - self_discharge, 4)

    network.madd("StorageUnit", "Storage PHS " + psp_pow_cap.index,
                 bus=psp_pow_cap.index,
                 type='phs',
                 p_nom=psp_pow_cap.values,
                 p_nom_min=psp_pow_cap.values,
                 p_nom_extendable=extendable,
                 max_hours=max_hours.values,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency_store=efficiency_store,
                 efficiency_dispatch=efficiency_dispatch,
                 self_discharge=self_discharge,
                 cyclic_state_of_charge=cyclic_sof,
                 x=buses_onshore.loc[psp_pow_cap.index].x.values,
                 y=buses_onshore.loc[psp_pow_cap.index].y.values)

    return network


# ----- ROR ----- #

def ror_inputs_nuts_to_countries(countries: List[str], nuts2_cap: pd.Series, nuts2_inflows: pd.DataFrame) \
        -> (pd.Series, pd.DataFrame):
    """
    This function takes in inputs for ROR plants at the nuts2 levels and computes equivalent inputs at country level.

    Parameters
    ----------
    countries: List[str]
        List of ISO codes of countries
    nuts2_cap: pd.Series (index: nuts2 regions)
        ROR power capacity (GW) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        ROR energy inflow (per unit) for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_cap: pd.Series (index: ids of bus for which capacity exists)
        ROR power capacity (GW) for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        ROR energy inflow (per unit) for each bus for which there is installed capacity
    """
    nuts2_cap = nuts2_cap.fillna(0)

    bus_cap = pd.Series(index=countries)
    bus_inflows = pd.DataFrame(index=nuts2_inflows.index, columns=countries)

    for country in countries:

        def correct_countries(c):
            if c == "GB":
                return "UK"
            if c == "GR":
                return "EL"
            return c

        # If the code corresponds to a country, get the correspond list of NUTS2
        nuts2_codes = [code for code in nuts2_inflows.keys() if code[0:2] == correct_countries(country)]
        if len(nuts2_codes) == 0:
            bus_cap = bus_cap.drop(country)
            bus_inflows = bus_inflows.drop(country, axis=1)
            continue
        bus_cap.loc[country] = nuts2_cap.loc[nuts2_codes].sum()
        nuts2_cap_for_bus = nuts2_cap.loc[nuts2_codes]
        bus_inflows[country] = nuts2_inflows[nuts2_codes].mul(nuts2_cap_for_bus/nuts2_cap_for_bus.sum()).sum(axis=1)

    return bus_cap, bus_inflows


def ror_inputs_nuts_to_ehighway(bus_ids: List[str], nuts2_cap: pd.Series, nuts2_inflows: pd.DataFrame) \
        -> (pd.Series, pd.DataFrame):
    """
    This function takes in inputs for ROR plants at the nuts2 levels and computes equivalent inputs at e-highway
    bus levels.
    Capacity is rescaled based on regions areas and inflows, as capacity factors, are obtained using an weighted
    average of the inflows of the underlying NUTS3 (or NUTS2) regions composing the bus region where the weights
    are the capacities.

    Parameters
    ----------
    bus_ids: List[str]
    nuts2_cap: pd.Series (index: nuts2 regions)
        ROR power capacity (GW) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        ROR energy inflow (per unit) for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_cap: pd.Series (index: ids of bus for which capacity exists)
        ROR power capacity (GW) for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        ROR energy inflow (per unit) for each bus for which there is installed capacity
    """
    nuts2_cap = nuts2_cap.fillna(0)

    bus_cap = pd.Series(index=bus_ids)
    bus_inflows = pd.DataFrame(index=nuts2_inflows.index, columns=bus_ids)

    area_df = get_nuts_area()
    eh_clusters = get_ehighway_clusters()

    for i, bus_id in enumerate(bus_ids):

        # Get e-highway clusters codes (NUTS3 or countries)
        codes = eh_clusters.loc[bus_id].codes.split(",")

        # If the codes are NUTS3
        if len(codes[0]) != 2:

            nuts3_codes = [code for code in codes if code[:4] in nuts2_inflows.keys()]
            if len(nuts3_codes) == 0:
                bus_cap = bus_cap.drop(bus_id)
                bus_inflows = bus_inflows.drop(bus_id, axis=1)
                continue
            nuts2_codes = nuts3_to_nuts2(nuts3_codes)

            nuts2_area = np.array([area_df.loc[code]["2016"] for code in nuts2_codes])
            nuts3_area = np.array([area_df.loc[code]["2016"] for code in nuts3_codes])
            nuts_area_prop = nuts3_area/nuts2_area

            nuts3_cap = nuts2_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0)

            # Power cap (GW)
            bus_cap.loc[bus_id] = nuts3_cap.sum()
            # Inflow, compute an average with weights proportional to capacity
            bus_inflows[bus_id] = nuts2_inflows[nuts2_codes].mul(nuts3_cap/nuts3_cap.sum()).sum(axis=1)

        else:
            # If the code corresponds to a country, get the correspond list of NUTS2
            nuts2_codes = [code for code in nuts2_inflows.keys() if code[0:2] == codes[0]]
            if len(nuts2_codes) == 0:
                bus_cap = bus_cap.drop(bus_id)
                bus_inflows = bus_inflows.drop(bus_id, axis=1)
                continue
            bus_cap.loc[bus_id] = nuts2_cap.loc[nuts2_codes].sum()
            nuts2_cap_for_bus = nuts2_cap.loc[nuts2_codes]
            bus_inflows[bus_id] = nuts2_inflows[nuts2_codes].mul(nuts2_cap_for_bus/nuts2_cap_for_bus.sum()).sum(axis=1)

    return bus_cap, bus_inflows


def add_ror_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    hydro_capacities_fn = join(dirname(abspath(__file__)), "../../data/hydro/generated/hydro_capacities_per_nuts.csv")
    hydro_capacities = pd.read_csv(hydro_capacities_fn, index_col=0, delimiter=";",
                                   usecols=[0, 1])

    ror_inflow_fn = join(dirname(abspath(__file__)),
                         "../../data/hydro/generated/hydro_ror_time_series_per_nuts_pu.csv")
    ror_inflow = pd.read_csv(ror_inflow_fn, index_col=0, delimiter=";")
    ror_inflow.index = pd.DatetimeIndex(ror_inflow.index)
    ror_inflow = ror_inflow.loc[network.snapshots]

    buses_onshore = network.buses[network.buses.onshore]
    if topology_type == "ehighway":
        bus_cap, bus_inflows = ror_inputs_nuts_to_ehighway(buses_onshore.index,
                                                           hydro_capacities["ROR_CAP [GW]"],
                                                           ror_inflow)
        countries = list(set([i[2:] for i in bus_cap.index]))
    else:  # topology_type == "countries"
        bus_cap, bus_inflows = ror_inputs_nuts_to_countries(buses_onshore.index,
                                                            hydro_capacities["ROR_CAP [GW]"],
                                                            ror_inflow)
        countries = bus_cap.index.values

    logger.info(f"Adding {bus_cap.sum():.2f} GW of ROR hydro in {countries}.")

    bus_inflows = bus_inflows.round(2)

    capital_cost, marginal_cost = get_cost('ror', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency = tech_info.loc[get_plant_type('ror')]["efficiency_ds"]

    network.madd("Generator", "Generator ror " + bus_cap.index,
                 bus=bus_cap.index.values,
                 type='ror',
                 p_nom=bus_cap.values,
                 p_nom_min=bus_cap.values,
                 p_nom_extendable=extendable,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency=efficiency,
                 p_max_pu=bus_inflows.values,
                 x=buses_onshore.loc[bus_cap.index].x.values,
                 y=buses_onshore.loc[bus_cap.index].y.values)

    return network


###################
# ----- STO ----- #
###################

def sto_inputs_nuts_to_countries(countries: List[str], nuts2_pow_cap: pd.Series, nuts2_en_cap: pd.Series,
                                 nuts2_inflows: pd.DataFrame) -> (pd.Series, pd.Series, pd.DataFrame):
    """
    This function takes in inputs for STO plants at the nuts2 levels and computes equivalent inputs at country levels.

    Parameters
    ----------
    countries: List[str]
        List of ISO codes of countries
    nuts2_pow_cap: pd.Series (index: nuts2 regions)
        STO power capacity (GW) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity (GWh) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        STO energy inflow (GWh) for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_pow_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity (GW) for each bus for which there is installed capacity
    bus_en_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity (GWh) for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        STO energy inflow (GWh) for each bus for which there is installed capacity
    """

    # TODO: need to know what we do that?
    nuts2_pow_cap = nuts2_pow_cap.fillna(0)
    nuts2_en_cap = nuts2_en_cap.fillna(0)

    bus_pow_cap = pd.Series(index=countries)
    bus_en_cap = pd.Series(index=countries)
    bus_inflows = pd.DataFrame(index=nuts2_inflows.index, columns=countries)

    for country in countries:

        def correct_countries(c):
            if c == "GB":
                return "UK"
            if c == "GR":
                return "EL"
            return c

        # Get the correspond list of NUTS2
        nuts2_codes = [code for code in nuts2_inflows.keys() if code[0:2] == correct_countries(country)]
        # If there is no code in the inflow data file for this bus, drop it
        if len(nuts2_codes) == 0:
            bus_pow_cap = bus_pow_cap.drop(country)
            bus_en_cap = bus_en_cap.drop(country)
            bus_inflows = bus_inflows.drop(country, axis=1)
            continue

        # Pow cap (GW)
        bus_pow_cap.loc[country] = np.sum(nuts2_pow_cap.loc[nuts2_codes])
        # En cap (GWh)
        bus_en_cap.loc[country] = np.sum(nuts2_en_cap.loc[nuts2_codes])
        # Inflow (GWh)
        bus_inflows[country] = nuts2_inflows[nuts2_codes].sum(axis=1)

    return bus_pow_cap, bus_en_cap, bus_inflows


def sto_inputs_nuts_to_ehighway(bus_ids: List[str], nuts2_pow_cap: pd.Series, nuts2_en_cap: pd.Series,
                                nuts2_inflows: pd.DataFrame) -> (pd.Series, pd.Series, pd.DataFrame):
    """
    This function takes in inputs for STO plants at the nuts2 levels and computes equivalent inputs at e-highway
    bus levels.
    All inputs are rescaled based on regions areas.

    Parameters
    ----------
    bus_ids: List[str]
    nuts2_pow_cap: pd.Series (index: nuts2 regions)
        STO power capacity (GW) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity (GWh) for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        STO energy inflow (GWh) for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_pow_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity (GW) for each bus for which there is installed capacity
    bus_en_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity (GWh) for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        STO energy inflow (GWh) for each bus for which there is installed capacity
    """

    nuts2_pow_cap = nuts2_pow_cap.fillna(0)
    nuts2_en_cap = nuts2_en_cap.fillna(0)

    bus_pow_cap = pd.Series(index=bus_ids)
    bus_en_cap = pd.Series(index=bus_ids)
    bus_inflows = pd.DataFrame(index=nuts2_inflows.index, columns=bus_ids)

    area_df = get_nuts_area()
    eh_clusters = get_ehighway_clusters()

    for i, bus_id in enumerate(bus_ids):

        # Get e-highway clusters codes (NUTS3 or countries)
        codes = eh_clusters.loc[bus_id].codes.split(",")

        # If the codes are NUTS3
        if len(codes[0]) != 2:

            # If there is no code in the inflow data file for this bus, drop it
            nuts3_codes = [code for code in codes if code[:4] in nuts2_inflows.keys()]
            if len(nuts3_codes) == 0:
                bus_pow_cap = bus_pow_cap.drop(bus_id)
                bus_en_cap = bus_en_cap.drop(bus_id)
                bus_inflows = bus_inflows.drop(bus_id, axis=1)
                continue
            # Obtain corresponding NUTS2 codes
            nuts2_codes = [code[:4] for code in nuts3_codes]

            nuts2_area = np.array([area_df.loc[code]["2016"] for code in nuts2_codes])
            nuts3_area = np.array([area_df.loc[code]["2016"] for code in nuts3_codes])
            nuts_area_prop = nuts3_area/nuts2_area

            # Power cap (GW)
            bus_pow_cap.loc[bus_id] = nuts2_pow_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()
            # Energy cap (GWh)
            bus_en_cap.loc[bus_id] = nuts2_en_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()
            # Inflow (GWh)
            bus_inflows[bus_id] = nuts2_inflows[nuts2_codes].mul(nuts_area_prop).sum(axis=1)

        else:
            # If the code corresponds to a country, get the correspond list of NUTS2
            nuts2_codes = [code for code in nuts2_inflows.keys() if code[0:2] == codes[0]]
            # If there is no code in the inflow data file for this bus, drop it
            if len(nuts2_codes) == 0:
                bus_pow_cap = bus_pow_cap.drop(bus_id)
                bus_en_cap = bus_en_cap.drop(bus_id)
                bus_inflows = bus_inflows.drop(bus_id, axis=1)
                continue

            # Pow cap (GW)
            bus_pow_cap.loc[bus_id] = np.sum(nuts2_pow_cap.loc[nuts2_codes])
            # En cap (GWh)
            bus_en_cap.loc[bus_id] = np.sum(nuts2_en_cap.loc[nuts2_codes])
            # Inflow (GWh)
            bus_inflows[bus_id] = nuts2_inflows[nuts2_codes].sum(axis=1)

    return bus_pow_cap, bus_en_cap, bus_inflows


def add_sto_plants(network: pypsa.Network, topology_type: str = "countries",
                   extendable: bool = False, cyclic_sof: bool = True) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    topology_type: str
        Can currently be countries (for one node per country topologies) or ehighway (for topologies based on ehighway)
    extendable: bool (default: False)
        Whether generators are extendable
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    accepted_topologies = ["countries", "ehighway"]
    assert topology_type in accepted_topologies, \
        f"Error: Topology type {topology_type} is not one of {accepted_topologies}"

    data_dir = join(dirname(abspath(__file__)), "../../data/hydro/generated/")

    hydro_capacities = pd.read_csv(data_dir + "hydro_capacities_per_nuts.csv",
                                   index_col=0, delimiter=";", usecols=[0, 2, 3])
    reservoir_inflow = pd.read_csv(data_dir + "hydro_sto_inflow_time_series_per_nuts_GWh.csv",
                                   index_col=0, delimiter=";")
    reservoir_inflow.index = pd.DatetimeIndex(reservoir_inflow.index)
    reservoir_inflow = reservoir_inflow.loc[network.snapshots]

    # Convert nuts2 inputs to buses inputs
    buses_onshore = network.buses[network.buses.onshore]
    if topology_type == "ehighway":
        bus_pow_cap, bus_en_cap, bus_inflows = sto_inputs_nuts_to_ehighway(buses_onshore.index,
                                                                           hydro_capacities["STO_CAP [GW]"],
                                                                           hydro_capacities["STO_EN_CAP [GWh]"],
                                                                           reservoir_inflow)
        countries = list(set([i[2:] for i in bus_pow_cap.index]))
    else:  # topology_type == countries
        bus_pow_cap, bus_en_cap, bus_inflows = sto_inputs_nuts_to_countries(buses_onshore.index,
                                                                            hydro_capacities["STO_CAP [GW]"],
                                                                            hydro_capacities["STO_EN_CAP [GWh]"],
                                                                            reservoir_inflow)
        countries = bus_pow_cap.index.values

    logger.info(f"Adding {bus_pow_cap.sum():.2f} GW of STO hydro "
                f"with {bus_en_cap.sum()*1e-3:.2f} TWh of storage "
                f"in {countries}.")
    bus_inflows = bus_inflows.round(3)

    max_hours = bus_en_cap/bus_pow_cap

    capital_cost, marginal_cost = get_cost('sto', len(network.snapshots))

    # Get efficiencies
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    efficiency_dispatch = tech_info.loc[get_plant_type('sto')]["efficiency_ds"]

    network.madd("StorageUnit", "Storage reservoir " + bus_pow_cap.index,
                 bus=bus_pow_cap.index.values,
                 type='sto',
                 p_nom=bus_pow_cap.values,
                 p_nom_min=bus_pow_cap.values,
                 p_nom_extendable=extendable,
                 capital_cost=capital_cost,
                 marginal_cost=marginal_cost,
                 efficiency_store=0.,
                 efficiency_dispatch=efficiency_dispatch,
                 cyclic_state_of_charge=cyclic_sof,
                 max_hours=max_hours.values,
                 inflow=bus_inflows.values,
                 x=buses_onshore.loc[bus_pow_cap.index].x.values,
                 y=buses_onshore.loc[bus_pow_cap.index].y.values)

    return network
