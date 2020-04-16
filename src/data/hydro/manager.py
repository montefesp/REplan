from os.path import join, dirname, abspath
from typing import List, Tuple, Union

import pandas as pd
import numpy as np

from src.data.geographics import nuts3_to_nuts2, get_nuts_area
from src.data.topologies.ehighways import get_ehighway_clusters

# TODO: still need to revise the e-highway function, removing the dependencies with regard to e-highway?


def get_hydro_capacities(aggregation_level: str, plant_type: str) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Returns available hydro capacities per NUTS in which it exists.
    If sto or psp, returns power (in GW) and energy (in GWh) capacities
    If ror, returns power capacities (in GW)

    Parameters
    ----------
    aggregation_level: str
        Whether to return the capacities per NUTS2, NUTS0 or per country
    plant_type: str
        One of phs, ror or sto

    Returns
    -------
    (pd.Series, pd.Series) or pd.Series
    """

    accepted_levels = ["NUTS0", "NUTS2", "country"]
    assert aggregation_level in accepted_levels, \
        f"Error: Accepted aggregation levels are {accepted_levels}, received {aggregation_level}"
    accepted_plant_types = ["phs", "ror", "sto"]
    assert plant_type in accepted_plant_types, \
        f"Error: Accepted plant types are {accepted_plant_types}, received {plant_type}"

    hydro_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    nuts_type = "NUTS2" if aggregation_level == "NUTS2" else "NUTS0"
    hydro_capacities = pd.read_csv(f"{hydro_dir}hydro_capacities_per_{nuts_type}.csv", index_col=0)
    # If aggregation level is country, just change index names for UK and EL
    if aggregation_level == "country":
        hydro_capacities.rename(index={'UK': 'GB', 'EL': 'GR'}, inplace=True)

    if plant_type == "sto":
        return hydro_capacities["STO_CAP [GW]"].dropna(), hydro_capacities["STO_EN_CAP [GWh]"].dropna()
    elif plant_type == "phs":
        return hydro_capacities["PSP_CAP [GW]"].dropna(), hydro_capacities["PSP_EN_CAP [GWh]"].dropna()
    else:  # plant_type == "ror"
        return hydro_capacities["ROR_CAP [GW]"].dropna()


def get_hydro_inflows(aggregation_level: str, plant_type: str, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Returns available hydro inflows per NUTS in which it exists
    If sto, returns inflows (in GWh).
    If ror, returns normalized inflows (per unit of installed capacity)
    If 'timestamps' is specified, return just data for those timestamps, otherwise return all available timestamps.

    Parameters
    ----------
    aggregation_level: str
        Whether to return the capacities per NUTS2, NUTS0 or country
    plant_type: str
        One of ror or sto
    timestamps: pd.DatetimeIndex

    Returns
    -------
    nuts_inflows: pd.DataFrame
        DataFrame indexed by timestamps and whose columns are NUTS codes

    """

    accepted_levels = ["NUTS0", "NUTS2", "country"]
    assert aggregation_level in accepted_levels, \
        f"Error: Accepted aggregation levels are {accepted_levels}, received {aggregation_level}"
    accepted_plant_types = ["ror", "sto"]
    assert plant_type in accepted_plant_types, \
        f"Error: Accepted plant types are {accepted_plant_types}, received {plant_type}"

    hydro_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    nuts_type = "NUTS2" if aggregation_level == "NUTS2" else "NUTS0"
    if plant_type == "sto":
        inflows_fn = f"{hydro_dir}hydro_sto_inflow_time_series_per_{nuts_type}_GWh.csv"
    else:  # plant_type == "ror"
        inflows_fn = f"{hydro_dir}hydro_ror_time_series_per_{nuts_type}_pu.csv"
    inflows_df = pd.read_csv(inflows_fn, index_col=0)
    inflows_df.index = pd.DatetimeIndex(inflows_df.index)
    # If aggregation level is country, just change index names for UK and EL
    if aggregation_level == "country":
        inflows_df.rename(columns={'UK': 'GB', 'EL': 'GR'}, inplace=True)

    if timestamps is not None:
        missing_timestamps = set(timestamps) - set(inflows_df.index)
        assert not missing_timestamps, f"Error: Data is not available for timestamps {missing_timestamps}"
        inflows_df = inflows_df.loc[timestamps]

    return inflows_df


# ----- PHS ----- #

def get_phs_capacities(aggregation_level: str) -> Tuple[pd.Series, pd.Series]:
    """Returns available PHS power (in GW) and energy (in GWh) capacities per NUTS or country in which it exists"""
    return get_hydro_capacities(aggregation_level, 'phs')


# TODO: check if this still works now that id dropped NA in nuts2_pow_cap and nuts2_en_cap
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

    # nuts2_pow_cap = nuts2_pow_cap.fillna(0)
    # nuts2_en_cap = nuts2_en_cap.fillna(0)

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


# ----- ROR ----- #

def get_ror_capacities(aggregation_level: str) -> pd.Series:
    """Returns available ROR power capacities (in GW) per NUTS or countries in which it exists"""
    return get_hydro_capacities(aggregation_level, 'ror')


def get_ror_inflows(aggregation_level: str, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """Returns ROR inflows (per unit of installed power capacity) per NUTS or country in which it exists"""
    return get_hydro_inflows(aggregation_level, 'ror', timestamps)


# TODO: check if this still works now that id dropped NA in nuts2_pow_cap and nuts2_en_cap
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
        ROR power capacity (GW) for each NUTS2 region for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        ROR energy inflow (per unit) for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_cap: pd.Series (index: ids of bus for which capacity exists)
        ROR power capacity (GW) for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        ROR energy inflow (per unit) for each bus for which there is installed capacity
    """
    # nuts2_cap = nuts2_cap.fillna(0)

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


# ----- STO ----- #


def get_sto_capacities(aggregation_level: str) -> Tuple[pd.Series, pd.Series]:
    """Returns available STO power (in GW) and energy (in GWh) capacities per NUTS or country in which it exists."""
    return get_hydro_capacities(aggregation_level, 'sto')


def get_sto_inflows(aggregation_level: str, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """Returns STO inflows (in GWh) per NUTS or country in which it exists"""
    return get_hydro_inflows(aggregation_level, 'sto', timestamps)


# TODO: check if this still works now that id dropped NA in nuts2_pow_cap and nuts2_en_cap
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
        STO power capacity (GW) for each NUTS2 region for which there is installed capacity
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity (GWh) for each NUTS2 region for which there is installed capacity
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

    # nuts2_pow_cap = nuts2_pow_cap.fillna(0)
    # nuts2_en_cap = nuts2_en_cap.fillna(0)

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
