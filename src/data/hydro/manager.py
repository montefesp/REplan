from os.path import join, dirname, abspath
from typing import Tuple, Union, List

import pandas as pd

from src.data.topologies.ehighways import get_ehighway_clusters


def get_hydro_capacities(aggregation_level: str, plant_type: str) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Return available hydro capacities per NUTS in which it exists.

    If sto or psp, return power (in GW) and energy (in GWh) capacities
    If ror, return power capacities (in GW)

    Parameters
    ----------
    aggregation_level: str
        Whether to return the capacities per NUTS2, NUTS3 or countries
    plant_type: str
        One of phs, ror or sto

    Returns
    -------
    (pd.Series, pd.Series) or pd.Series
    """

    accepted_levels = ["NUTS3", "NUTS2", "countries"]
    assert aggregation_level in accepted_levels, \
        f"Error: Accepted aggregation levels are {accepted_levels}, received {aggregation_level}"
    accepted_plant_types = ["phs", "ror", "sto"]
    assert plant_type in accepted_plant_types, \
        f"Error: Accepted plant types are {accepted_plant_types}, received {plant_type}"

    hydro_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    hydro_capacities = pd.read_csv(f"{hydro_dir}hydro_capacities_per_{aggregation_level}.csv", index_col=0)
    # If aggregation level is country, just change index names for UK and EL
    if aggregation_level == "countries":
        hydro_capacities.rename(index={'UK': 'GB', 'EL': 'GR'}, inplace=True)

    if plant_type == "sto":
        return hydro_capacities["STO_CAP [GW]"].dropna(), hydro_capacities["STO_EN_CAP [GWh]"].dropna()
    elif plant_type == "phs":
        return hydro_capacities["PSP_CAP [GW]"].dropna(), hydro_capacities["PSP_EN_CAP [GWh]"].dropna()
    else:  # plant_type == "ror"
        return hydro_capacities["ROR_CAP [GW]"].dropna()


def get_hydro_inflows(aggregation_level: str, plant_type: str, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Return available hydro inflows per NUTS in which it exists.

    If sto, return inflows (in GWh).
    If ror, return normalized inflows (per unit of installed capacity).
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

    accepted_levels = ["NUTS3", "NUTS2", "countries"]
    assert aggregation_level in accepted_levels, \
        f"Error: Accepted aggregation levels are {accepted_levels}, received {aggregation_level}"
    accepted_plant_types = ["ror", "sto"]
    assert plant_type in accepted_plant_types, \
        f"Error: Accepted plant types are {accepted_plant_types}, received {plant_type}"

    hydro_dir = join(dirname(abspath(__file__)), "../../../data/hydro/generated/")
    if plant_type == "sto":
        inflows_fn = f"{hydro_dir}hydro_sto_inflow_time_series_per_{aggregation_level}_GWh.csv"
    else:  # plant_type == "ror"
        inflows_fn = f"{hydro_dir}hydro_ror_time_series_per_{aggregation_level}_pu.csv"
    inflows_df = pd.read_csv(inflows_fn, index_col=0)
    inflows_df.index = pd.DatetimeIndex(inflows_df.index)
    # If aggregation level is country, just change index names for UK and EL
    if aggregation_level == "countries":
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


def phs_nuts_to_ehighway(eh_buses: List[str], p_cap: pd.Series, e_cap: pd.Series) -> (pd.Series, pd.Series, set):
    """
    Mapping PHS capacities from NUTS3 to PHS

    Parameters
    ----------
    eh_buses: List[str]
        List of e-highway buses.
    p_cap: pd.Series
        PHS power capacities per NUTS3.
    e_cap: pd.Series
        PHS energy capacities per NUTS3.

    Returns
    -------
    bus_p_cap: pd.Series
        PHS e-highway power capacities.
    bus_e_cap: pd.Series
        PHS e-highway energy capacities.
    # TODO: make no sense to me to return this as argument
    regions_with_cap: set
        Regions (i.e., countries) with capacity.

    """

    eh_clusters = get_ehighway_clusters().loc[eh_buses]

    bus_p_cap = pd.Series(index=eh_buses)
    bus_e_cap = pd.Series(index=eh_buses)

    for eh_bus in eh_buses:

        nuts3_codes = eh_clusters.loc[eh_bus, 'codes'].split(',')

        bus_p_cap.loc[eh_bus] = p_cap.reindex(nuts3_codes).sum()
        bus_e_cap.loc[eh_bus] = e_cap.reindex(nuts3_codes).sum()

    regions_with_cap = set(bus_p_cap.index.str[2:])

    return bus_p_cap, bus_e_cap, regions_with_cap


# ----- ROR ----- #

def get_ror_capacities(aggregation_level: str) -> pd.Series:
    """Returns available ROR power capacities (in GW) per NUTS or countries in which it exists"""
    return get_hydro_capacities(aggregation_level, 'ror')


def get_ror_inflows(aggregation_level: str, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """Returns ROR inflows (per unit of installed power capacity) per NUTS or country in which it exists"""
    return get_hydro_inflows(aggregation_level, 'ror', timestamps)


def ror_inputs_nuts_to_ehighway(eh_buses: List[str], p_cap: pd.Series, inflow_ts: pd.DataFrame) \
        -> (pd.Series, pd.DataFrame):
    """
    Rescale ROR plants inputs from NUTS3 levels to e-highway bus levels.

    Parameters
    ----------
    eh_buses: List[str]
    p_cap: pd.Series (index: nuts3 regions)
        ROR power capacity (GW) for each NUTS3 region for which there is installed capacity
    inflow_ts: pd.DataFrame (index: time, columns: nuts3 regions for which data exists)
        ROR energy inflow (per unit) for each NUTS3 region for which there is installed capacity

    Returns
    -------
    bus_p_cap: pd.Series (index: ids of bus for which capacity exists)
        ROR power capacity (GW) for each bus for which there is installed capacity
    bus_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        ROR energy inflow (per unit) for each bus for which there is installed capacity
    regions_with_cap: set
        Regions (i.e., countries) with capacity.
    """

    eh_clusters = get_ehighway_clusters().loc[eh_buses]

    bus_p_cap = pd.Series(index=eh_buses)
    bus_inflows = pd.DataFrame(index=inflow_ts.index, columns=eh_buses)

    for eh_bus in eh_buses:

        nuts3_codes = eh_clusters.loc[eh_bus, 'codes'].split(',')

        bus_p_cap.loc[eh_bus] = p_cap.reindex(nuts3_codes).sum()
        # ROR capacity factor is taken as the mean across all NUTS3 areas in the same ehighway cluster.
        bus_inflows[eh_bus] = inflow_ts.loc[:, inflow_ts.columns.isin(nuts3_codes)].mean(axis=1)

    regions_with_cap = set(bus_p_cap.index.str[2:])

    return bus_p_cap, bus_inflows, regions_with_cap


# ----- STO ----- #

def get_sto_capacities(aggregation_level: str) -> Tuple[pd.Series, pd.Series]:
    """Returns available STO power (in GW) and energy (in GWh) capacities per NUTS or country in which it exists."""
    return get_hydro_capacities(aggregation_level, 'sto')


def get_sto_inflows(aggregation_level: str, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
    """Returns STO inflows (in GWh) per NUTS or country in which it exists"""
    return get_hydro_inflows(aggregation_level, 'sto', timestamps)


def sto_inputs_nuts_to_ehighway(eh_buses: List[str], p_cap: pd.Series, e_cap: pd.Series,
                                inflow_ts: pd.DataFrame) -> (pd.Series, pd.Series, pd.DataFrame, set):
    """


    Parameters
    ----------
    eh_buses: List[str]
    p_cap: pd.Series (index: nuts3 regions)
        STO power capacity (GW) for each NUTS3 region for which there is installed capacity
    e_cap: pd.Series (index: nuts3 regions)
        STO energy capacity (GWh) for each NUTS3 region for which there is installed capacity
    inflow_ts: pd.DataFrame (index: time, columns: nuts3 regions for which data exists)
        STO energy inflow (GWh) for each NUTS3 region for which there is installed capacity

    Returns
    -------
    bus_p_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity (GW) for each bus for which there is installed capacity
    bus_e_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity (GWh) for each bus for which there is installed capacity
    bus_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        STO energy inflow (GWh) for each bus for which there is installed capacity
    regions_with_cap: set
        Regions (i.e., countries) with STO capacity.
    """

    eh_clusters = get_ehighway_clusters().loc[eh_buses]

    bus_p_cap = pd.Series(index=eh_buses)
    bus_e_cap = pd.Series(index=eh_buses)
    bus_inflows = pd.DataFrame(index=inflow_ts.index, columns=eh_buses)

    for eh_bus in eh_buses:

        nuts3_codes = eh_clusters.loc[eh_bus, 'codes'].split(',')

        # TODO: why use reindex?
        bus_p_cap.loc[eh_bus] = p_cap.reindex(nuts3_codes).sum()
        bus_e_cap.loc[eh_bus] = e_cap.reindex(nuts3_codes).sum()
        # STO inflows computed as sum over one ehighway cluster (as they are expressed in energy units)
        bus_inflows[eh_bus] = inflow_ts.loc[:, inflow_ts.columns.isin(nuts3_codes)].sum(axis=1)

    regions_with_cap = set(bus_p_cap.index.str[2:])

    return bus_p_cap, bus_e_cap, bus_inflows, regions_with_cap


