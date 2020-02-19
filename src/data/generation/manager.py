import powerplantmatching as pm
from shapely.geometry import Point
import pandas as pd

from typing import *
import os
from src.network import Network
# from pypsa import Network as pp_Network
import pypsa
import numpy as np
# TODO: need to make all this more generic

from src.data.geographics.manager import nuts3_to_nuts2, get_nuts_area
from src.data.topologies.ehighway import get_ehighway_clusters

# TODO: maybe divide this file into several ones (expl: one for hydro, one for ppm)
#  Try to stick using the files of the folder of the same name

def add_conventional_gen(network: Network, tech: str, costs: Dict[str, float]) -> Network:
    """Adds conventional generators to a Network instance.

    Parameters
    ----------
    network: Network
        A Network instance with nodes associated to regions.
    tech: str
        Type of conventional generator (ccgt or ocgt)
    costs: Dict[str, float]
        Contains capex and opex

    Returns
    -------
    network: Network
        Updated network
    """

    for bus_id in network.buses.id.values:
        attrs = {"bus": [bus_id],
                 "p_nom": [0],
                 "p_nom_extendable": [True],
                 "type": [tech],
                 "marginal_cost": [costs["opex"]],
                 "capital_cost": [costs["capex"]]}
        network.add("generator", ["Gen " + tech + " " + bus_id], attrs)

    return network


def load_ppm():
    """Load the power plant matching database. Needs to be done only once"""
    pm.powerplants(from_url=True)


def get_gen_from_ppm(fuel_type: str = "", technology: str = "") -> pd.DataFrame:
    """Returns information about generator using a certain fuel type and/or technology
     as extracted from power plant matching tool

    Parameters
    ----------
    fuel_type: str
        One of the generator's fuel type contained in the power plant matching tool
        ['Bioenergy', 'Geothermal', 'Hard Coal', 'Hydro', 'Lignite', 'Natural Gas', 'Nuclear',
        'Oil', 'Other', 'Solar', 'Waste', 'Wind']
    technology: str
        One of the generator's technology contained in the power plant matching tool
        ['Pv', 'Reservoir', 'Offshore', 'OCGT', 'Storage Technologies', 'Run-Of-River', 'CCGT', 'CCGT, Thermal',
        'Steam Turbine', 'Pumped Storage']

    Returns
    -------
    fuel_type_plants: pandas.DataFrame
        Dataframe giving for each generator having the right fuel_type and technology
        ['Volume_Mm3', 'YearCommissioned', 'Duration', 'Set', 'Name', 'projectID', 'Country', 'DamHeight_m', 'Retrofit',
         'Technology', 'Efficiency', 'Capacity', 'lat', 'lon', 'Fueltype']
         Note that the Country field is converted to the associated country code
    """

    plants = pm.collection.matched_data()

    if fuel_type != "":
        plants = plants[plants.Fueltype == fuel_type]
    if technology != "":
        plants = plants[plants.Technology == technology]

    # Convert country to code
    countries_codes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/countries-codes.csv")
    countries_code = pd.read_csv(countries_codes_fn)

    def convert_country_name_to_code(country_name):
        if country_name == "Macedonia, Republic of":
            country_name = "Macedonia"
        return countries_code[countries_code.Name == country_name]["Code"].values[0]
    plants["Country"] = plants["Country"].map(convert_country_name_to_code)

    return plants


def add_nuclear_gen(network: Network, costs: Dict[str, float], use_ex_cap: bool, extendable: bool, ramp_rate: float,
                    ppm_file_name: str = None) -> Network:
    """Adds nuclear generators to a Network instance.

    Parameters
    ----------
    network: Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    use_ex_cap: bool
        Whether to consider existing capacity or not # TODO: will probably remove that at some point
    extendable: bool
        Whether generators are extendable
    ramp_rate: float
        Percentage of the total capacity for which the generation can be increased or decreased between two time-steps
    ppm_file_name: str
        Name of the file from which to retrieve the data if value is not None
    Returns
    -------
    network: Network
        Updated network
    """

    # TODO: add the possibility to remove some plants and allow it to built where it doesn't exist

    # Load existing nuclear plants
    ppm_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/ppm/")
    if ppm_file_name is not None:
        gens = pd.read_csv(ppm_folder + "/" + ppm_file_name, index_col=0, delimiter=";")
    else:
        gens = get_gen_from_ppm(fuel_type="Nuclear")

    for idx in gens.index:

        # Get bus id based on region
        bus_id = network.buses.id[[region.contains(Point(gens.loc[idx].lon, gens.loc[idx].lat))
                                   for region in network.buses.region.values]].values

        p_nom = 0
        if use_ex_cap:
            p_nom = gens.loc[idx].Capacity/1000.0  # Transform to GW

        if len(bus_id) != 0:
            attrs = {"bus": [bus_id[0]],
                     "p_nom": [p_nom],
                     "p_nom_extendable": [extendable],
                     "type": ['nuclear'],
                     "marginal_cost": [costs["opex"]],
                     "capital_cost": [costs["capex"]],
                     "ramp_limit_up": [ramp_rate],
                     "ramp_limit_down": [ramp_rate]}
            network.add("generator", ["Gen nuclear " + gens.loc[idx].Name + " " + bus_id[0]], attrs)

    return network


def add_conventional_gen_pypsa(network: pypsa.Network, tech: str, costs: Dict[str, float]) -> pypsa.Network:
    """Adds conventional generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with nodes associated to regions.
    tech: str
        Type of conventional generator (ccgt or ocgt)
    costs: Dict[str, float]
        Contains capex and opex

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Filter to keep only onshore buses
    buses = network.buses[network.buses.onshore]

    network.madd("Generator", "Gen " + tech + " " + buses.index,
                 bus=buses.index,
                 p_nom_extendable=True,
                 type=tech,
                 carrier=tech,
                 marginal_cost=costs["opex"]/1000.0,
                 capital_cost=costs["capex"]*len(network.snapshots)/(8760*1000.0),
                 x=buses.x.values,
                 y=buses.y.values)

    return network


def find_associated_buses_ehighway(plants: pd.DataFrame, net: pypsa.Network):
    """
    Updates the DataFrame by adding a column indicating for each plant the id of the bus in the network it should be
    associated with. Works only for ehighwat topology where each bus id starts with the code of country it is associated
    with.

    Parameters
    ----------
    plants: pd.DataFrame
        DataFrame representing a set of plants containing as columns at least lon (longitude of the plant),
        lat (latitude) and Country (country code of the country where the plant is located).
    net: pypsa.Network
        PyPSA network with a set of bus whose id start with two characters followed by the country code
        # TODO: to make it more generic would probably need to add a field to the buses indicating their country
        # TODO: would maybe need to make it work also when some of this data is missing
    Returns
    -------
    plants: pd.DataFrame
        Updated plants

    """

    # For each plant, we compute the distances to each bus contained in the same country
    dist_to_buses_region_all = \
        {idx: {
            Point(plants.loc[idx].lon, plants.loc[idx].lat).distance(net.buses.loc[bus_id].region): bus_id
            for bus_id in net.buses.index
            if bus_id[2:4] == plants.loc[idx].Country}
            for idx in plants.index}

    # We then associated the id of bus that is closest to the plant.
    plants["bus_id"] = pd.Series(index=plants.index)
    for idx in plants.index:
        dist_to_buses_region = dist_to_buses_region_all[idx]
        if len(dist_to_buses_region) != 0:
            plants.loc[idx, "bus_id"] = dist_to_buses_region[np.min(list(dist_to_buses_region.keys()))]

    # Some plants might not be in the same country as any of the bus, we remove them
    plants = plants.dropna()

    return plants


def add_nuclear_gen_pypsa(network: pypsa.Network, costs: Dict[str, float], use_ex_cap: bool, extendable: bool,
                          ramp_rate: float, ppm_file_name: str = None) -> pypsa.Network:
    """Adds nuclear generators to a PyPsa Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    use_ex_cap: bool
        Whether to consider existing capacity or not # TODO: will probably remove that at some point
    extendable: bool
        Whether generators are extendable
    ramp_rate: float
        Percentage of the total capacity for which the generation can be increased or decreased between two time-steps
    ppm_file_name: str
        Name of the file from which to retrieve the data if value is not None

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # TODO: add the possibility to remove some plants and allow it to built where it doesn't exist

    # Load existing nuclear plants
    if ppm_file_name is not None:
        ppm_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/ppm/")
        gens = pd.read_csv(ppm_folder + "/" + ppm_file_name, index_col=0, delimiter=";")
        # Convert countries code
        countries_codes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "../../../data/countries-codes.csv")
        countries_code = pd.read_csv(countries_codes_fn)

        # TODO: this is repeated in another function
        def convert_country_name_to_code(country_name):
            return countries_code[countries_code.Name == country_name]["Code"].values[0]
        gens["Country"] = gens["Country"].map(convert_country_name_to_code)
    else:
        gens = get_gen_from_ppm(fuel_type="Nuclear")

    gens = find_associated_buses_ehighway(gens, network)

    if not use_ex_cap:
        gens.Capacity = 0.

    network.madd("Generator", "Gen nuclear " + gens.Name + " " + gens.bus_id,
                 bus=gens.bus_id.values,
                 p_nom=gens.Capacity.values,
                 p_nom_min=gens.Capacity.values,
                 p_nom_extendable=extendable,
                 type='nuclear',
                 carrier='nuclear',
                 marginal_cost=costs["opex"]/1000.0,
                 capital_cost=costs["capex"]*len(network.snapshots)/(8760*1000.0),
                 ramp_limit_up=ramp_rate,
                 ramp_limit_down=ramp_rate,
                 x=gens.lon.values,
                 y=gens.lat.values)

    return network


def add_phs_plants_ppm(network: pypsa.Network, costs: Dict[str, float], use_ex_cap: bool = True,
                       extendable: bool = False, efficiency_store: float = 1.0, efficiency_dispatch: float = 1.0,
                       cyclic_sof: bool = True, max_hours: int = 6) -> pypsa.Network:
    """Adds pumped-hydro storage units to a PyPSA Network instance using powerplantmatching

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    use_ex_cap: bool (default: True)
        Whether to consider existing capacity or not # TODO: will probably remove that at some point
    extendable: bool (default: False)
        Whether generators are extendable
    efficiency_store: float (default: 1.0)
        Efficiency at storing between [0., 1.]
    efficiency_dispatch: float (default: 1.0)
        Efficiency at dispatching between [0., 1.]
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component
    max_hours: int (default: 6)
        Maximum state of charge capacity in terms of hours at full output capacity

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Load existing PHS plants
    phs = get_gen_from_ppm(technology="Pumped Storage")
    phs = phs[["Name", "Capacity", "Country", "lon", "lat"]]
    phs = find_associated_buses_ehighway(phs, network)

    if not use_ex_cap:
        phs.Capacity = 0.

    network.madd("StorageUnit", "Storage PHS " + phs.Name + " " + phs.bus_id,
                 bus=phs.bus_id.values,
                 carrier='phs',
                 p_nom=phs.Capacity.values,
                 p_nom_min=phs.Capacity.values,
                 p_nom_extendable=extendable,
                 max_hours=max_hours,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0),
                 efficiency_store=efficiency_store,
                 efficiency_dispatch=efficiency_dispatch,
                 cyclic_state_of_charge=cyclic_sof,
                 x=phs.lon.values,
                 y=phs.lat.values)

    return network


def phs_inputs_nuts_to_eh(bus_ids: List[str], nuts2_pow_cap: pd.Series, nuts2_en_cap: pd.Series) \
        -> (pd.Series, pd.Series):
    """
    This function takes in inputs for PHS plants at the nuts2 levels and computes equivalent inputs at e-highway
    bus levels.
    All inputs are rescaled based on regions areas.

    Parameters
    ----------
    bus_ids: List[str]
    nuts2_pow_cap: pd.Series (index: nuts2 regions)
        STO power capacity for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity for each NUTS2 region (if no capacity for a region, value=NaN)

    Returns
    -------
    bus_pow_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity for each bus for which there is installed capacity
    bus_en_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity for each bus for which there is installed capacity
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

            # Power cap
            bus_pow_cap.loc[bus_id] = nuts2_pow_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()
            # Energy cap
            bus_en_cap.loc[bus_id] = nuts2_en_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()

        else:
            # If the code corresponds to a country, get the correspond list of NUTS2
            nuts2_codes = [code for code in nuts2_pow_cap.index if code[0:2] == codes[0]]
            # Pow cap
            bus_pow_cap.loc[bus_id] = np.sum(nuts2_pow_cap.loc[nuts2_codes])
            # En cap
            bus_en_cap.loc[bus_id] = np.sum(nuts2_en_cap.loc[nuts2_codes])

    # Remove buses with no capacity
    bus_pow_cap = bus_pow_cap.drop(bus_pow_cap.loc[bus_pow_cap == 0].index)
    bus_en_cap = bus_en_cap.drop(bus_en_cap.loc[bus_en_cap == 0].index)

    return bus_pow_cap, bus_en_cap


def add_phs_plants(network: pypsa.Network, costs: Dict[str, float], extendable: bool = False,
                   efficiency_store: float = 1.0, efficiency_dispatch: float = 1.0,
                   cyclic_sof: bool = True) -> pypsa.Network:
    """Adds pumped-hydro storage units to a PyPSA Network instance using other data

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    extendable: bool (default: False)
        Whether generators are extendable
    efficiency_store: float (default: 1.0)
        Efficiency at storing between [0., 1.]
    efficiency_dispatch: float (default: 1.0)
        Efficiency at dispatching between [0., 1.]
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    hydro_capacities_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../../../data/hydro/generated/hydro_capacities_per_nuts.csv")
    hydro_capacities = pd.read_csv(hydro_capacities_fn, index_col=0, delimiter=";",
                                   usecols=[0, 4, 5])

    buses_onshore = network.buses[network.buses.onshore]
    psp_pow_cap, psp_en_cap = phs_inputs_nuts_to_eh(buses_onshore.index,
                                                    hydro_capacities["PSP_CAP [GW]"],
                                                    hydro_capacities["PSP_EN_CAP [GWh]"])
    max_hours = psp_en_cap/psp_pow_cap

    network.madd("StorageUnit", "Storage PHS " + psp_pow_cap.index,
                 bus=psp_pow_cap.index,
                 carrier='phs',
                 p_nom=psp_pow_cap.values*1000,
                 p_nom_min=psp_pow_cap.values*1000,
                 p_nom_extendable=extendable,
                 max_hours=max_hours.values,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0),
                 efficiency_store=efficiency_store,
                 efficiency_dispatch=efficiency_dispatch,
                 cyclic_state_of_charge=cyclic_sof,
                 x=buses_onshore.loc[psp_pow_cap.index].x.values,
                 y=buses_onshore.loc[psp_pow_cap.index].y.values)

    return network


def add_ror_plants_ppm(network: pypsa.Network, costs: Dict[str, float], use_ex_cap: bool = True,
                       extendable: bool = False, efficiency: float = 1.0) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance using powerplantmatching

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    use_ex_cap: bool (default: True)
        Whether to consider existing capacity or not
    extendable: bool (default: False)
        Whether generators are extendable
    efficiency: float (default: 1.0)
        Efficiency at generating power from inflow

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Load existing ror plants
    rors = get_gen_from_ppm(technology="Run-Of-River")
    rors = rors[["Name", "Capacity", "Country", "lon", "lat"]]
    rors = find_associated_buses_ehighway(rors, network)

    if not use_ex_cap:
        rors.Capacity = 0.

    # TODO this is shit
    def get_ror_inflow(net: pypsa.Network, gens):
        return np.array([[1.0] * len(gens.index)] * len(net.snapshots))

    network.madd("Generator", "Generator ror " + rors.Name + " " + rors.index.astype(str) + " " + rors.bus_id,
                 bus=rors.bus_id.values,
                 carrier='ror',
                 p_nom=rors.Capacity.values,
                 p_nom_min=rors.Capacity.values,
                 p_nom_extendable=extendable,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0),
                 efficiency=efficiency,
                 p_max_pu=get_ror_inflow(network, rors),
                 x=rors.lon.values,
                 y=rors.lat.values)

    return network


def ror_inputs_nuts_to_eh(bus_ids: List[str], nuts2_cap: pd.Series, nuts2_inflows: pd.DataFrame) \
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
        ROR power capacity for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        ROR energy inflow (as capacity factors) for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_cap: pd.Series (index: ids of bus for which capacity exists)
        ROR power capacity for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        ROR energy inflow for each bus for which there is installed capacity
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

            # Power cap
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


def add_ror_plants(network: pypsa.Network, costs: Dict[str, float], extendable: bool = False,
                   efficiency: float = 1.0) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    extendable: bool (default: False)
        Whether generators are extendable
    efficiency: float (default: 1.0)
        Efficiency at generating power from inflow

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    hydro_capacities_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../../../data/hydro/generated/hydro_capacities_per_nuts.csv")
    hydro_capacities = pd.read_csv(hydro_capacities_fn, index_col=0, delimiter=";",
                                   usecols=[0, 1])

    ror_inflow_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../../../data/hydro/generated/hydro_ror_time_series_per_nuts_pu.csv")
    ror_inflow = pd.read_csv(ror_inflow_fn, index_col=0, delimiter=";")
    ror_inflow.index = pd.DatetimeIndex(ror_inflow.index)
    ror_inflow = ror_inflow.loc[network.snapshots]

    buses_onshore = network.buses[network.buses.onshore]
    bus_cap, bus_inflows = ror_inputs_nuts_to_eh(buses_onshore.index,
                                                 hydro_capacities["ROR_CAP [GW]"],
                                                 ror_inflow)

    network.madd("Generator", "Generator ror " + bus_cap.index,
                 bus=bus_cap.index.values,
                 carrier='ror',
                 p_nom=bus_cap.values*1000,
                 p_nom_min=bus_cap.values*1000,
                 p_nom_extendable=extendable,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0),
                 efficiency=efficiency,
                 p_max_pu=bus_inflows.values,
                 x=buses_onshore.loc[bus_cap.index].x.values,
                 y=buses_onshore.loc[bus_cap.index].y.values)

    return network


def add_reservoir_plants_ppm(network: pypsa.Network, costs: Dict[str, float], use_ex_cap: bool = True,
                             extendable: bool = False, efficiency_dispatch: float = 1.0, cyclic_sof: bool = True,
                             max_hours: int = 6) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance using power plant matching tool

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    use_ex_cap: bool (default: True)
        Whether to consider existing capacity or not
    extendable: bool (default: False)
        Whether generators are extendable
    efficiency_dispatch: float (default: 1.0)
        Efficiency of dispatch between [0., 1.]
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component
    max_hours: int (default: 6)
        Maximum state of charge capacity in terms of hours at full output capacity

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Load existing reservoir plants
    reservoirs = get_gen_from_ppm(technology="Reservoir")
    reservoirs = reservoirs[["Name", "Capacity", "Country", "lon", "lat"]]
    reservoirs = find_associated_buses_ehighway(reservoirs, network)

    if not use_ex_cap:
        reservoirs.Capacity = 0.

    # TODO: this is shit
    def get_reservoir_inflow(net: pypsa.Network, gens):
        return np.array([[1.0] * len(gens.index)] * len(net.snapshots))

    network.madd("StorageUnit", "Storage reservoir " + reservoirs.Name + " " + " " + reservoirs.bus_id,
                 bus=reservoirs.bus_id.values,
                 carrier='sto',
                 p_nom=reservoirs.Capacity.values,
                 p_nom_min=reservoirs.Capacity.values,
                 p_nom_extendable=extendable,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0),
                 efficiency_store=0.,
                 efficiency_dispatch=efficiency_dispatch,
                 cyclic_state_of_charge=cyclic_sof,
                 max_hours=max_hours,
                 inflow=get_reservoir_inflow(network, reservoirs),
                 x=reservoirs.lon.values,
                 y=reservoirs.lat.values)

    return network


def sto_inputs_nuts_to_eh(bus_ids: List[str], nuts2_pow_cap: pd.Series, nuts2_en_cap: pd.Series,
                          nuts2_inflows: pd.DataFrame) -> (pd.Series, pd.Series, pd.DataFrame):
    """
    This function takes in inputs for STO plants at the nuts2 levels and computes equivalent inputs at e-highway
    bus levels.
    All inputs are rescaled based on regions areas.

    Parameters
    ----------
    bus_ids: List[str]
    nuts2_pow_cap: pd.Series (index: nuts2 regions)
        STO power capacity for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_en_cap: pd.Series (index: nuts2 regions)
        STO energy capacity for each NUTS2 region (if no capacity for a region, value=NaN)
    nuts2_inflows: pd.DataFrame (index: time, columns: nuts2 regions for which data exists)
        STO energy inflow for each NUTS2 region for which there is installed capacity

    Returns
    -------
    bus_pow_cap: pd.Series (index: ids of bus for which capacity exists)
        STO power capacity for each bus for which there is installed capacity
    bus_en_cap: pd.Series (index: ids of bus for which capacity exists)
        STO energy capacity for each bus for which there is installed capacity
    nuts2_inflows: pd.DataFrame (index: time, columns: ids of bus for which capacity exists)
        STO energy inflow for each bus for which there is installed capacity
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

            # Power cap
            bus_pow_cap.loc[bus_id] = nuts2_pow_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()
            # Energy cap
            bus_en_cap.loc[bus_id] = nuts2_en_cap.loc[nuts2_codes].mul(nuts_area_prop, axis=0).sum()
            # Inflow
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

            # Pow cap
            bus_pow_cap.loc[bus_id] = np.sum(nuts2_pow_cap.loc[nuts2_codes])
            # En cap
            bus_en_cap.loc[bus_id] = np.sum(nuts2_en_cap.loc[nuts2_codes])
            # Inflow
            bus_inflows[bus_id] = nuts2_inflows[nuts2_codes].sum(axis=1)

    return bus_pow_cap, bus_en_cap, bus_inflows


def add_sto_plants(network: pypsa.Network, costs: Dict[str, float], extendable: bool = False,
                   efficiency_dispatch: float = 1.0, cyclic_sof: bool = True) -> pypsa.Network:
    """Adds run-of-river generators to a Network instance.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with nodes associated to regions.
    costs: Dict[str, float]
        Contains capex and opex
    extendable: bool (default: False)
        Whether generators are extendable
    efficiency_dispatch: float (default: 1.0)
        Efficiency of dispatch between [0., 1.]
    cyclic_sof: bool (default: True)
        Whether to set to True the cyclic_state_of_charge for the storage_unit component

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/hydro/generated/")

    hydro_capacities = pd.read_csv(data_dir + "hydro_capacities_per_nuts.csv",
                                   index_col=0, delimiter=";", usecols=[0, 2, 3])
    reservoir_inflow = pd.read_csv(data_dir + "hydro_sto_inflow_time_series_per_nuts_GWh.csv",
                                   index_col=0, delimiter=";")
    reservoir_inflow.index = pd.DatetimeIndex(reservoir_inflow.index)
    reservoir_inflow = reservoir_inflow.loc[network.snapshots]

    buses_onshore = network.buses[network.buses.onshore]
    bus_pow_cap, bus_en_cap, bus_inflows = sto_inputs_nuts_to_eh(buses_onshore.index,
                                                                 hydro_capacities["STO_CAP [GW]"],
                                                                 hydro_capacities["STO_EN_CAP [GWh]"],
                                                                 reservoir_inflow)
    max_hours = bus_en_cap/bus_pow_cap

    network.madd("StorageUnit", "Storage reservoir " + bus_pow_cap.index,
                 bus=bus_pow_cap.index.values,
                 carrier='sto',
                 p_nom=bus_pow_cap.values*1000,
                 p_nom_min=bus_pow_cap.values*1000,
                 p_nom_extendable=extendable,
                 capital_cost=costs["capex"] * len(network.snapshots) / (8760 * 1000.0),
                 efficiency_store=0.,
                 efficiency_dispatch=efficiency_dispatch,
                 cyclic_state_of_charge=cyclic_sof,
                 max_hours=max_hours.values,
                 inflow=bus_inflows.values*1000,
                 x=buses_onshore.loc[bus_pow_cap.index].x.values,
                 y=buses_onshore.loc[bus_pow_cap.index].y.values)

    return network

"""
all_plants = pm.collection.matched_data()
all_plants = all_plants[all_plants.Fueltype == 'Hydro']
print(all_plants.keys())
#print(all_plants[all_plants.Technology == 'Reservoir'].Capacity.sum())
print(all_plants[all_plants.Technology == 'Run-Of-River'])
print(all_plants[all_plants.Name == "Uppenborn"])
#print(all_plants[all_plants.Technology == 'Pumped Storage'].Efficiency)
print(set(all_plants.Technology))
"""
