from os.path import join, dirname, abspath
from typing import List

import pandas as pd

import pypsa

from src.data.geographics import convert_country_codes, match_points_to_regions
from src.data.generation import get_gen_from_ppm
from src.data.technologies import get_cost, get_plant_type

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def add_generators(network: pypsa.Network, countries: List[str], use_ex_cap: bool, extendable: bool,
                   ppm_file_name: str = None) -> pypsa.Network:
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
    ppm_file_name: str
        Name of the file from which to retrieve the data if value is not None

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # Load existing nuclear plants
    if ppm_file_name is not None:
        ppm_folder = join(dirname(abspath(__file__)), "../../data/generation/source/ppm/")
        gens = pd.read_csv(f"{ppm_folder}/{ppm_file_name}", index_col=0, delimiter=";")
        gens["Country"] = gens["Country"].apply(lambda c: convert_country_codes('alpha_2', name=c))
        gens = gens[gens["Country"].isin(countries)]
    else:
        gens = get_gen_from_ppm(fuel_type="Nuclear", countries=countries)

    onshore_buses = network.buses[network.buses.onshore]
    gens_bus_ds = match_points_to_regions(gens[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values,
                                          onshore_buses.region)
    points = list(gens_bus_ds.index)
    gens = gens[gens[["lon", "lat"]].apply(lambda x: (x[0], x[1]) in points, axis=1)]

    def add_region(lon, lat):
        bus = gens_bus_ds[lon, lat]
        # Need the if because some points are exactly at the same position
        return bus if isinstance(bus, str) else bus.iloc[0]
    gens["bus_id"] = gens[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)

    logger.info(f"Adding {gens['Capacity'].sum()*1e-3:.2f} GW of nuclear capacity in {gens['Country'].unique()}.")

    if not use_ex_cap:
        gens.Capacity = 0.
    gens.Capacity /= 1000.  # Convert MW to GW

    capital_cost, marginal_cost = get_cost('nuclear', len(network.snapshots))

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
