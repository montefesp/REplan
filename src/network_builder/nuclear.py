from os.path import join, dirname, abspath
from typing import List

import pandas as pd

import pypsa

from src.data.geographics.manager import _get_country, match_points_to_region
from src.data.generation.manager import get_gen_from_ppm, find_associated_buses_ehighway
from src.parameters.costs import get_cost, get_plant_type


# TODO: this should not depend on e-highway
def add_generators(network: pypsa.Network, countries: List[str], use_ex_cap: bool, extendable: bool,
                   ppm_file_name: str = None) -> pypsa.Network:
    """Adds nuclear generators to a PyPsa Network instance.

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
        ppm_folder = join(dirname(abspath(__file__)), "../../data/ppm/")
        gens = pd.read_csv(ppm_folder + "/" + ppm_file_name, index_col=0, delimiter=";")
        gens["Country"] = gens["Country"].apply(lambda c: _get_country('alpha_2', name=c))
        gens = gens[gens["Country"].isin(countries)]
    else:
        gens = get_gen_from_ppm(fuel_type="Nuclear", countries=countries)

    gens = find_associated_buses_ehighway(gens, network)
    # TODO: this could make the function more generic but for some weird reasons there is a fucking bug happening
    #    in match_points_to_region
    # Round lon and lat
    # gens[["lon", "lat"]] = gens[["lon", "lat"]].round(2)
    # print(gens)
    # onshore_buses = network.buses[network.buses.onshore]
    # match_points_to_region(gens[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values,
    #                              onshore_buses.region)

    if not use_ex_cap:
        gens.Capacity = 0.
    gens.Capacity /= 1000.  # Convert MW to GW

    capital_cost, marginal_cost = get_cost('nuclear', len(network.snapshots))

    # Get fuel type, efficiency and ramp rates
    tech_info_fn = join(dirname(abspath(__file__)), "../parameters/tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])
    fuel, efficiency, ramp_rate, base_level = tech_info.loc[get_plant_type('nuclear')][["fuel", "efficiency_ds", "ramp_rate", "base_level"]]

    network.madd("Generator", "Gen nuclear " + gens.Name + " " + gens.bus_id,
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
