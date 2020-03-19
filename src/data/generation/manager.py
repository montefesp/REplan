from typing import List

import pandas as pd
import numpy as np
from shapely.geometry import Point

import pypsa
import powerplantmatching as pm

from src.data.geographics.manager import _get_country


def load_ppm():
    """Load the power plant matching database. Needs to be done only once"""
    pm.powerplants(from_url=True)


def get_gen_from_ppm(fuel_type: str = "", technology: str = "", countries: List[str] = None) -> pd.DataFrame:
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
    countries: List[str]
        List of ISO codes of countries for which we want to obtain plants

    Returns
    -------
    fuel_type_plants: pandas.DataFrame
        Dataframe giving for each generator having the right fuel_type and technology
        ['Volume_Mm3', 'YearCommissioned', 'Duration', 'Set', 'Name', 'projectID', 'Country', 'DamHeight_m', 'Retrofit',
         'Technology', 'Efficiency', 'Capacity' (in MW), 'lat', 'lon', 'Fueltype']
         Note that the Country field is converted to the associated country code
    """

    plants = pm.collection.matched_data()

    if fuel_type != "":
        plants = plants[plants.Fueltype == fuel_type]
    if technology != "":
        plants = plants[plants.Technology == technology]

    # Convert country to code
    def correct_countries(c: str):
        if c == "Macedonia, Republic of":
            return "North Macedonia"
        if c == "Czech Republic":
            return "Czechia"
        return c
    plants["Country"] = plants["Country"].apply(lambda c: correct_countries(c))
    plants["Country"] = plants["Country"].apply(lambda c: _get_country('alpha_2', name=c))

    # Get only plants in countries over which the network is defined
    if countries is not None:
        plants = plants[plants["Country"].isin(countries)]

    return plants


# TODO: this should not be in this file
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

    def correct_countries(c: str):
        if c == "GB":
            return "UK"
    plants["Country"] = plants["Country"].apply(lambda c: correct_countries(c))

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


if __name__ == "__main__":
    all_plants = get_gen_from_ppm(fuel_type="Hydro")
    print(all_plants)
    print(all_plants[all_plants.Technology == 'Reservoir']["Country"])
    print(all_plants[all_plants.Technology == 'Run-Of-River']["Country"])
    print(all_plants[all_plants.Technology == 'Pumped Storage']["Country"])
