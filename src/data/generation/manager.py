from typing import List

import pandas as pd

import powerplantmatching as pm

from src.data.geographics.manager import convert_country_codes


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
        ['Pv', 'Reservoir', 'Offshore', 'OCGT', 'Storage Technologies', 'Run-Of-River',
         'CCGT', 'CCGT, Thermal', 'Steam Turbine', 'Pumped Storage']
    countries: List[str]
        List of ISO codes of countries for which we want to obtain plants

    Returns
    -------
    fuel_type_plants: pandas.DataFrame
        Dataframe giving for each generator having the right fuel_type and technology
        ['Volume_Mm3', 'YearCommissioned', 'Duration', 'Set', 'Name', 'projectID', 'Country', 'DamHeight_m',
         'Retrofit', 'Technology', 'Efficiency', 'Capacity' (in MW), 'lat', 'lon', 'Fueltype']
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
    plants["Country"] = plants["Country"].apply(lambda c: convert_country_codes('alpha_2', name=c))

    # Get only plants in countries over which the network is defined
    if countries is not None:
        plants = plants[plants["Country"].isin(countries)]

    return plants


if __name__ == "__main__":
    all_plants = get_gen_from_ppm(fuel_type="Hydro")
    print(all_plants)
    print(all_plants[all_plants.Technology == 'Reservoir']["Country"])
    print(all_plants[all_plants.Technology == 'Run-Of-River']["Country"])
    print(all_plants[all_plants.Technology == 'Pumped Storage']["Country"])
