from typing import List
from os.path import join, dirname, abspath
from os import listdir

import pandas as pd

import powerplantmatching as pm

from src.data.geographics import convert_country_codes


def load_ppm():
    """Load the power plant matching database. Needs to be done only once"""
    pm.powerplants(from_url=True)


def get_gen_from_ppm(fuel_type: str = "", technology: str = "", countries: List[str] = None) -> pd.DataFrame:
    """
    Return information about generator using a certain fuel type and/or technology
     as extracted from power plant matching tool.

    Parameters
    ----------
    fuel_type: str
        One of the generator's fuel type contained in the power plant matching tool
        ['Bioenergy', 'Geothermal', 'Hard Coal', 'Hydro', 'Lignite', 'Natural Gas', 'Nuclear',
        'Oil', 'Other', 'Solar', 'Waste', 'Wind']
    technology: str
        One of the generator's technology contained in the power plant matching tool
        ['Pv', 'Reservoir', 'Offshore', 'OCGT', 'Storage Technologies', 'Run-Of-River',
         'CCGT, 'CCGT, Thermal', 'Steam Turbine', 'Pumped Storage']
    countries: List[str]
        List of ISO codes of countries for which we want to obtain plants
        Available countries:
        ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE',
         'IT', 'LT', 'LU', 'LV', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'SI', 'SK', 'SE']

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
        assert fuel_type in set(plants.Fueltype), f"Error: Fuel type {fuel_type} does not exist."
        plants = plants[plants.Fueltype == fuel_type]
    if technology != "":
        assert technology in set(plants.Technology), f"Error: Technology {technology} does not exist " \
                                                     f"(possibly for fuel type you chose)."
        plants = plants[plants.Technology == technology]

    # Convert country name to ISO code
    def correct_countries(c: str):
        if c == "Macedonia, Republic of":
            return "North Macedonia"
        if c == "Czech Republic":
            return "Czechia"
        return c
    plants["Country"] = plants["Country"].apply(lambda c: correct_countries(c))
    plants["Country"] = convert_country_codes(plants["Country"].values, 'name', 'alpha_2', True)

    # Get only plants in countries over which the network is defined
    if countries is not None:
        plants = plants[plants["Country"].isin(countries)]

    return plants


def get_hydro_production(countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
    """
    Return yearly national hydro-electric production (in GWh) for a set of countries and years.

    Parameters
    ----------
    countries: List[str] (default: None)
        List of ISO codes. If None, returns data for all countries for which it is available.
    years: List[str] (default: None)
        List of years. If None, returns data for all years for which it is available.

    Returns
    -------
    prod_df: pd.DataFrame (index: countries, columns: years)
        National hydro-electric production in (GWh)
    """

    assert countries is None or len(countries) != 0, "Error: List of countries is empty."
    assert years is None or len(years) != 0, "Error: List of years is empty."

    prod_dir = join(dirname(abspath(__file__)), "../../../data/generation/source/")
    # Data from eurostat
    eurostat_fn = f"{prod_dir}eurostat/nrg_ind_peh.xls"
    eurostat_df = pd.read_excel(eurostat_fn, skiprows=12, index_col=0, na_values=":")[:-3]
    eurostat_df.columns = eurostat_df.columns.astype(int)
    eurostat_df.rename(index={"EL": "GR", "UK": "GB"}, inplace=True)

    # Data from IEA
    iea_dir = f"{prod_dir}iea/hydro/"
    iea_df = pd.DataFrame()
    for file in listdir(iea_dir):
        ds = pd.read_csv(f"{iea_dir}{file}", squeeze=True, index_col=0)
        ds.name = file.strip(".csv")
        iea_df = iea_df.append(ds)

    # Merge the two dataset (if the two source contain data for the same country, data from IEA will be kept)
    prod_df = eurostat_df.append(iea_df)
    prod_df = prod_df.loc[~prod_df.index.duplicated(keep='last')]

    # Slice on time
    if years is not None:
        missing_years = set(years) - set(prod_df.columns)
        assert not missing_years, f"Error: Data is not available for any country for years {sorted(list(missing_years))}"
        prod_df = prod_df[years]
        prod_df = prod_df.dropna()

    # Slice on countries
    if countries is not None:
        missing_countries = set(countries) - set(prod_df.index)
        assert not missing_countries, f"Error: Data is not available for countries " \
                                      f"{sorted(list(missing_countries))} for years {years}"
        prod_df = prod_df.loc[countries]

    return prod_df


if __name__ == '__main__':
    if 1:
        df = get_gen_from_ppm(fuel_type='Hydro', technology="Pumped Storage")
        print(len(df))
        print(len(df.dropna(subset=["Duration"])))
        print(df["Duration"])
        print(df.keys())
        exit()
        #print(set(df["Set"]))
        missing_technology_df = df[df["Technology"].apply(lambda x: not isinstance(x, str))]
        #print(df[df["Set"].apply(lambda x: not isinstance(x, str))]["Capacity"].sum())
        print(len(missing_technology_df))
        print(missing_technology_df.iloc[100])

        df = pd.read_csv("/home/utilisateur/Global_Grid/code/py_ggrid/data/hydro/source/pp_fresna_hydro_updated.csv",
                         index_col=0, sep=";")
        print(len(df))
        missing_technology_df = df[df["Technology"].apply(lambda x: not isinstance(x, str))]
        print(len(missing_technology_df))
        #print(df.loc[missing_technology_df.index].dropna(subset=["Technology"])["Capacity"].sum())