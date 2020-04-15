from os.path import join, abspath, dirname

import pandas as pd

from src.data.geographics.manager import get_subregions, convert_country_codes

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def get_co2_emission_level_for_country(country_code: str, year: int) -> float:
    """
    Returns co2 emissions (in kT) from the electricity sector for a given country at a given year.

    Parameters
    ----------
    country_code: str
        ISO codes of country
    year: int
        Year between 1990 and 2018

    Returns
    -------
    float
        kT of CO2 emitted

    """

    assert 1990 <= year <= 2018, "Error: Data is only available for the period 1990-2018"

    # First try to access co2 intensity from EEA database
    eea_emission_fn = join(dirname(abspath(__file__)),
                           "../../../data/emission/source/eea/co2-emission-intensity-5.csv")
    eea_emission_df = pd.read_csv(eea_emission_fn, index_col=0, usecols=[0, 1, 4])
    eea_emission_df.columns = ["Country", "co2 (g/kWh)"]
    country_name = convert_country_codes('name', alpha_2=country_code)

    if country_name in set(eea_emission_df["Country"].values) and \
            year in eea_emission_df[eea_emission_df["Country"] == country_name].index:

        co2_intensity = eea_emission_df[eea_emission_df["Country"] == country_name].loc[year, "co2 (g/kWh)"]
        # Multiply by production to obtain total co2 emissions (in kT)
        iea_production_fn = join(dirname(abspath(__file__)),
                                 f"../../../data/production/source/iea/{country_code}.csv")
        iea_production_df = pd.read_csv(iea_production_fn, index_col=0)
        return co2_intensity*iea_production_df.loc[year, "Electricity Production (GWh)"]*1e6/1e9
    else:
        # If data for the country is not accessible from EEA, use data from IEA
        iea_emission_fn = join(dirname(abspath(__file__)),
                               f"../../../data/emission/source/iea/{country_code}.csv")
        iea_emission_df = pd.read_csv(iea_emission_fn, index_col=0).dropna()
        co2_emissions = 0.
        if year in iea_emission_df.index:
            co2_emissions = iea_emission_df.loc[year, "CO2 from electricity and heat producers (MT)"]
        else:
            logger.info(f"Warning: No available value for {country_code} for year {year}, setting emissions to 0.")
        return co2_emissions*1e3


def get_reference_emission_levels_for_region(region: str, ref_year: int) -> float:
    """
    Returns the total CO2 emissions (in kT) emitted by a series of countries in a given region for a given year

    Parameters
    ----------
    region: str
        Region consisting of one or several countries.
    ref_year: int
        Year

    Returns
    -------
    emission_ref: float
        Total Co2 emissions in kT

    """
    return sum([get_co2_emission_level_for_country(country, ref_year) for country in get_subregions(region)])


if __name__ == '__main__':
    for y in range(1990, 2019):
        print(y)
        print(get_reference_emission_levels_for_region("EU", y))
