from os.path import join, abspath, dirname

import pandas as pd

from src.data.geographics.manager import get_subregions, convert_country_codes

# def get_max_emission(max_co2_global_per_year: float, countries: List[str], nb_hours: float) -> float:
#     """Returns an absolute value of maximum CO2 emission in tons for a group of countries over a given number of hours.
#
#     Parameters:
#     -----------
#     max_co2_global_per_year: float
#         Maximum CO2 emission for the whole globe over one year in Gt
#     countries: List[str]
#         List of countries codes
#     nb_hours: float
#         Number of hours
#
#     Returns:
#     --------
#     max_co2: float
#     """
#
#     total_pop: Final = 8.0  # in billion
#
#     max_co2_per_person = (max_co2_global_per_year/total_pop)*(nb_hours/8760)  # in tons
#
#     max_co2 = 0
#     for country in countries:
#         key_ind_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                   "../../../data/key_indicators/source/" + country + ".csv")
#         pop = pd.read_csv(key_ind_fn, index_col=0).loc[2016]["Population (millions)"]*1000000
#         max_co2 += max_co2_per_person*pop
#
#     return max_co2


# def get_max_emission_from_emission_per_mwh(max_emission_per_mwh: float, loads_p: pd.DataFrame):
#
#     return loads_p.values.sum()*max_emission_per_mwh


# def get_reference_emission_levels(region: str, ref_year: int) -> float:
#     """
#     Returns the total CO2 emissions (in kT) emitted by a series of countries in a given region for a given year
#
#     Parameters
#     ----------
#     region: str
#         Region consisting of one or several countries.
#     ref_year: int
#         Year
#
#     Returns
#     -------
#     emission_ref: float
#         Total Co2 emissions in kT
#
#     """
#     # TODO: Shift from readily created file to EEA and IEA source files to have flexibility wrt reference year.
#
#     if ref_year != 1990:
#         raise ValueError(f"Reference year {ref_year} not available yet. Try 1990.")
#
#     emission_fn = join(dirname(abspath(__file__)), "../../../data/emission/generated/EU_28_1990_emission_levels.xlsx")
#     emission_df = pd.read_excel(emission_fn, index_col=0)
#
#     subregions = get_subregions(region)
#
#     emission_subregion = emission_df.reindex(subregions)
#     emission_subregion['kT'] = emission_subregion['Intensity']*emission_subregion['Production']
#
#     emission_ref = emission_subregion['kT'].sum()
#
#     return emission_ref


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
    eea_emission_fn = join(dirname(abspath(__file__)), "../../../data/emission/source/eea/co2-emission-intensity-5.csv")
    eea_emission_df = pd.read_csv(eea_emission_fn, index_col=0, usecols=[0, 1, 4])
    eea_emission_df.columns = ["Country", "co2 (g/kWh)"]
    country_name = convert_country_codes('name', alpha_2=country_code)

    if country_name in set(eea_emission_df["Country"].values) and \
            year in eea_emission_df[eea_emission_df["Country"] == country_name].index:

        co2_intensity = eea_emission_df[eea_emission_df["Country"] == country_name].loc[year, "co2 (g/kWh)"]
        # Multiply by production to obtain total co2 emissions (in kT)
        iea_production_fn = join(dirname(abspath(__file__)), f"../../../data/production/source/iea/{country_code}.csv")
        iea_production_df = pd.read_csv(iea_production_fn, index_col=0)
        return co2_intensity*iea_production_df.loc[year, "Electricity Production (GWh)"]*1e6/1e9
    else:
        # If data for the country is not accessible from EEA, use data from IEA
        iea_emission_fn = join(dirname(abspath(__file__)), f"../../../data/emission/source/iea/{country_code}.csv")
        iea_emission_df = pd.read_csv(iea_emission_fn, index_col=0).dropna()
        co2_emissions = 0.
        if year in iea_emission_df.index:
            co2_emissions = iea_emission_df.loc[year, "CO2 from electricity and heat producers (MT)"]
        else:
            print(country_code)
            print(f"Warning: No available value for {country_code} for year {year}, setting emissions to 0.")
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
    countries = get_subregions(region)
    total_emissions = 0.
    for c in countries:
        total_emissions += get_co2_emission_level_for_country(c, ref_year)

    return total_emissions


if __name__ == '__main__':
    for year in range(1990, 2019):
        print(year)
        print(get_reference_emission_levels_for_region("EU", year))



