import os
import pandas as pd
from typing import *
from os.path import join, abspath, dirname
from src.data.geographics.manager import get_subregions

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
#                                   "../../../data/key_indicators/generated/" + country + ".csv")
#         pop = pd.read_csv(key_ind_fn, index_col=0).loc[2016]["Population (millions)"]*1000000
#         max_co2 += max_co2_per_person*pop
#
#     return max_co2


# def get_max_emission_from_emission_per_mwh(max_emission_per_mwh: float, loads_p: pd.DataFrame):
#
#     return loads_p.values.sum()*max_emission_per_mwh


def get_reference_emission_levels(region: str, ref_year: int) -> float:
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
    #TODO: Shift from readily created file to EEA and IEA source files to have flexibility wrt reference year.

    if ref_year != 1990:
        raise ValueError('Reference year {} not available yet. Try 1990.'.format(ref_year))

    emission_fn = join(dirname(abspath(__file__)), "../../../data/emission/generated/EU_28_1990_emission_levels.xlsx")
    emission_df = pd.read_excel(emission_fn, index_col=0)

    subregions = get_subregions(region)

    emission_subregion = emission_df.reindex(subregions)
    emission_subregion['kT'] = emission_subregion['Intensity']*emission_subregion['Production']

    emission_ref = emission_subregion['kT'].sum()

    return emission_ref


